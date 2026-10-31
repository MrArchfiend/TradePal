"""
Trade Verification and Risk Prevention Using Agentic AI
Single-file implementation: Synthetic Data Generator, Multi-Agent System,
Orchestrator, Evaluation Pipeline, and Streamlit Dashboard.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# SECTION 3: GLOBAL CONSTANTS & ENUMS
# ==============================================================================

SEED = 42
N_TRADES = 2000
DATA_DIR = "data"
BASE_DATE = "2026-10-29"          # "today" trade date (T). Previous day = T-1 (2026-10-28)
SETTLEMENT_CYCLE_DAYS = 1         # T+1
T0_SHARE = 0.05                   # 5% of trades settle same day (T+0)
HOLIDAYS = ["2026-11-10"]         # skip in business-day math
CUTOFF_HOUR = 14                  # settlement cut-off 14:00 on settlement_date
TICK_SECONDS = 60                 # replay step
CLEARING_SLA_MIN = 15             # clearing record expected within 15 min of broker trade_time
CUSTODY_SLA_MIN = 120             # custodian record expected within 120 min
PRICE_TOL_ABS = 0.01              # rupees; below this = ROUNDING_DIFF
AMOUNT_TOL_ABS = 1.00             # rupees
FUZZY_MIN_SCORE = 0.80            # fuzzy match acceptance
AUTO_RESOLVE_LIMIT = 1_000_000    # max trade_amount (INR) an agent may auto-fix
AUTO_CONFIDENCE_MIN = 0.90
AUTO_TYPES = ["DUPLICATE", "ROUNDING_DIFF", "DATE_CONVENTION"]   # only these may be auto-resolved
KILL_SWITCH = False               # UI toggle: if True, no auto-resolution at all
OPENING_CASH = 50_000_000         # INR, treasury opening balance
MIN_CASH_BUFFER = 5_000_000       # flag if projected cash falls below this
ANOMALY_Z = 3.5                   # robust z-score threshold
RISK_HIGH = 0.70
RISK_MEDIUM = 0.40

# Enums
SIDES = ["BUY", "SELL"]
BREAK_TYPES = [
    "MISSING_CLEARING", "MISSING_CUSTODY", "QTY_MISMATCH", "PRICE_MISMATCH",
    "DATE_MISMATCH", "DATE_CONVENTION", "SIDE_MISMATCH", "DUPLICATE",
    "ROUNDING_DIFF", "CASH_MISMATCH"
]
SEVERITY = {
    "MISSING_CLEARING": 0.9, "MISSING_CUSTODY": 0.8, "QTY_MISMATCH": 0.9,
    "PRICE_MISMATCH": 0.6, "DATE_MISMATCH": 0.8, "DATE_CONVENTION": 0.2,
    "SIDE_MISMATCH": 1.0, "DUPLICATE": 0.2, "ROUNDING_DIFF": 0.1, "CASH_MISMATCH": 0.7
}
BREAK_STATUS = ["OPEN", "AUTO_RESOLVED", "ESCALATED"]

# Explicit Column Schemas
CANON_COLUMNS = [
    "record_id", "source", "trade_key", "isin", "side", "quantity", "price",
    "amount", "trade_date", "settlement_date", "status", "event_time", "raw_ref"
]
MATCH_COLUMNS = [
    "trade_id", "trade_key", "isin", "side", "quantity", "price", "amount",
    "settlement_date", "trade_time", "clearing_id", "custody_id", "cash_id",
    "clearing_match_level", "custody_match_level", "match_status"
]
BREAK_COLUMNS = [
    "break_id", "trade_id", "trade_key", "break_type", "field",
    "broker_value", "other_value", "other_source", "amount", "severity",
    "detected_time", "status", "detail"
]
RISK_COLUMNS = [
    "trade_id", "isin", "amount", "settlement_date", "cutoff_time", "hours_to_cutoff",
    "break_severity", "time_pressure", "value_norm", "anomaly_score", "risk_score",
    "risk_level", "reason"
]
ACTION_COLUMNS = [
    "action_id", "break_id", "trade_id", "break_type", "decision",
    "policy_checks_passed", "confidence", "action_text", "decided_time"
]
CASH_FORECAST_COLUMNS = [
    "settlement_date", "front_office_net", "treasury_net", "difference",
    "opening_cash", "projected_closing_cash", "shortfall_flag"
]
SEC_FORECAST_COLUMNS = [
    "isin", "symbol", "opening_quantity", "net_settling_quantity",
    "projected_quantity", "short_flag"
]
ALERT_COLUMNS = [
    "alert_id", "type", "key", "message", "severity"
]

def make_dataframe(data: Any, columns: List[str]) -> pd.DataFrame:
    """Safely construct a DataFrame ensuring expected columns are always present."""
    if data is None or (isinstance(data, list) and len(data) == 0):
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(data)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df

# ==============================================================================
# SECTION 14: DATE & STRING HELPERS
# ==============================================================================

def normalise_trade_key(ref: Any) -> Optional[str]:
    """Extract zero-padded 6-digit trade key from any reference format."""
    if ref is None or pd.isna(ref):
        return None
    ref_str = str(ref).strip()
    match = re.search(r"(\d{6})", ref_str)
    if match:
        return match.group(1)
    return None

def is_business_day(d: datetime | pd.Timestamp) -> bool:
    """Check if date is a Monday-Friday and not in HOLIDAYS."""
    ts = pd.to_datetime(d)
    date_str = ts.strftime("%Y-%m-%d")
    return ts.weekday() < 5 and date_str not in HOLIDAYS

def add_business_days(d: datetime | pd.Timestamp, n: int) -> pd.Timestamp:
    """Add n business days skipping weekends and configured holidays."""
    cur = pd.to_datetime(d)
    added = 0
    step = 1 if n >= 0 else -1
    abs_n = abs(n)
    while added < abs_n:
        cur += timedelta(days=step)
        if is_business_day(cur):
            added += 1
    return cur

def previous_business_day(d: datetime | pd.Timestamp) -> pd.Timestamp:
    """Return the immediately preceding business day."""
    return add_business_days(d, -1)

# ==============================================================================
# SECTION 6: CANONICAL MODEL (CONTEXT)
# ==============================================================================

@dataclass
class Context:
    as_of: Optional[pd.Timestamp] = None          # simulated clock
    raw: dict = field(default_factory=dict)       # raw DataFrames: broker, clearing, custodian, cash, master, positions
    canon: Optional[pd.DataFrame] = None          # canonical records of all sources
    matches: Optional[pd.DataFrame] = None        # one row per broker trade
    breaks: Optional[pd.DataFrame] = None         # one row per break
    risk: Optional[pd.DataFrame] = None           # one row per broker trade with risk score
    liquidity: dict = field(default_factory=dict) # {"cash": DataFrame, "securities": DataFrame, "alerts": DataFrame}
    actions: Optional[pd.DataFrame] = None        # auto-resolutions and escalations
    cases: dict = field(default_factory=dict)     # {break_id: case_file_dict}
    agent_log: list = field(default_factory=list) # list of dict rows
    audit: list = field(default_factory=list)     # hash-chained entries
    seen_breaks: set = field(default_factory=set) # track (trade_id, break_type) to prevent duplicates
    kill_switch: bool = KILL_SWITCH
    auto_resolve_limit: float = AUTO_RESOLVE_LIMIT

    def __post_init__(self):
        if self.canon is None:
            self.canon = pd.DataFrame(columns=CANON_COLUMNS)
        if self.matches is None:
            self.matches = pd.DataFrame(columns=MATCH_COLUMNS)
        if self.breaks is None:
            self.breaks = pd.DataFrame(columns=BREAK_COLUMNS)
        if self.risk is None:
            self.risk = pd.DataFrame(columns=RISK_COLUMNS)
        if self.actions is None:
            self.actions = pd.DataFrame(columns=ACTION_COLUMNS)

# ==============================================================================
# SECTION 9: TAMPER-EVIDENT AUDIT TRAIL & LOGGING
# ==============================================================================

def audit_append(ctx: Context, agent: str, event: str, detail: str) -> None:
    """Append a cryptographically chained audit record."""
    prev_hash = ctx.audit[-1]["hash"] if ctx.audit else "GENESIS"
    ts_str = str(ctx.as_of) if ctx.as_of is not None else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "seq": len(ctx.audit) + 1,
        "ts": ts_str,
        "agent": agent,
        "event": event,
        "detail": str(detail),
        "prev_hash": prev_hash
    }
    payload = json.dumps({k: entry[k] for k in sorted(entry) if k != "hash"}, sort_keys=True)
    entry["hash"] = hashlib.sha256(payload.encode()).hexdigest()
    ctx.audit.append(entry)

def log(ctx: Context, agent: str, event: str, detail: str) -> None:
    """Central logging hook that records agent logs and appends to audit trail."""
    ts_str = str(ctx.as_of) if ctx.as_of is not None else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ctx.agent_log.append({
        "ts": ts_str,
        "agent": agent,
        "event": event,
        "detail": detail
    })
    audit_append(ctx, agent, event, detail)

def verify_chain(audit: List[Dict[str, Any]]) -> Tuple[bool, Optional[int]]:
    """Verify hash integrity and linkage of the audit chain."""
    if not audit:
        return True, None
    for i, entry in enumerate(audit):
        expected_prev = "GENESIS" if i == 0 else audit[i - 1]["hash"]
        if entry.get("prev_hash") != expected_prev:
            return False, entry.get("seq", i + 1)
        expected_payload = json.dumps({k: entry[k] for k in sorted(entry) if k != "hash"}, sort_keys=True)
        computed_hash = hashlib.sha256(expected_payload.encode()).hexdigest()
        if entry.get("hash") != computed_hash:
            return False, entry.get("seq", i + 1)
    return True, None

# ==============================================================================
# SECTION 5: SYNTHETIC DATA GENERATOR
# ==============================================================================

def generate_data(seed: int = SEED, n_trades: int = N_TRADES, out_dir: str = DATA_DIR) -> None:
    """Generate realistic synthetic datasets with controlled break injections."""
    os.makedirs(out_dir, exist_ok=True)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # 1. securities_master.csv (20 instruments)
    symbols_data = [
        ("INE002A01018", "RELIANCE", "Energy", 2450.0),
        ("INE467B01029", "TCS", "IT", 3550.0),
        ("INE009A01021", "INFY", "IT", 1480.0),
        ("INE040A01034", "HDFCBANK", "Banking", 1620.0),
        ("INE090A01021", "ICICIBANK", "Banking", 980.0),
        ("INE062A01020", "SBIN", "Banking", 580.0),
        ("INE154A01025", "ITC", "FMCG", 440.0),
        ("INE018A01030", "LT", "Other", 2950.0),
        ("INE238A01034", "AXISBANK", "Banking", 1020.0),
        ("INE237A01028", "KOTAKBANK", "Banking", 1750.0),
        ("INE075A01022", "WIPRO", "IT", 420.0),
        ("INE585B01010", "MARUTI", "Auto", 9600.0),
        ("INE280A01028", "TITAN", "FMCG", 3100.0),
        ("INE213A01029", "ONGC", "Energy", 185.0),
        ("INE733E01010", "NTPC", "Energy", 235.0),
        ("INE522F01014", "COALINDIA", "Energy", 310.0),
        ("INE397D01024", "BHARTIARTL", "Telecom", 920.0),
        ("INE021A01026", "ASIANPAINT", "FMCG", 3150.0),
        ("INE044A01036", "SUNPHARMA", "Pharma", 1120.0),
        ("INE860A01027", "HCLTECH", "IT", 1250.0),
    ]
    master_rows = []
    for isin, sym, sector, ref_p in symbols_data:
        master_rows.append({
            "isin": isin,
            "symbol": sym,
            "sector": sector,
            "lot_size": 1,
            "ref_price": ref_p,
            "avg_daily_volume": int(py_rng.randint(500_000, 5_000_000))
        })
    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(os.path.join(out_dir, "securities_master.csv"), index=False)

    # 2. opening_positions.csv
    pos_rows = []
    for row in master_rows:
        pos_rows.append({
            "isin": row["isin"],
            "opening_quantity": int(py_rng.randint(15_000, 60_000))
        })
    pos_df = pd.DataFrame(pos_rows)
    pos_df.to_csv(os.path.join(out_dir, "opening_positions.csv"), index=False)

    # 3. broker_trades.csv
    base_t = pd.to_datetime(BASE_DATE)
    prev_t = previous_business_day(base_t)
    counterparties = [
        "Alpha Securities", "Beta Capital", "Gamma Brokers", "Delta Global",
        "Epsilon Prime", "Zeta Institutional", "Eta Financial", "Theta Markets"
    ]

    broker_trades = []
    trade_keys = []
    for i in range(1, n_trades + 1):
        trade_key = f"{i:06d}"
        trade_keys.append(trade_key)
        trade_id = f"TRD-{trade_key}"
        order_id = f"ORD-{py_rng.randint(100_000, 999_999)}"
        sec = py_rng.choice(master_rows)
        isin = sec["isin"]
        symbol = sec["symbol"]
        side = py_rng.choice(SIDES)

        if py_rng.random() < 0.005:
            quantity = int(py_rng.randint(15_000, 40_000))  # Extreme anomaly
        else:
            quantity = int(min(max(10, np_rng.lognormal(mean=5.2, sigma=0.8)), 5000))

        price = round(float(sec["ref_price"] * py_rng.uniform(0.98, 1.02)), 2)
        trade_amount = round(quantity * price, 2)

        t_date = prev_t if py_rng.random() < 0.40 else base_t
        t_date_str = t_date.strftime("%Y-%m-%d")

        sec_offset = py_rng.randint(0, 6 * 3600 + 15 * 60)
        start_secs = 9 * 3600 + 15 * 60
        trade_time_dt = pd.to_datetime(t_date_str) + timedelta(seconds=start_secs + sec_offset)
        trade_time_str = trade_time_dt.strftime("%Y-%m-%d %H:%M:%S")

        if py_rng.random() < T0_SHARE:
            settle_dt = t_date
        else:
            settle_dt = add_business_days(t_date, SETTLEMENT_CYCLE_DAYS)
        settle_date_str = settle_dt.strftime("%Y-%m-%d")

        broker_trades.append({
            "trade_id": trade_id,
            "order_id": order_id,
            "isin": isin,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "trade_amount": trade_amount,
            "currency": "INR",
            "trade_date": t_date_str,
            "trade_time": trade_time_str,
            "settlement_date": settle_date_str,
            "counterparty": py_rng.choice(counterparties),
            "client_id": f"CL-{py_rng.randint(101, 150):03d}",
            "broker_id": f"BRK-{py_rng.randint(1, 3):02d}",
            "trade_key": trade_key
        })

    broker_df = pd.DataFrame(broker_trades)

    # 4. Injected Breaks Planning (12% of trades = 240 trades)
    target_breaks = int(round(n_trades * 0.12))
    distribution = {
        "MISSING_CLEARING": 12,
        "MISSING_CUSTODY": 10,
        "QTY_MISMATCH": 14,
        "PRICE_MISMATCH": 12,
        "DATE_MISMATCH": 10,
        "DATE_CONVENTION": 8,
        "SIDE_MISMATCH": 6,
        "DUPLICATE": 10,
        "ROUNDING_DIFF": 10,
        "CASH_MISMATCH": 8
    }
    break_type_pool: List[str] = []
    for b_type, weight in distribution.items():
        count = int(round(target_breaks * (weight / 100.0)))
        break_type_pool.extend([b_type] * count)

    py_rng.shuffle(break_type_pool)
    break_type_pool = break_type_pool[:target_breaks]

    all_indices = list(range(n_trades))
    py_rng.shuffle(all_indices)
    break_indices = all_indices[:len(break_type_pool)]
    trade_break_map: Dict[int, str] = {idx: b_type for idx, b_type in zip(break_indices, break_type_pool)}

    ground_truth_rows = []
    clearing_rows = []
    custodian_rows = []
    cash_rows = []

    fuzzy_trade_indices = set(py_rng.sample(
        [i for i in range(n_trades) if i not in trade_break_map],
        int(n_trades * 0.02)
    ))

    ccp_counter = 10_000_000
    cus_counter = 20_000_000
    csh_counter = 1

    for idx, b_trade in enumerate(broker_trades):
        b_type = trade_break_map.get(idx)
        trade_id = b_trade["trade_id"]
        trade_key = b_trade["trade_key"]
        t_time = pd.to_datetime(b_trade["trade_time"])
        s_date = pd.to_datetime(b_trade["settlement_date"])

        clr_ref = f"CCP{ccp_counter}"
        ccp_counter += 1
        clr_trade_ref = f"T{trade_key}"
        if idx in fuzzy_trade_indices:
            clr_trade_ref = "" if py_rng.random() < 0.5 else "INVALID_REF"
        clr_isin = b_trade["isin"]
        clr_side = b_trade["side"]
        clr_qty = b_trade["quantity"]
        clr_price = b_trade["price"]
        clr_t_date = b_trade["trade_date"]
        clr_s_date = b_trade["settlement_date"]
        clr_status = "NOVATED" if py_rng.random() < 0.90 else "PENDING"
        clr_rec_time = t_time + timedelta(seconds=py_rng.randint(20, 600))

        cus_ref = f"CUS{cus_counter}"
        cus_counter += 1
        cus_trade_ref = f"TRD{trade_key}"
        cus_isin = b_trade["isin"]
        cus_instruction = "RVP" if b_trade["side"] == "BUY" else "DVP"
        cus_qty = b_trade["quantity"]
        cus_s_date = b_trade["settlement_date"]
        cus_amount = b_trade["trade_amount"]
        cus_status = "MATCHED" if py_rng.random() < 0.80 else "PENDING"
        cus_rec_time = t_time + timedelta(minutes=py_rng.randint(5, 90))

        csh_entry_id = f"CSH-{csh_counter:06d}"
        csh_counter += 1
        csh_trade_ref = trade_id
        csh_val_date = b_trade["settlement_date"]
        csh_dir = "DEBIT" if b_trade["side"] == "BUY" else "CREDIT"
        csh_amount = b_trade["trade_amount"]
        csh_posted_time = t_time + timedelta(hours=float(py_rng.uniform(1.0, 3.0)))

        include_clearing = True
        include_custody = True
        duplicate_clearing = None
        injected_time = None
        correct_action = "ESCALATE"

        if b_type == "MISSING_CLEARING":
            include_clearing = False
            injected_time = t_time + timedelta(minutes=CLEARING_SLA_MIN)
            correct_action = "ESCALATE"

        elif b_type == "MISSING_CUSTODY":
            include_custody = False
            injected_time = t_time + timedelta(minutes=CUSTODY_SLA_MIN)
            correct_action = "ESCALATE"

        elif b_type == "QTY_MISMATCH":
            pct_diff = py_rng.uniform(0.01, 0.20) * py_rng.choice([-1, 1])
            diff_qty = int(round(b_trade["quantity"] * pct_diff))
            if diff_qty == 0:
                diff_qty = 1 if pct_diff > 0 else -1
            clr_qty = max(1, b_trade["quantity"] + diff_qty)
            injected_time = clr_rec_time
            correct_action = "ESCALATE"

        elif b_type == "PRICE_MISMATCH":
            pct_diff = py_rng.uniform(0.005, 0.03) * py_rng.choice([-1, 1])
            diff_price = round(b_trade["price"] * pct_diff, 2)
            if abs(diff_price) <= PRICE_TOL_ABS:
                diff_price = 0.05 if pct_diff > 0 else -0.05
            clr_price = round(b_trade["price"] + diff_price, 2)
            injected_time = clr_rec_time
            correct_action = "ESCALATE"

        elif b_type == "DATE_MISMATCH":
            shift_days = py_rng.choice([1, 2])
            shifted = add_business_days(s_date, shift_days)
            cus_s_date = shifted.strftime("%Y-%m-%d")
            injected_time = cus_rec_time
            correct_action = "ESCALATE"

        elif b_type == "DATE_CONVENTION":
            shifted = add_business_days(s_date, 1)
            clr_s_date = shifted.strftime("%Y-%m-%d")
            injected_time = clr_rec_time
            correct_action = "AUTO_FIX" if b_trade["trade_amount"] <= AUTO_RESOLVE_LIMIT else "ESCALATE"

        elif b_type == "SIDE_MISMATCH":
            clr_side = "SELL" if b_trade["side"] == "BUY" else "BUY"
            injected_time = clr_rec_time
            correct_action = "ESCALATE"

        elif b_type == "DUPLICATE":
            injected_time = clr_rec_time + timedelta(seconds=py_rng.randint(30, 300))
            duplicate_clearing = {
                "clearing_ref": f"CCP{ccp_counter}",
                "trade_ref": clr_trade_ref,
                "isin": clr_isin,
                "side": clr_side,
                "quantity": clr_qty,
                "price": clr_price,
                "trade_date": clr_t_date,
                "settlement_date": clr_s_date,
                "clearing_member": f"CM-{py_rng.randint(1, 3):02d}",
                "status": clr_status,
                "received_time": injected_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            ccp_counter += 1
            correct_action = "AUTO_FIX" if b_trade["trade_amount"] <= AUTO_RESOLVE_LIMIT else "ESCALATE"

        elif b_type == "ROUNDING_DIFF":
            if py_rng.random() < 0.5:
                clr_price = round(clr_price + py_rng.choice([-0.01, 0.01]), 2)
                injected_time = clr_rec_time
            else:
                cus_amount = round(cus_amount + py_rng.uniform(-0.99, 0.99), 2)
                injected_time = cus_rec_time
            correct_action = "AUTO_FIX" if b_trade["trade_amount"] <= AUTO_RESOLVE_LIMIT else "ESCALATE"

        elif b_type == "CASH_MISMATCH":
            pct = py_rng.uniform(0.01, 0.05) * py_rng.choice([-1, 1])
            csh_amount = round(csh_amount * (1.0 + pct), 2)
            injected_time = csh_posted_time
            correct_action = "ESCALATE"

        if b_type is not None:
            ground_truth_rows.append({
                "trade_id": trade_id,
                "break_type": b_type,
                "injected_time": injected_time.strftime("%Y-%m-%d %H:%M:%S"),
                "correct_action": correct_action
            })

        if include_clearing:
            clearing_rows.append({
                "clearing_ref": clr_ref,
                "trade_ref": clr_trade_ref,
                "isin": clr_isin,
                "side": clr_side,
                "quantity": clr_qty,
                "price": clr_price,
                "trade_date": clr_t_date,
                "settlement_date": clr_s_date,
                "clearing_member": f"CM-{py_rng.randint(1, 3):02d}",
                "status": clr_status,
                "received_time": clr_rec_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            if duplicate_clearing:
                clearing_rows.append(duplicate_clearing)

        if include_custody:
            custodian_rows.append({
                "custodian_ref": cus_ref,
                "trade_ref": cus_trade_ref,
                "isin": cus_isin,
                "instruction_type": cus_instruction,
                "quantity": cus_qty,
                "settlement_date": cus_s_date,
                "settlement_amount": cus_amount,
                "depository_status": cus_status,
                "received_time": cus_rec_time.strftime("%Y-%m-%d %H:%M:%S")
            })

        cash_rows.append({
            "entry_id": csh_entry_id,
            "trade_ref": csh_trade_ref,
            "value_date": csh_val_date,
            "direction": csh_dir,
            "amount": csh_amount,
            "currency": "INR",
            "entry_type": "TRADE_SETTLEMENT",
            "posted_time": csh_posted_time.strftime("%Y-%m-%d %H:%M:%S")
        })

    clean_broker = [{k: v for k, v in row.items() if k != "trade_key"} for row in broker_trades]
    pd.DataFrame(clean_broker).to_csv(os.path.join(out_dir, "broker_trades.csv"), index=False)
    pd.DataFrame(clearing_rows).to_csv(os.path.join(out_dir, "clearing_records.csv"), index=False)
    pd.DataFrame(custodian_rows).to_csv(os.path.join(out_dir, "custodian_records.csv"), index=False)
    pd.DataFrame(cash_rows).to_csv(os.path.join(out_dir, "cash_ledger.csv"), index=False)
    truth_df = pd.DataFrame(ground_truth_rows)
    truth_df.to_csv(os.path.join(out_dir, "ground_truth.csv"), index=False)

    print(f"Generated synthetic dataset in '{out_dir}/':")
    print(f"  • Broker trades:      {len(clean_broker)}")
    print(f"  • Clearing records:   {len(clearing_rows)}")
    print(f"  • Custodian records:  {len(custodian_rows)}")
    print(f"  • Cash ledger rows:   {len(cash_rows)}")
    print(f"  • Injected breaks:    {len(truth_df)}")

def load_data(data_dir: str = DATA_DIR) -> dict:
    """Load all raw CSV data tables into memory."""
    if not os.path.exists(os.path.join(data_dir, "broker_trades.csv")):
        generate_data(seed=SEED, n_trades=N_TRADES, out_dir=data_dir)

    raw = {
        "broker": pd.read_csv(os.path.join(data_dir, "broker_trades.csv")),
        "clearing": pd.read_csv(os.path.join(data_dir, "clearing_records.csv")),
        "custody": pd.read_csv(os.path.join(data_dir, "custodian_records.csv")),
        "cash": pd.read_csv(os.path.join(data_dir, "cash_ledger.csv")),
        "master": pd.read_csv(os.path.join(data_dir, "securities_master.csv")),
        "positions": pd.read_csv(os.path.join(data_dir, "opening_positions.csv")),
    }
    truth_path = os.path.join(data_dir, "ground_truth.csv")
    if os.path.exists(truth_path):
        raw["truth"] = pd.read_csv(truth_path)
    return raw

# ==============================================================================
# SECTION 7.1: INGESTION AGENT
# ==============================================================================

class IngestionAgent:
    """Normalises diverse data feeds into a unified canonical schema."""
    name = "IngestionAgent"

    def run(self, ctx: Context, now: Optional[pd.Timestamp] = None) -> None:
        canon_records = []
        rejected_count = 0

        # 1. Broker trades
        b_df = ctx.raw["broker"]
        for _, r in b_df.iterrows():
            evt_time = pd.to_datetime(r["trade_time"])
            if now is not None and evt_time > now:
                continue
            if pd.isna(r["isin"]) or pd.isna(r["quantity"]):
                rejected_count += 1
                continue
            t_key = normalise_trade_key(r["trade_id"])
            canon_records.append({
                "record_id": f"BROKER:{r['trade_id']}",
                "source": "BROKER",
                "trade_key": t_key,
                "isin": str(r["isin"]).strip(),
                "side": str(r["side"]).strip().upper(),
                "quantity": int(r["quantity"]),
                "price": float(r["price"]),
                "amount": float(r["trade_amount"]),
                "trade_date": str(r["trade_date"]),
                "settlement_date": str(r["settlement_date"]),
                "status": "CAPTURED",
                "event_time": evt_time,
                "raw_ref": str(r["trade_id"])
            })

        # 2. Clearing records
        c_df = ctx.raw["clearing"]
        for _, r in c_df.iterrows():
            evt_time = pd.to_datetime(r["received_time"])
            if now is not None and evt_time > now:
                continue
            if pd.isna(r["isin"]) or pd.isna(r["quantity"]):
                rejected_count += 1
                continue
            t_key = normalise_trade_key(r.get("trade_ref"))
            canon_records.append({
                "record_id": f"CLEARING:{r['clearing_ref']}",
                "source": "CLEARING",
                "trade_key": t_key,
                "isin": str(r["isin"]).strip(),
                "side": str(r["side"]).strip().upper(),
                "quantity": int(r["quantity"]),
                "price": float(r["price"]),
                "amount": round(float(r["quantity"]) * float(r["price"]), 2),
                "trade_date": str(r["trade_date"]),
                "settlement_date": str(r["settlement_date"]),
                "status": str(r["status"]),
                "event_time": evt_time,
                "raw_ref": str(r["clearing_ref"])
            })

        # 3. Custodian records
        u_df = ctx.raw["custody"]
        for _, r in u_df.iterrows():
            evt_time = pd.to_datetime(r["received_time"])
            if now is not None and evt_time > now:
                continue
            if pd.isna(r["isin"]) or pd.isna(r["quantity"]):
                rejected_count += 1
                continue
            t_key = normalise_trade_key(r.get("trade_ref"))
            side = "BUY" if str(r["instruction_type"]).strip().upper() == "RVP" else "SELL"
            qty = int(r["quantity"])
            amt = float(r["settlement_amount"])
            price = round(amt / qty, 2) if qty > 0 else np.nan
            canon_records.append({
                "record_id": f"CUSTODY:{r['custodian_ref']}",
                "source": "CUSTODY",
                "trade_key": t_key,
                "isin": str(r["isin"]).strip(),
                "side": side,
                "quantity": qty,
                "price": price,
                "amount": amt,
                "trade_date": None,
                "settlement_date": str(r["settlement_date"]),
                "status": str(r["depository_status"]),
                "event_time": evt_time,
                "raw_ref": str(r["custodian_ref"])
            })

        # 4. Cash records
        h_df = ctx.raw["cash"]
        for _, r in h_df.iterrows():
            evt_time = pd.to_datetime(r["posted_time"])
            if now is not None and evt_time > now:
                continue
            amt = float(r["amount"])
            side = "BUY" if str(r["direction"]).strip().upper() == "DEBIT" else "SELL"
            t_key = normalise_trade_key(r.get("trade_ref"))
            canon_records.append({
                "record_id": f"CASH:{r['entry_id']}",
                "source": "CASH",
                "trade_key": t_key,
                "isin": None,
                "side": side,
                "quantity": None,
                "price": np.nan,
                "amount": amt,
                "trade_date": None,
                "settlement_date": str(r["value_date"]),
                "status": "POSTED",
                "event_time": evt_time,
                "raw_ref": str(r["entry_id"])
            })

        ctx.canon = make_dataframe(canon_records, CANON_COLUMNS)
        log(ctx, self.name, "INGESTION_COMPLETE", f"Visible records: {len(canon_records)}, Rejected: {rejected_count}")

# ==============================================================================
# SECTION 7.2: MATCHING AGENT
# ==============================================================================

class MatchingAgent:
    """Multi-source 3-level matching engine with SLA & break classification."""
    name = "MatchingAgent"

    def run(self, ctx: Context, now: Optional[pd.Timestamp] = None) -> None:
        if ctx.canon is None or ctx.canon.empty:
            return

        now_ts = now if now is not None else ctx.as_of
        canon = ctx.canon

        b_records = canon[canon["source"] == "BROKER"].copy()
        c_records = canon[canon["source"] == "CLEARING"].copy()
        u_records = canon[canon["source"] == "CUSTODY"].copy()
        h_records = canon[canon["source"] == "CASH"].copy()

        c_by_key = {}
        for _, row in c_records[c_records["trade_key"].notna()].iterrows():
            c_by_key.setdefault(row["trade_key"], []).append(row)

        u_by_key = {}
        for _, row in u_records[u_records["trade_key"].notna()].iterrows():
            u_by_key.setdefault(row["trade_key"], []).append(row)

        h_by_key = {}
        for _, row in h_records[h_records["trade_key"].notna()].iterrows():
            h_by_key.setdefault(row["trade_key"], []).append(row)

        unmatched_c = c_records[c_records["trade_key"].isna()].to_dict("records")
        matched_c_record_ids = set()

        matches_rows = []
        breaks_list = ctx.breaks.to_dict("records") if (ctx.breaks is not None and not ctx.breaks.empty) else []
        break_counter = len(breaks_list) + 1

        def add_break(t_id: str, t_k: str, b_type: str, fld: str, b_val: Any, o_val: Any,
                      o_src: str, amt: float, detail_txt: str) -> None:
            nonlocal break_counter
            pair = (t_id, b_type)
            if pair in ctx.seen_breaks:
                return
            ctx.seen_breaks.add(pair)
            breaks_list.append({
                "break_id": f"BRK-{break_counter:04d}",
                "trade_id": t_id,
                "trade_key": t_k,
                "break_type": b_type,
                "field": fld,
                "broker_value": str(b_val),
                "other_value": str(o_val),
                "other_source": o_src,
                "amount": float(amt),
                "severity": SEVERITY.get(b_type, 0.5),
                "detected_time": now_ts.strftime("%Y-%m-%d %H:%M:%S") if now_ts else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "OPEN",
                "detail": detail_txt
            })
            break_counter += 1

        for _, b_row in b_records.iterrows():
            t_id = b_row["raw_ref"]
            t_key = b_row["trade_key"]
            b_amt = b_row["amount"]
            b_qty = b_row["quantity"]
            b_price = b_row["price"]
            b_side = b_row["side"]
            b_sdate = b_row["settlement_date"]
            b_isin = b_row["isin"]
            b_time = b_row["event_time"]

            clr_id = None
            cus_id = None
            csh_id = None
            c_level = "NONE"
            u_level = "NONE"
            has_break = False

            # Clearing Match
            c_candidates = c_by_key.get(t_key, [])
            c_matched_row = None

            if len(c_candidates) > 1:
                sorted_c = sorted(c_candidates, key=lambda x: x["event_time"])
                c_matched_row = sorted_c[0]
                matched_c_record_ids.add(c_matched_row["record_id"])
                for dup in sorted_c[1:]:
                    add_break(t_id, t_key, "DUPLICATE", "clearing_ref",
                              c_matched_row["raw_ref"], dup["raw_ref"], "CLEARING", b_amt,
                              f"Duplicate clearing record {dup['raw_ref']} detected")
                    has_break = True
            elif len(c_candidates) == 1:
                c_matched_row = c_candidates[0]
                matched_c_record_ids.add(c_matched_row["record_id"])
            elif unmatched_c:
                best_score = -1.0
                best_cand = None
                for cand in unmatched_c:
                    if cand["record_id"] in matched_c_record_ids:
                        continue
                    if cand["isin"] != b_isin or cand["side"] != b_side:
                        continue
                    q1, q2 = b_qty, cand["quantity"]
                    qty_sim = 1.0 - abs(q1 - q2) / max(q1, q2)
                    p1, p2 = b_price, cand["price"]
                    price_sim = max(0.0, 1.0 - (abs(p1 - p2) / max(p1, p2)) * 50.0)
                    t_diff = abs((cand["event_time"] - b_time).total_seconds()) / 60.0
                    time_sim = max(0.0, 1.0 - min(t_diff, 60.0) / 60.0)
                    score = 0.4 * qty_sim + 0.4 * price_sim + 0.2 * time_sim
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                if best_score >= FUZZY_MIN_SCORE and best_cand is not None:
                    c_matched_row = best_cand
                    matched_c_record_ids.add(best_cand["record_id"])
                    c_level = "FUZZY"

            if c_matched_row is not None:
                clr_id = c_matched_row["raw_ref"]
                if c_level != "FUZZY":
                    c_level = "EXACT"

                if c_matched_row["side"] != b_side:
                    add_break(t_id, t_key, "SIDE_MISMATCH", "side", b_side, c_matched_row["side"],
                              "CLEARING", b_amt, f"Clearing side {c_matched_row['side']} != broker side {b_side}")
                    has_break = True
                if c_matched_row["quantity"] != b_qty:
                    add_break(t_id, t_key, "QTY_MISMATCH", "quantity", b_qty, c_matched_row["quantity"],
                              "CLEARING", b_amt, f"Clearing qty {c_matched_row['quantity']} != broker qty {b_qty}")
                    has_break = True
                p_diff = abs(c_matched_row["price"] - b_price)
                if p_diff > PRICE_TOL_ABS:
                    add_break(t_id, t_key, "PRICE_MISMATCH", "price", b_price, c_matched_row["price"],
                              "CLEARING", b_amt, f"Clearing price {c_matched_row['price']} differs by {p_diff:.2f}")
                    has_break = True
                elif 0.0 < p_diff <= PRICE_TOL_ABS:
                    c_level = "TOLERANCE"
                    add_break(t_id, t_key, "ROUNDING_DIFF", "price", b_price, c_matched_row["price"],
                              "CLEARING", b_amt, f"Clearing price differs by rounding tolerance {p_diff:.2f}")
                    has_break = True
                if c_matched_row["settlement_date"] != b_sdate:
                    c_sdate_dt = pd.to_datetime(c_matched_row["settlement_date"])
                    b_sdate_dt = pd.to_datetime(b_sdate)
                    expected_shift = add_business_days(b_sdate_dt, 1)
                    if c_sdate_dt == expected_shift:
                        add_break(t_id, t_key, "DATE_CONVENTION", "settlement_date", b_sdate,
                                  c_matched_row["settlement_date"], "CLEARING", b_amt,
                                  "Clearing date shifted by +1 business day (market holiday convention)")
                    else:
                        add_break(t_id, t_key, "DATE_MISMATCH", "settlement_date", b_sdate,
                                  c_matched_row["settlement_date"], "CLEARING", b_amt,
                                  f"Clearing settlement date {c_matched_row['settlement_date']} != {b_sdate}")
                    has_break = True
            else:
                sla_deadline = b_time + timedelta(minutes=CLEARING_SLA_MIN)
                if now_ts is not None and now_ts >= sla_deadline:
                    add_break(t_id, t_key, "MISSING_CLEARING", "clearing_ref", "PRESENT", "MISSING",
                              "CLEARING", b_amt, f"Clearing confirmation SLA expired at {sla_deadline}")
                    has_break = True

            # Custody Match
            u_candidates = u_by_key.get(t_key, [])
            if u_candidates:
                u_row = u_candidates[0]
                cus_id = u_row["raw_ref"]
                u_level = "EXACT"

                if u_row["quantity"] != b_qty:
                    add_break(t_id, t_key, "QTY_MISMATCH", "quantity", b_qty, u_row["quantity"],
                              "CUSTODY", b_amt, f"Custody quantity {u_row['quantity']} != broker qty {b_qty}")
                    has_break = True
                amt_diff = abs(u_row["amount"] - b_amt)
                if 0.0 < amt_diff <= AMOUNT_TOL_ABS:
                    u_level = "TOLERANCE"
                    add_break(t_id, t_key, "ROUNDING_DIFF", "settlement_amount", b_amt, u_row["amount"],
                              "CUSTODY", b_amt, f"Custody settlement amount diff {amt_diff:.2f} within tolerance")
                    has_break = True
                elif amt_diff > AMOUNT_TOL_ABS:
                    add_break(t_id, t_key, "PRICE_MISMATCH", "settlement_amount", b_amt, u_row["amount"],
                              "CUSTODY", b_amt, f"Custody settlement amount difference ₹{amt_diff:.2f}")
                    has_break = True
                if u_row["settlement_date"] != b_sdate:
                    add_break(t_id, t_key, "DATE_MISMATCH", "settlement_date", b_sdate, u_row["settlement_date"],
                              "CUSTODY", b_amt, f"Custodian settlement date {u_row['settlement_date']} != {b_sdate}")
                    has_break = True
            else:
                sla_deadline = b_time + timedelta(minutes=CUSTODY_SLA_MIN)
                if now_ts is not None and now_ts >= sla_deadline:
                    add_break(t_id, t_key, "MISSING_CUSTODY", "custodian_ref", "PRESENT", "MISSING",
                              "CUSTODY", b_amt, f"Custodian delivery instruction SLA expired at {sla_deadline}")
                    has_break = True

            # Cash Match
            h_candidates = h_by_key.get(t_key, [])
            if h_candidates:
                h_row = h_candidates[0]
                csh_id = h_row["raw_ref"]
                csh_amt_diff = abs(h_row["amount"] - b_amt)
                if csh_amt_diff > AMOUNT_TOL_ABS or h_row["side"] != b_side:
                    add_break(t_id, t_key, "CASH_MISMATCH", "amount", b_amt, h_row["amount"],
                              "CASH", b_amt, f"Treasury cash diff ₹{csh_amt_diff:.2f} or direction mismatch")
                    has_break = True

            match_status = "BREAK" if (has_break or clr_id is None or cus_id is None) else "MATCHED"
            matches_rows.append({
                "trade_id": t_id,
                "trade_key": t_key,
                "isin": b_isin,
                "side": b_side,
                "quantity": b_qty,
                "price": b_price,
                "amount": b_amt,
                "settlement_date": b_sdate,
                "trade_time": str(b_time),
                "clearing_id": clr_id,
                "custody_id": cus_id,
                "cash_id": csh_id,
                "clearing_match_level": c_level,
                "custody_match_level": u_level,
                "match_status": match_status
            })

        ctx.matches = make_dataframe(matches_rows, MATCH_COLUMNS)
        ctx.breaks = make_dataframe(breaks_list, BREAK_COLUMNS)
        open_breaks_cnt = len(ctx.breaks[ctx.breaks["status"] == "OPEN"]) if (ctx.breaks is not None and not ctx.breaks.empty and "status" in ctx.breaks.columns) else 0
        log(ctx, self.name, "MATCHING_COMPLETE", f"Matched trades: {len(matches_rows)}, Open breaks: {open_breaks_cnt}")

# ==============================================================================
# SECTION 7.3 & 7.4: ANOMALY DETECTOR & RISK AGENT
# ==============================================================================

class RiskAgent:
    """Computes settlement-failure risk scores and identifies statistical anomalies."""
    name = "RiskAgent"

    def run(self, ctx: Context, now: Optional[pd.Timestamp] = None) -> None:
        if ctx.matches is None or ctx.matches.empty:
            return

        now_ts = now if now is not None else ctx.as_of
        if now_ts is None:
            now_ts = pd.to_datetime(BASE_DATE + " 15:30:00")

        broker_df = ctx.raw["broker"].copy()
        anomaly_scores: Dict[str, float] = {}
        anomaly_flags: Dict[str, bool] = {}

        for isin, grp in broker_df.groupby("isin"):
            med_q = grp["quantity"].median()
            mad_q = (grp["quantity"] - med_q).abs().median()
            med_p = grp["price"].median()
            mad_p = (grp["price"] - med_p).abs().median()
            med_amt = grp["trade_amount"].median()

            for _, trd in grp.iterrows():
                tid = trd["trade_id"]
                z_q = (0.6745 * (trd["quantity"] - med_q) / mad_q) if mad_q > 0 else 0.0
                z_p = (0.6745 * (trd["price"] - med_p) / mad_p) if mad_p > 0 else 0.0
                is_anom = (abs(z_q) > ANOMALY_Z) or (abs(z_p) > ANOMALY_Z) or (trd["trade_amount"] > 20.0 * med_amt)
                score = min(1.0, max(abs(z_q), abs(z_p)) / (2.0 * ANOMALY_Z))
                anomaly_flags[tid] = is_anom
                anomaly_scores[tid] = round(score, 3)

        # Defensive lookup for active breaks
        breaks_by_trade: Dict[str, List[dict]] = {}
        if ctx.breaks is not None and not ctx.breaks.empty and "status" in ctx.breaks.columns:
            active_breaks = ctx.breaks[ctx.breaks["status"].isin(["OPEN", "ESCALATED"])]
            for _, brk in active_breaks.iterrows():
                breaks_by_trade.setdefault(brk["trade_id"], []).append(brk)

        risk_rows = []
        for _, trd in ctx.matches.iterrows():
            t_id = trd["trade_id"]
            amt = float(trd["amount"])
            isin = trd["isin"]
            s_date = trd["settlement_date"]

            cutoff_time = pd.to_datetime(f"{s_date} {CUTOFF_HOUR:02d}:00:00")
            hours_to_cutoff = max(0.0, (cutoff_time - now_ts).total_seconds() / 3600.0)

            trade_brks = breaks_by_trade.get(t_id, [])
            break_severity = max([b["severity"] for b in trade_brks], default=0.0)
            time_pressure = float(np.clip(1.0 - (hours_to_cutoff / 24.0), 0.0, 1.0))
            value_norm = float(np.clip(amt / 5_000_000.0, 0.0, 1.0))
            anom_score = anomaly_scores.get(t_id, 0.0)
            has_anomaly = anomaly_flags.get(t_id, False)

            if trade_brks or has_anomaly:
                risk_score = (0.45 * break_severity +
                              0.25 * time_pressure +
                              0.15 * value_norm +
                              0.15 * anom_score)
            else:
                risk_score = 0.10 * time_pressure

            risk_score = round(float(np.clip(risk_score, 0.0, 1.0)), 3)
            risk_level = "HIGH" if risk_score >= RISK_HIGH else ("MEDIUM" if risk_score >= RISK_MEDIUM else "LOW")

            reasons = []
            if trade_brks:
                brk_types = ",".join(list(set([b["break_type"] for b in trade_brks])))
                reasons.append(f"{brk_types}")
            if has_anomaly:
                reasons.append("Extreme Volume/Price Anomaly")
            reasons.append(f"{hours_to_cutoff:.1f}h to cutoff")
            reasons.append(f"value ₹{amt/100_000:.1f}L")
            reason_txt = "; ".join(reasons)

            risk_rows.append({
                "trade_id": t_id,
                "isin": isin,
                "amount": amt,
                "settlement_date": s_date,
                "cutoff_time": cutoff_time.strftime("%Y-%m-%d %H:%M:%S"),
                "hours_to_cutoff": round(hours_to_cutoff, 1),
                "break_severity": round(break_severity, 2),
                "time_pressure": round(time_pressure, 2),
                "value_norm": round(value_norm, 2),
                "anomaly_score": anom_score,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "reason": reason_txt
            })

        ctx.risk = make_dataframe(risk_rows, RISK_COLUMNS)
        high_risk_cnt = len(ctx.risk[ctx.risk["risk_level"] == "HIGH"]) if not ctx.risk.empty else 0
        log(ctx, self.name, "RISK_EVALUATION_COMPLETE", f"HIGH risk trades: {high_risk_cnt}")

# ==============================================================================
# SECTION 7.5: LIQUIDITY AGENT
# ==============================================================================

class LiquidityAgent:
    """Monitors cash balance trajectories and detects depository shortfalls."""
    name = "LiquidityAgent"

    def run(self, ctx: Context, now: Optional[pd.Timestamp] = None) -> None:
        if ctx.matches is None or ctx.raw.get("positions") is None:
            return

        broker_df = ctx.raw["broker"].copy()
        cash_df = ctx.raw["cash"].copy()
        pos_df = ctx.raw["positions"].copy()
        master_df = ctx.raw["master"].copy()

        # 1. Cash Forecasting
        all_dates = sorted(list(set(broker_df["settlement_date"].unique()) | set(cash_df["value_date"].unique())))
        cash_rows = []
        running_cash = float(OPENING_CASH)

        for d in all_dates:
            b_day = broker_df[broker_df["settlement_date"] == d]
            sell_amt = b_day[b_day["side"] == "SELL"]["trade_amount"].sum()
            buy_amt = b_day[b_day["side"] == "BUY"]["trade_amount"].sum()
            fo_net = round(float(sell_amt - buy_amt), 2)

            c_day = cash_df[cash_df["value_date"] == d]
            credit_amt = c_day[c_day["direction"] == "CREDIT"]["amount"].sum()
            debit_amt = c_day[c_day["direction"] == "DEBIT"]["amount"].sum()
            treasury_net = round(float(credit_amt - debit_amt), 2)

            diff = round(fo_net - treasury_net, 2)
            running_cash = round(running_cash + fo_net, 2)
            shortfall = running_cash < MIN_CASH_BUFFER

            cash_rows.append({
                "settlement_date": d,
                "front_office_net": fo_net,
                "treasury_net": treasury_net,
                "difference": diff,
                "opening_cash": OPENING_CASH,
                "projected_closing_cash": running_cash,
                "shortfall_flag": shortfall
            })

        cash_forecast_df = make_dataframe(cash_rows, CASH_FORECAST_COLUMNS)

        # 2. Securities Forecasting
        sec_rows = []
        sym_lookup = dict(zip(master_df["isin"], master_df["symbol"]))
        open_trades = ctx.matches[ctx.matches["match_status"].isin(["MATCHED", "BREAK"])] if (ctx.matches is not None and not ctx.matches.empty) else pd.DataFrame()

        for _, pos in pos_df.iterrows():
            isin = pos["isin"]
            sym = sym_lookup.get(isin, isin)
            open_qty = int(pos["opening_quantity"])

            if not open_trades.empty:
                t_isin = open_trades[open_trades["isin"] == isin]
                buys = t_isin[t_isin["side"] == "BUY"]["quantity"].sum()
                sells = t_isin[t_isin["side"] == "SELL"]["quantity"].sum()
                net_settling = int(buys - sells)
            else:
                net_settling = 0

            proj_qty = open_qty + net_settling
            short_flag = proj_qty < 0

            sec_rows.append({
                "isin": isin,
                "symbol": sym,
                "opening_quantity": open_qty,
                "net_settling_quantity": net_settling,
                "projected_quantity": proj_qty,
                "short_flag": short_flag
            })

        sec_forecast_df = make_dataframe(sec_rows, SEC_FORECAST_COLUMNS)

        # 3. Liquidity Alerts
        alerts = []
        alert_idx = 1
        for _, r in cash_forecast_df.iterrows():
            if abs(r["difference"]) > 100_000:
                alerts.append({
                    "alert_id": f"ALT-{alert_idx:04d}",
                    "type": "CASH_DIFF",
                    "key": r["settlement_date"],
                    "message": f"Front office vs Treasury net discrepancy of ₹{r['difference']:,.2f} on {r['settlement_date']}",
                    "severity": "MEDIUM"
                })
                alert_idx += 1
            if r["shortfall_flag"]:
                alerts.append({
                    "alert_id": f"ALT-{alert_idx:04d}",
                    "type": "CASH_SHORTFALL",
                    "key": r["settlement_date"],
                    "message": f"Projected treasury balance ₹{r['projected_closing_cash']:,.2f} below minimum reserve buffer",
                    "severity": "HIGH"
                })
                alert_idx += 1

        for _, r in sec_forecast_df.iterrows():
            if r["short_flag"]:
                alerts.append({
                    "alert_id": f"ALT-{alert_idx:04d}",
                    "type": "SEC_SHORT",
                    "key": r["symbol"],
                    "message": f"Projected depository shortfall of {r['projected_quantity']} shares for {r['symbol']}",
                    "severity": "HIGH"
                })
                alert_idx += 1

        ctx.liquidity = {
            "cash": cash_forecast_df,
            "securities": sec_forecast_df,
            "alerts": make_dataframe(alerts, ALERT_COLUMNS)
        }
        log(ctx, self.name, "LIQUIDITY_FORECAST_COMPLETE", f"Cash alerts: {len(alerts)}")

# ==============================================================================
# SECTION 7.6: RESOLUTION AGENT (GUARDRAILS & ESCALATION)
# ==============================================================================

class ResolutionAgent:
    """Applies strict policy guardrails to auto-resolve or escalate breaks."""
    name = "ResolutionAgent"

    SUGGESTED_STEPS = {
        "MISSING_CLEARING": "Call clearing member operations to request immediate CCP trade novation message.",
        "MISSING_CUSTODY": "Instruct custodian settlement desk to match pending RVP/DVP depository ticket.",
        "QTY_MISMATCH": "Contact broker counterparty to confirm allocated lot count; request amended contract note.",
        "PRICE_MISMATCH": "Verify execution fill against electronic order book; request broker re-booking.",
        "DATE_MISMATCH": "Re-align custodian settlement instruction date with broker trade contract date.",
        "DATE_CONVENTION": "Apply market holiday convention adjustment (+1 business day novation shift).",
        "SIDE_MISMATCH": "URGENT: Reconcile trade blotter. Counterparty recorded opposite buy/sell side.",
        "DUPLICATE": "Suppress duplicate CCP novation record; confirm single settlement instruction with clearing member.",
        "ROUNDING_DIFF": "Accept fractional rounding differential within permitted tolerance threshold.",
        "CASH_MISMATCH": "Instruct bank treasury to update cash ledger debit/credit amount to match gross settlement amount."
    }

    def run(self, ctx: Context, now: Optional[pd.Timestamp] = None) -> None:
        if ctx.breaks is None or ctx.breaks.empty or "status" not in ctx.breaks.columns:
            return

        actions_list = ctx.actions.to_dict("records") if (ctx.actions is not None and not ctx.actions.empty) else []
        action_idx = len(actions_list) + 1

        risk_lookup = {}
        if ctx.risk is not None and not ctx.risk.empty and "trade_id" in ctx.risk.columns:
            for _, r in ctx.risk.iterrows():
                risk_lookup[r["trade_id"]] = r

        trade_open_breaks: Dict[str, int] = {}
        for _, b in ctx.breaks[ctx.breaks["status"] == "OPEN"].iterrows():
            trade_open_breaks[b["trade_id"]] = trade_open_breaks.get(b["trade_id"], 0) + 1

        now_str = now.strftime("%Y-%m-%d %H:%M:%S") if now is not None else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for idx, brk in ctx.breaks.iterrows():
            if brk["status"] != "OPEN":
                continue

            b_id = brk["break_id"]
            t_id = brk["trade_id"]
            b_type = brk["break_type"]
            amt = float(brk["amount"])

            if b_type == "DUPLICATE":
                confidence = 0.99
            elif b_type == "ROUNDING_DIFF":
                confidence = 0.95
            elif b_type == "DATE_CONVENTION":
                confidence = 0.92
            else:
                confidence = 0.50

            checks = [
                {
                    "policy_id": "POL-001",
                    "passed": not ctx.kill_switch,
                    "note": "Kill switch ACTIVE - manual intervention enforced" if ctx.kill_switch else "Kill switch inactive"
                },
                {
                    "policy_id": "POL-002",
                    "passed": b_type in AUTO_TYPES,
                    "note": f"Break type '{b_type}' is eligible for auto-resolution" if b_type in AUTO_TYPES else f"Type '{b_type}' requires human review"
                },
                {
                    "policy_id": "POL-003",
                    "passed": amt <= ctx.auto_resolve_limit,
                    "note": f"Amount ₹{amt:,.2f} within limit ₹{ctx.auto_resolve_limit:,.2f}" if amt <= ctx.auto_resolve_limit else f"Amount ₹{amt:,.2f} exceeds limit"
                },
                {
                    "policy_id": "POL-004",
                    "passed": confidence >= AUTO_CONFIDENCE_MIN,
                    "note": f"Confidence {confidence:.2f} meets threshold {AUTO_CONFIDENCE_MIN}" if confidence >= AUTO_CONFIDENCE_MIN else f"Confidence {confidence:.2f} insufficient"
                },
                {
                    "policy_id": "POL-005",
                    "passed": trade_open_breaks.get(t_id, 0) <= 1,
                    "note": "Single break on trade" if trade_open_breaks.get(t_id, 0) <= 1 else f"Multiple breaks ({trade_open_breaks.get(t_id, 0)}) on trade"
                }
            ]

            all_passed = all(c["passed"] for c in checks)

            if all_passed:
                decision = "AUTO_RESOLVED"
                ctx.breaks.at[idx, "status"] = "AUTO_RESOLVED"
                if b_type == "DUPLICATE":
                    act_text = f"Suppress duplicate clearing record {brk['other_value']}"
                elif b_type == "ROUNDING_DIFF":
                    act_text = f"Accept clearing/custody rounding tolerance of ₹{abs(float(brk['broker_value']) - float(brk['other_value'])):.2f}"
                elif b_type == "DATE_CONVENTION":
                    act_text = "Accept clearing date +1 business day per exchange holiday calendar"
                else:
                    act_text = "Auto-resolved per policy rule"
            else:
                decision = "ESCALATED"
                ctx.breaks.at[idx, "status"] = "ESCALATED"
                failed_policies = [c["policy_id"] for c in checks if not c["passed"]]
                act_text = f"Escalated to Operations (Failed: {', '.join(failed_policies)})"

                r_info = risk_lookup.get(t_id, {})
                r_score = r_info.get("risk_score", 0.5)
                r_lvl = r_info.get("risk_level", "MEDIUM")
                h_to_cut = r_info.get("hours_to_cutoff", 0.0)
                prio = 1 if r_lvl == "HIGH" else (2 if r_lvl == "MEDIUM" else 3)

                ctx.cases[b_id] = {
                    "break_id": b_id,
                    "trade_id": t_id,
                    "break_type": b_type,
                    "summary": f"{b_type} on trade {t_id} (value ₹{amt:,.2f}) - {act_text}",
                    "evidence": {
                        "broker": {"field": brk["field"], "value": brk["broker_value"]},
                        "other": {"source": brk["other_source"], "value": brk["other_value"]}
                    },
                    "checks": checks,
                    "risk_score": r_score,
                    "risk_level": r_lvl,
                    "hours_to_cutoff": h_to_cut,
                    "suggested_next_step": self.SUGGESTED_STEPS.get(b_type, "Investigate trade discrepancy."),
                    "priority": prio
                }

            actions_list.append({
                "action_id": f"ACT-{action_idx:04d}",
                "break_id": b_id,
                "trade_id": t_id,
                "break_type": b_type,
                "decision": decision,
                "policy_checks_passed": all_passed,
                "confidence": confidence,
                "action_text": act_text,
                "decided_time": now_str
            })
            action_idx += 1

        ctx.actions = make_dataframe(actions_list, ACTION_COLUMNS)
        auto_cnt = len([a for a in actions_list if a["decision"] == "AUTO_RESOLVED"])
        esc_cnt = len([a for a in actions_list if a["decision"] == "ESCALATED"])
        log(ctx, self.name, "RESOLUTION_COMPLETE", f"Auto-resolved: {auto_cnt}, Escalated: {esc_cnt}")

# ==============================================================================
# SECTION 8: ORCHESTRATOR & STREAM PIPELINE
# ==============================================================================

class Orchestrator:
    """Coordinates agent execution across simulated temporal ticks."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.ingestion = IngestionAgent()
        self.matching = MatchingAgent()
        self.risk = RiskAgent()
        self.resolution = ResolutionAgent()
        self.liquidity = LiquidityAgent()

    def run_stream(self, progress_cb=None) -> Context:
        b_df = self.ctx.raw["broker"]
        c_df = self.ctx.raw["clearing"]
        u_df = self.ctx.raw["custody"]
        h_df = self.ctx.raw["cash"]

        event_times = set()
        for ts_col, df_item in [
            ("trade_time", b_df),
            ("received_time", c_df),
            ("received_time", u_df),
            ("posted_time", h_df)
        ]:
            for val in df_item[ts_col].dropna():
                event_times.add(pd.to_datetime(val))

        for _, r in b_df.iterrows():
            bt = pd.to_datetime(r["trade_time"])
            event_times.add(bt + timedelta(minutes=CLEARING_SLA_MIN))
            event_times.add(bt + timedelta(minutes=CUSTODY_SLA_MIN))

        sorted_ticks = sorted(list(event_times))
        if len(sorted_ticks) > 300:
            step = max(1, len(sorted_ticks) // 200)
            sampled_ticks = sorted_ticks[::step]
            if sorted_ticks[-1] not in sampled_ticks:
                sampled_ticks.append(sorted_ticks[-1])
        else:
            sampled_ticks = sorted_ticks

        total_ticks = len(sampled_ticks)
        for i, now in enumerate(sampled_ticks):
            self.ctx.as_of = now

            self.ingestion.run(self.ctx, now)
            self.matching.run(self.ctx, now)
            self.risk.run(self.ctx, now)
            self.resolution.run(self.ctx, now)

            if i % 30 == 0 or i == total_ticks - 1:
                self.liquidity.run(self.ctx, now)

            if progress_cb is not None:
                progress_cb((i + 1) / total_ticks)

        log(self.ctx, "Orchestrator", "PIPELINE_COMPLETE", f"Processed {total_ticks} stream ticks")
        return self.ctx

def run_pipeline(data_dir: str = DATA_DIR, kill_switch: bool = KILL_SWITCH,
                 auto_limit: float = AUTO_RESOLVE_LIMIT, progress_cb=None) -> Context:
    """End-to-end execution pipeline."""
    raw = load_data(data_dir)
    ctx = Context(
        raw=raw,
        kill_switch=kill_switch,
        auto_resolve_limit=auto_limit
    )
    orch = Orchestrator(ctx)
    return orch.run_stream(progress_cb=progress_cb)

# ==============================================================================
# SECTION 10: EVALUATION FRAMEWORK
# ==============================================================================

def evaluate(ctx: Context, truth_df: pd.DataFrame) -> dict:
    """Benchmark detected breaks and auto-resolutions against ground truth."""
    if truth_df is None or truth_df.empty or ctx.breaks is None:
        return {}

    detected_breaks = ctx.breaks.copy() if not ctx.breaks.empty else pd.DataFrame(columns=BREAK_COLUMNS)
    truth_set = set(zip(truth_df["trade_id"], truth_df["break_type"]))
    detected_set = set(zip(detected_breaks["trade_id"], detected_breaks["break_type"]))

    tp_set = truth_set & detected_set
    fp_set = detected_set - truth_set
    fn_set = truth_set - detected_set

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    actions_df = ctx.actions if (ctx.actions is not None and not ctx.actions.empty) else pd.DataFrame(columns=ACTION_COLUMNS)
    truth_action_map = dict(zip(zip(truth_df["trade_id"], truth_df["break_type"]), truth_df["correct_action"]))

    auto_resolved_actions = actions_df[actions_df["decision"] == "AUTO_RESOLVED"] if "decision" in actions_df.columns else pd.DataFrame()
    correct_auto = 0
    wrong_auto = 0

    if not auto_resolved_actions.empty:
        for _, act in auto_resolved_actions.iterrows():
            pair = (act["trade_id"], act["break_type"])
            expected_action = truth_action_map.get(pair)
            if expected_action == "AUTO_FIX":
                correct_auto += 1
            else:
                wrong_auto += 1

    total_auto = len(auto_resolved_actions)
    auto_accuracy = (correct_auto / total_auto) if total_auto > 0 else 1.0
    escalation_rate = len(actions_df[actions_df["decision"] == "ESCALATED"]) / len(actions_df) if not actions_df.empty else 0.0

    latencies = []
    truth_injected_map = dict(zip(zip(truth_df["trade_id"], truth_df["break_type"]), truth_df["injected_time"]))
    if "detected_time" in detected_breaks.columns:
        for _, brk in detected_breaks.iterrows():
            pair = (brk["trade_id"], brk["break_type"])
            if pair in tp_set:
                inj_str = truth_injected_map.get(pair)
                if inj_str:
                    inj_t = pd.to_datetime(inj_str)
                    det_t = pd.to_datetime(brk["detected_time"])
                    lat_sec = max(0.0, (det_t - inj_t).total_seconds())
                    latencies.append(lat_sec)

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    med_lat = float(np.median(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    per_type_rows = []
    for b_type in BREAK_TYPES:
        t_sub = {p for p in truth_set if p[1] == b_type}
        d_sub = {p for p in detected_set if p[1] == b_type}
        sub_tp = len(t_sub & d_sub)
        sub_fp = len(d_sub - t_sub)
        sub_fn = len(t_sub - d_sub)
        sub_prec = sub_tp / (sub_tp + sub_fp) if (sub_tp + sub_fp) > 0 else 1.0
        sub_rec = sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else 1.0
        per_type_rows.append({
            "break_type": b_type,
            "truth_count": len(t_sub),
            "detected_count": len(d_sub),
            "precision": round(sub_prec, 3),
            "recall": round(sub_rec, 3)
        })

    high_risk_trades = len(ctx.risk[ctx.risk["risk_level"] == "HIGH"]) if (ctx.risk is not None and not ctx.risk.empty and "risk_level" in ctx.risk.columns) else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "auto_resolution_accuracy": round(auto_accuracy, 4),
        "wrong_auto_resolutions": wrong_auto,
        "escalation_rate": round(escalation_rate, 4),
        "latency_mean_sec": round(mean_lat, 1),
        "latency_median_sec": round(med_lat, 1),
        "latency_p95_sec": round(p95_lat, 1),
        "total_trades": len(ctx.matches) if ctx.matches is not None else 0,
        "total_breaks": len(detected_breaks),
        "auto_resolved": total_auto,
        "escalated": len(actions_df) - total_auto if not actions_df.empty else 0,
        "high_risk_trades": high_risk_trades,
        "per_type": pd.DataFrame(per_type_rows)
    }

# ==============================================================================
# SECTION 11: STREAMLIT USER INTERFACE
# ==============================================================================

def run_ui():
    """Streamlit web dashboard for multi-agent trade verification."""
    import streamlit as st

    st.set_page_config(
        page_title="Agentic Trade Verification & Risk Prevention",
        page_icon="🛡️",
        layout="wide"
    )

    st.sidebar.title("🛡️ System Control Center")
    st.sidebar.markdown("---")

    kill_switch_toggle = st.sidebar.checkbox("🚨 Emergency KILL SWITCH", value=False,
                                             help="If enabled, blocks all automated resolution actions.")
    auto_limit_val = st.sidebar.slider("Auto-Resolve Cap (₹ INR)", min_value=100_000,
                                       max_value=5_000_000, value=AUTO_RESOLVE_LIMIT, step=100_000)

    st.sidebar.markdown("### Data Operations")
    if st.sidebar.button("🔄 Generate Synthetic Data (Seed 42)", use_container_width=True):
        with st.spinner("Generating 2,000 synthetic trades across 4 data feeds..."):
            generate_data(seed=SEED, n_trades=N_TRADES, out_dir=DATA_DIR)
            st.session_state.pop("ctx", None)
            st.sidebar.success("New synthetic dataset generated!")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Custom CSVs (Optional)",
        type=["csv"],
        accept_multiple_files=True
    )
    if uploaded_files:
        os.makedirs(DATA_DIR, exist_ok=True)
        for uf in uploaded_files:
            target_path = os.path.join(DATA_DIR, uf.name)
            with open(target_path, "wb") as f:
                f.write(uf.getbuffer())
        st.sidebar.info(f"Saved {len(uploaded_files)} custom files.")

    run_pipeline_btn = st.sidebar.button("▶️ Run Agent Pipeline", type="primary", use_container_width=True)

    if run_pipeline_btn or "ctx" not in st.session_state:
        with st.spinner("Executing agentic trade verification stream..."):
            prog_bar = st.progress(0.0)
            ctx = run_pipeline(
                data_dir=DATA_DIR,
                kill_switch=kill_switch_toggle,
                auto_limit=auto_limit_val,
                progress_cb=lambda p: prog_bar.progress(p)
            )
            prog_bar.empty()
            st.session_state["ctx"] = ctx

    ctx: Context = st.session_state["ctx"]
    truth_df = ctx.raw.get("truth")

    st.title("Autonomous Multi-Agent Trade Verification & Risk Engine")
    st.caption("Cross-system reconciliation across Broker, Clearing CCP, Depository, and Treasury Ledgers")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "🔍 Breaks Blotter", "📁 Case Files", "⚡ Risk & Failure Scoring",
        "💧 Liquidity Forecast", "📜 Audit & Logs", "🎯 Model Evaluation"
    ])

    # TAB 1: OVERVIEW
    with tab1:
        st.markdown("### Real-Time Key Performance Indicators")
        tot_trades = len(ctx.matches) if ctx.matches is not None else 0
        tot_breaks = len(ctx.breaks) if ctx.breaks is not None else 0
        tot_auto = len(ctx.actions[ctx.actions["decision"] == "AUTO_RESOLVED"]) if (ctx.actions is not None and not ctx.actions.empty and "decision" in ctx.actions.columns) else 0
        tot_esc = len(ctx.actions[ctx.actions["decision"] == "ESCALATED"]) if (ctx.actions is not None and not ctx.actions.empty and "decision" in ctx.actions.columns) else 0
        tot_high_risk = len(ctx.risk[ctx.risk["risk_level"] == "HIGH"]) if (ctx.risk is not None and not ctx.risk.empty and "risk_level" in ctx.risk.columns) else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades Captured", f"{tot_trades:,}")
        c2.metric("Breaks Detected", f"{tot_breaks:,}", delta=f"{tot_breaks/max(tot_trades,1):.1%} of blotter", delta_color="inverse")
        c3.metric("Auto-Resolved (Guardrails)", f"{tot_auto:,}", delta=f"{tot_auto/max(tot_breaks,1):.1%}")
        c4.metric("Escalated to Ops", f"{tot_esc:,}", delta=f"{tot_esc/max(tot_breaks,1):.1%}", delta_color="inverse")
        c5.metric("High Settlement Risk", f"{tot_high_risk:,}", delta="Urgent cutoff", delta_color="inverse")

        st.markdown("---")
        col_chart1, col_chart2 = st.columns([1.2, 1.0])

        with col_chart1:
            st.subheader("Break Distribution by Classification")
            if ctx.breaks is not None and not ctx.breaks.empty and "break_type" in ctx.breaks.columns:
                brk_counts = ctx.breaks["break_type"].value_counts().reset_index()
                brk_counts.columns = ["Break Type", "Count"]
                fig_bar = px.bar(
                    brk_counts, x="Break Type", y="Count",
                    color="Count", color_continuous_scale="Reds",
                    text_auto=True
                )
                fig_bar.update_layout(xaxis_tickangle=-35, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            st.subheader("Resolution Decision Ratio")
            if ctx.actions is not None and not ctx.actions.empty and "decision" in ctx.actions.columns:
                res_counts = ctx.actions["decision"].value_counts().reset_index()
                res_counts.columns = ["Decision", "Count"]
                fig_donut = px.pie(
                    res_counts, names="Decision", values="Count",
                    hole=0.55,
                    color="Decision",
                    color_discrete_map={"AUTO_RESOLVED": "#2ECC71", "ESCALATED": "#E74C3C"}
                )
                fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_donut, use_container_width=True)

    # TAB 2: BREAKS BLOTTER
    with tab2:
        st.subheader("Active & Resolved Breaks Blotter")
        if ctx.breaks is not None and not ctx.breaks.empty:
            merged_breaks = ctx.breaks.copy()
            if ctx.risk is not None and not ctx.risk.empty and "trade_id" in ctx.risk.columns:
                merged_breaks = merged_breaks.merge(
                    ctx.risk[["trade_id", "risk_score", "risk_level"]], on="trade_id", how="left"
                )
            else:
                merged_breaks["risk_score"] = 0.0
                merged_breaks["risk_level"] = "LOW"

            if ctx.actions is not None and not ctx.actions.empty and "break_id" in ctx.actions.columns:
                merged_breaks = merged_breaks.merge(
                    ctx.actions[["break_id", "decision", "action_text"]], on="break_id", how="left"
                )
            else:
                merged_breaks["decision"] = "OPEN"
                merged_breaks["action_text"] = "Pending analysis"

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                f_type = st.multiselect("Filter by Break Type", options=BREAK_TYPES, default=[])
            with col_f2:
                f_status = st.multiselect("Filter by Status", options=BREAK_STATUS, default=[])
            with col_f3:
                f_risk = st.multiselect("Filter by Risk Level", options=["HIGH", "MEDIUM", "LOW"], default=[])

            filtered_df = merged_breaks.copy()
            if f_type and "break_type" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["break_type"].isin(f_type)]
            if f_status and "status" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["status"].isin(f_status)]
            if f_risk and "risk_level" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["risk_level"].isin(f_risk)]

            display_cols = [
                c for c in [
                    "break_id", "trade_id", "break_type", "status", "risk_level",
                    "risk_score", "amount", "other_source", "broker_value", "other_value", "action_text"
                ] if c in filtered_df.columns
            ]
            st.dataframe(
                filtered_df[display_cols].style.format({"amount": "₹{:,.2f}", "risk_score": "{:.2f}"}),
                use_container_width=True,
                height=450
            )

            csv_export = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Filtered Breaks (CSV)", csv_export, "breaks_report.csv", "text/csv")
        else:
            st.info("No breaks recorded.")

    # TAB 3: CASE FILES
    with tab3:
        st.subheader("Escalated Exception Case Files")
        if ctx.cases:
            escalated_ids = sorted(
                list(ctx.cases.keys()),
                key=lambda x: ctx.cases[x].get("priority", 3)
            )
            selected_case_id = st.selectbox("Select Escalated Case File", escalated_ids)
            cfile = ctx.cases[selected_case_id]

            st.markdown(f"#### Case: `{cfile['break_id']}` | Trade: `{cfile['trade_id']}` | Type: `{cfile['break_type']}`")
            p_badge = "🔴 High Priority" if cfile["priority"] == 1 else ("🟠 Medium Priority" if cfile["priority"] == 2 else "🟢 Normal Priority")
            st.markdown(f"**Status**: Escalated to Operations &nbsp;|&nbsp; **Risk**: {cfile['risk_level']} ({cfile['risk_score']:.2f}) &nbsp;|&nbsp; **Priority**: {p_badge}")

            st.info(f"**Executive Summary:** {cfile['summary']}")

            col_ev1, col_ev2 = st.columns(2)
            with col_ev1:
                st.markdown("##### 🏛️ Broker Evidence (Internal Ledger)")
                st.json(cfile["evidence"]["broker"])
            with col_ev2:
                st.markdown(f"##### 🏢 Counterparty Evidence ({cfile['evidence']['other'].get('source', 'External')})")
                st.json(cfile["evidence"]["other"])

            st.markdown("##### 🛡️ Autonomous Guardrail Policy Check Evaluation")
            check_rows = []
            for chk in cfile["checks"]:
                icon = "✅ PASSED" if chk["passed"] else "❌ FAILED"
                check_rows.append({"Policy Code": chk["policy_id"], "Evaluation": icon, "Audit Note": chk["note"]})
            st.table(pd.DataFrame(check_rows))

            st.markdown("##### 🎯 Prescribed Next Action Step")
            st.success(f"**Recommended Action:** {cfile['suggested_next_step']}")

            st.download_button(
                "📥 Download Case File (JSON)",
                json.dumps(cfile, indent=2),
                file_name=f"case_{cfile['break_id']}.json",
                mime="application/json"
            )
        else:
            st.success("No escalated case files! All breaks were either absent or auto-resolved.")

    # TAB 4: RISK & FAILURE SCORING
    with tab4:
        st.subheader("Settlement Risk Distribution & Anomaly Scoring")
        if ctx.risk is not None and not ctx.risk.empty and "risk_score" in ctx.risk.columns:
            col_r1, col_r2 = st.columns([1, 1.2])

            with col_r1:
                fig_hist = px.histogram(
                    ctx.risk, x="risk_score", nbins=20,
                    color="risk_level",
                    color_discrete_map={"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#2ECC71"},
                    title="Risk Score Histogram"
                )
                fig_hist.update_layout(xaxis_title="Risk Score (0 to 1)", yaxis_title="Trade Count")
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_r2:
                st.markdown("##### Top 15 Riskiest Settlement Exposures")
                top_risky = ctx.risk.sort_values(by="risk_score", ascending=False).head(15)
                st.dataframe(
                    top_risky[["trade_id", "isin", "amount", "hours_to_cutoff", "anomaly_score", "risk_score", "risk_level", "reason"]]
                    .style.format({"amount": "₹{:,.2f}", "risk_score": "{:.2f}", "anomaly_score": "{:.2f}"}),
                    use_container_width=True,
                    height=350
                )

    # TAB 5: LIQUIDITY FORECAST
    with tab5:
        st.subheader("Treasury Cash Trajectory & Depository Position Projections")
        cash_df = ctx.liquidity.get("cash")
        sec_df = ctx.liquidity.get("securities")
        alerts_df = ctx.liquidity.get("alerts")

        if alerts_df is not None and not alerts_df.empty:
            st.warning(f"🚨 **{len(alerts_df)} Active Liquidity Alerts Triggered**")
            st.dataframe(alerts_df, use_container_width=True)

        if cash_df is not None and not cash_df.empty:
            st.markdown("##### Projected Closing Treasury Cash vs Minimum Regulatory Buffer")
            fig_cash = go.Figure()
            fig_cash.add_trace(go.Scatter(
                x=cash_df["settlement_date"],
                y=cash_df["projected_closing_cash"],
                mode="lines+markers",
                name="Projected Treasury Cash",
                line=dict(color="#2980B9", width=3)
            ))
            fig_cash.add_trace(go.Scatter(
                x=cash_df["settlement_date"],
                y=[MIN_CASH_BUFFER] * len(cash_df),
                mode="lines",
                name="Minimum Cash Buffer (₹50L)",
                line=dict(color="#C0392B", dash="dash", width=2)
            ))
            fig_cash.update_layout(yaxis_title="INR (₹)", hovermode="x unified", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_cash, use_container_width=True)

        if sec_df is not None and not sec_df.empty:
            st.markdown("##### Securities Depository Settlement Balances")
            short_secs = sec_df[sec_df["short_flag"]] if "short_flag" in sec_df.columns else pd.DataFrame()
            if not short_secs.empty:
                st.error(f"Depository short position detected on {len(short_secs)} instruments!")
            st.dataframe(
                sec_df.style.format({
                    "opening_quantity": "{:,}",
                    "net_settling_quantity": "{:,}",
                    "projected_quantity": "{:,}"
                }),
                use_container_width=True,
                height=250
            )

    # TAB 6: AUDIT & LOGS
    with tab6:
        st.subheader("Tamper-Evident SHA-256 Audit Blockchain")
        chain_valid, bad_seq = verify_chain(ctx.audit)

        col_a1, col_a2 = st.columns([1.5, 1])
        with col_a1:
            if chain_valid:
                st.success(f"Audit Trail Integrity: **VERIFIED VALID** ({len(ctx.audit):,} hashed blocks)")
            else:
                st.error(f"Audit Trail Compromised! Verification failed at Block Sequence #{bad_seq}")

        with col_a2:
            if st.button("🧪 Execute Tamper Test (Simulate Attack)"):
                if ctx.audit:
                    target_idx = len(ctx.audit) // 2
                    ctx.audit[target_idx]["detail"] += " [UNAUTHORIZED_MODIFICATION]"
                    st.rerun()

        st.markdown("##### Recent Audit Chain Blocks")
        if ctx.audit:
            audit_display_df = pd.DataFrame(ctx.audit[-50:])
            st.dataframe(audit_display_df[["seq", "ts", "agent", "event", "detail", "hash", "prev_hash"]], use_container_width=True)

        st.markdown("##### Agent Execution Reasoning Stream (Last 150 events)")
        if ctx.agent_log:
            st.dataframe(pd.DataFrame(ctx.agent_log[-150:]), use_container_width=True, height=300)

    # TAB 7: MODEL EVALUATION
    with tab7:
        st.subheader("Reconciliation Accuracy & Latency Benchmark")
        if truth_df is not None and not truth_df.empty:
            eval_metrics = evaluate(ctx, truth_df)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Break Precision", f"{eval_metrics['precision']:.1%}", help="TP / (TP + FP)")
            m2.metric("Break Recall", f"{eval_metrics['recall']:.1%}", help="TP / (TP + FN)")
            m3.metric("F1 Score", f"{eval_metrics['f1']:.3f}")
            m4.metric("Auto-Resolution Accuracy", f"{eval_metrics['auto_resolution_accuracy']:.1%}",
                      delta=f"{eval_metrics['wrong_auto_resolutions']} errors", delta_color="inverse")

            l1, l2, l3, l4 = st.columns(4)
            l1.metric("Mean Latency", f"{eval_metrics['latency_mean_sec']:.1f}s")
            l2.metric("Median Latency", f"{eval_metrics['latency_median_sec']:.1f}s")
            l3.metric("P95 Detection Latency", f"{eval_metrics['latency_p95_sec']:.1f}s")
            l4.metric("Escalation Rate", f"{eval_metrics['escalation_rate']:.1%}")

            st.markdown("---")
            st.markdown("##### Accuracy & Detection Performance by Break Class")
            st.dataframe(eval_metrics["per_type"], use_container_width=True)
        else:
            st.info("ground_truth.csv was not detected. Benchmark metrics skipped.")

# ==============================================================================
# SECTION 12: COMMAND LINE INTERFACE (CLI)
# ==============================================================================

def run_cli():
    """Command-line interface for headless training and pipeline verification."""
    parser = argparse.ArgumentParser(description="Multi-Agent Trade Verification & Risk Engine")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic CSV datasets")
    parser.add_argument("--run", action="store_true", help="Execute verification pipeline and display benchmark evaluation")
    parser.add_argument("--dir", type=str, default=DATA_DIR, help="Data directory path")
    args = parser.parse_args()

    if args.generate:
        generate_data(seed=SEED, n_trades=N_TRADES, out_dir=args.dir)
        return

    if args.run:
        print("=" * 80)
        print("  AUTONOMOUS MULTI-AGENT TRADE VERIFICATION & RISK PREV ENGINE")
        print("=" * 80)
        print(f"Loading data feeds from: {args.dir} ...")

        ctx = run_pipeline(data_dir=args.dir, kill_switch=KILL_SWITCH, auto_limit=AUTO_RESOLVE_LIMIT)

        truth_df = ctx.raw.get("truth")
        if truth_df is not None:
            metrics = evaluate(ctx, truth_df)
            print("\n" + "=" * 35 + " EVALUATION " + "=" * 35)
            print(f"Total Trades Evaluated:      {metrics['total_trades']}")
            print(f"Total Breaks Detected:       {metrics['total_breaks']}")
            print(f"Precision:                   {metrics['precision']:.4f} (Target >= 0.95)")
            print(f"Recall:                      {metrics['recall']:.4f} (Target >= 0.95)")
            print(f"F1 Score:                    {metrics['f1']:.4f}")
            print(f"Auto-Resolution Accuracy:    {metrics['auto_resolution_accuracy']:.2%} (Target = 100%)")
            print(f"Wrong Auto-Resolutions:      {metrics['wrong_auto_resolutions']} (Target = 0)")
            print(f"Escalation Rate:             {metrics['escalation_rate']:.2%}")
            print(f"Detection Latency (Mean):    {metrics['latency_mean_sec']:.1f}s")
            print(f"Detection Latency (Median):  {metrics['latency_median_sec']:.1f}s")
            print(f"Detection Latency (P95):     {metrics['latency_p95_sec']:.1f}s")
            print(f"High-Risk Trades:            {metrics['high_risk_trades']}")
            print("\nBreakdown by Break Type:")
            print(metrics["per_type"].to_string(index=False))

            valid, seq = verify_chain(ctx.audit)
            print(f"\nAudit Chain Integrity:       {'VALID ✅' if valid else f'CORRUPTED at #{seq} ❌'}")
            print("=" * 82)
        else:
            print("\nPipeline run complete. No ground_truth.csv detected for evaluation.")
        return

    parser.print_help()

# ==============================================================================
# SECTION 2: STREAMLIT RUNTIME DETECTION & ENTRYPOINT
# ==============================================================================

try:
    from streamlit.runtime import exists as _st_exists
    RUNNING_IN_STREAMLIT = _st_exists()
except Exception:
    RUNNING_IN_STREAMLIT = False

if RUNNING_IN_STREAMLIT:
    run_ui()
elif __name__ == "__main__":
    run_cli()