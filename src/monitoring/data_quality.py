from typing import Any, Dict, List, Optional
import pandas as pd


DEFAULT_NUMERIC_BOUNDS = {
    "Price": (0.0, 1_000_000.0),
    "price": (0.0, 1_000_000.0),
}


def validate_batch_quality(df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
    required_columns: List[str] = rules.get("required_columns", [])
    max_null_rate: float = float(rules.get("max_null_rate", 0.4))
    numeric_bounds: Dict[str, Any] = rules.get("numeric_bounds", DEFAULT_NUMERIC_BOUNDS)

    issues: List[Dict[str, Any]] = []

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        issues.append({"type": "missing_columns", "columns": missing_cols})

    for col in required_columns:
        if col not in df.columns:
            continue
        null_rate = float(df[col].isna().mean())
        if null_rate > max_null_rate:
            issues.append(
                {
                    "type": "high_null_rate",
                    "column": col,
                    "null_rate": null_rate,
                    "threshold": max_null_rate,
                }
            )

    for col, bounds in numeric_bounds.items():
        if col not in df.columns:
            continue
        lo = float(bounds[0])
        hi = float(bounds[1])
        vals = pd.to_numeric(df[col], errors="coerce")
        invalid = vals[(vals < lo) | (vals > hi)]
        if len(invalid) > 0:
            issues.append(
                {
                    "type": "out_of_bounds",
                    "column": col,
                    "count": int(len(invalid)),
                    "min_allowed": lo,
                    "max_allowed": hi,
                }
            )

    reject_on = set(rules.get("reject_on", ["missing_columns", "out_of_bounds"]))
    should_reject = any(i.get("type") in reject_on for i in issues)

    policy = "allow"
    if should_reject:
        policy = "reject"
    elif issues:
        policy = "quarantine"

    return {
        "policy": policy,
        "issues": issues,
        "rows": int(len(df)),
    }
