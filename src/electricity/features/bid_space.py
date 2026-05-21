from __future__ import annotations

import numpy as np
import pandas as pd

PRED_COLS = {
    "load": "系统负荷预测值",
    "renewable": "风光总加预测值",
    "tie_line": "联络线预测值",
    "wind": "风电预测值",
    "solar": "光伏预测值",
    "hydro": "水电预测值",
    "non_market": "非市场化机组预测值",
}
ACTUAL_COLS = {
    "load": "系统负荷实际值",
    "renewable": "风光总加实际值",
    "tie_line": "联络线实际值",
    "wind": "风电实际值",
    "solar": "光伏实际值",
    "hydro": "水电实际值",
    "non_market": "非市场化机组实际值",
}


def add_bid_space_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bid_space_fct"] = (
        out[PRED_COLS["load"]]
        - out[PRED_COLS["renewable"]]
        - out[PRED_COLS["tie_line"]]
        - out[PRED_COLS["hydro"]]
        - out[PRED_COLS["non_market"]]
    )
    out["bid_space_act"] = (
        out[ACTUAL_COLS["load"]]
        - out[ACTUAL_COLS["renewable"]]
        - out[ACTUAL_COLS["tie_line"]]
        - out[ACTUAL_COLS["hydro"]]
        - out[ACTUAL_COLS["non_market"]]
    )
    out["bid_space_error"] = out["bid_space_fct"] - out["bid_space_act"]
    out["price"] = out["A"]
    out["date"] = out["times"].dt.normalize()
    out["slot"] = out["times"].dt.hour * 4 + out["times"].dt.minute // 15
    out["month"] = out["times"].dt.month
    out["hour"] = out["times"].dt.hour
    return out


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)

    def fmt(value: object) -> str:
        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
        if isinstance(value, np.datetime64):
            return pd.Timestamp(value).date().isoformat()
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return format(value, floatfmt)
        return str(value)

    rows = [[fmt(row[col]) for col in cols] for _, row in df.iterrows()]
    out = ["| " + " | ".join(cols) + " |"]
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)
