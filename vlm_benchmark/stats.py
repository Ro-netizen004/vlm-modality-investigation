"""Statistical tests for paired benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scipy.stats import mcnemar as scipy_mcnemar
except ImportError:  # pragma: no cover
    scipy_mcnemar = None


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(
        lambda x: str(x).strip().lower() in {"1", "true", "yes", "t"}
        if pd.notna(x)
        else False
    )


def _align_paired(
    df_a: pd.DataFrame,
    col_a: str,
    df_b: pd.DataFrame,
    col_b: str,
    id_col: str = "problem_id",
) -> tuple[pd.Series, pd.Series, int]:
    if id_col not in df_a.columns or id_col not in df_b.columns:
        raise ValueError(f"Both frames need column {id_col!r}")

    a = df_a[[id_col, col_a]].drop_duplicates(subset=[id_col], keep="last")
    b = df_b[[id_col, col_b]].drop_duplicates(subset=[id_col], keep="last")
    merged = a.merge(b, on=id_col, suffixes=("_a", "_b"), how="inner")

    if len(merged) == 0:
        raise ValueError("No overlapping problem_id values between the two inputs.")

    correct_a = _as_bool(merged[col_a])
    correct_b = _as_bool(merged[col_b])
    return correct_a, correct_b, len(merged)


def mcnemar_table(correct_a: pd.Series, correct_b: pd.Series) -> dict[str, int]:
    """
    Build 2x2 table for scipy.stats.mcnemar.

    Rows = condition A, cols = condition B (0 = correct, 1 = wrong).
    """
    a = _as_bool(correct_a).reset_index(drop=True)
    b = _as_bool(correct_b).reset_index(drop=True)
    n_cc = int(((a) & (b)).sum())
    n_cw = int(((a) & (~b)).sum())  # A correct, B wrong
    n_wc = int(((~a) & (b)).sum())  # A wrong, B correct
    n_ww = int(((~a) & (~b)).sum())
    return {
        "both_correct": n_cc,
        "a_correct_b_wrong": n_cw,
        "a_wrong_b_correct": n_wc,
        "both_wrong": n_ww,
        "n": len(a),
    }


def mcnemar_test(
    correct_a: pd.Series,
    correct_b: pd.Series,
    *,
    label_a: str = "A",
    label_b: str = "B",
    exact: bool | None = None,
) -> dict[str, Any]:
    """
    McNemar test on paired binary outcomes (same problems, two conditions).

    Uses scipy.stats.mcnemar on the discordant pairs (A wrong/B correct vs A correct/B wrong).
    """
    if scipy_mcnemar is None:
        raise ImportError("McNemar test requires scipy. Install with: pip install scipy")

    counts = mcnemar_table(correct_a, correct_b)
    n = counts["n"]
    b = counts["a_wrong_b_correct"]  # discordant: only B correct
    c = counts["a_correct_b_wrong"]  # discordant: only A correct

    table = [
        [counts["both_correct"], c],
        [b, counts["both_wrong"]],
    ]

    if exact is None:
        exact = (b + c) < 25

    result = scipy_mcnemar(table, exact=exact)
    acc_a = float(_as_bool(correct_a).mean())
    acc_b = float(_as_bool(correct_b).mean())

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": n,
        "accuracy_a": round(acc_a, 4),
        "accuracy_b": round(acc_b, 4),
        "accuracy_delta_b_minus_a": round(acc_b - acc_a, 4),
        "table": counts,
        "discordant_b_only": b,
        "discordant_a_only": c,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "exact_test": exact,
    }


def mcnemar_from_csv(
    csv_a: str | Path,
    col_a: str,
    csv_b: str | Path | None = None,
    col_b: str | None = None,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    id_col: str = "problem_id",
) -> dict[str, Any]:
    """
    Compare two conditions from one or two result CSVs.

    If csv_b is None, both columns are read from csv_a (wide legacy format).
    """
    path_a = Path(csv_a)
    df_a = pd.read_csv(path_a)

    if csv_b is None:
        if col_b is None:
            raise ValueError("col_b is required when using a single CSV with two columns.")
        df_b = df_a
        path_b = path_a
    else:
        path_b = Path(csv_b)
        df_b = pd.read_csv(path_b) if path_b != path_a else df_a

    if col_a not in df_a.columns:
        raise ValueError(f"Column {col_a!r} not in {path_a}")
    if col_b not in df_b.columns:
        raise ValueError(f"Column {col_b!r} not in {path_b}")

    correct_a, correct_b, n = _align_paired(df_a, col_a, df_b, col_b, id_col=id_col)
    return mcnemar_test(
        correct_a,
        correct_b,
        label_a=label_a or col_a,
        label_b=label_b or (col_b or col_a),
    )


def format_mcnemar_report(result: dict[str, Any]) -> str:
    """Human-readable summary for logs or papers."""
    t = result["table"]
    lines = [
        f"McNemar test: {result['label_a']} vs {result['label_b']}",
        f"  Paired n        : {result['n_paired']}",
        f"  Accuracy A      : {result['accuracy_a']:.1%}",
        f"  Accuracy B      : {result['accuracy_b']:.1%}",
        f"  Delta (B - A)   : {result['accuracy_delta_b_minus_a']:+.1%}",
        f"  Both correct    : {t['both_correct']}",
        f"  A only correct  : {t['a_correct_b_wrong']}",
        f"  B only correct  : {t['a_wrong_b_correct']}",
        f"  Both wrong      : {t['both_wrong']}",
        f"  Statistic       : {result['statistic']:.4f}",
        f"  p-value         : {result['pvalue']:.4g}",
        f"  Exact test      : {result['exact_test']}",
    ]
    sig = "significant" if result["pvalue"] < 0.05 else "not significant"
    lines.append(f"  (alpha=0.05: {sig})")
    return "\n".join(lines)
