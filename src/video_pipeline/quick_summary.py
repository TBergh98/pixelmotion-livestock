from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .analytics import InterDayMetrics


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "-"
    return str(value)


def _trend_label(avg_daily_change: float | None) -> str:
    if avg_daily_change is None:
        return "unknown"
    if avg_daily_change > 0.001:
        return "improving"
    if avg_daily_change < -0.001:
        return "declining"
    return "stable"


def _build_rows(daily_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in daily_summaries:
        rows.append(
            {
                "date": day.get("date"),
                "primary_value": day.get("primary_value"),
                "median": day.get("median"),
                "mean": day.get("mean"),
                "std": day.get("std"),
                "active_ratio": day.get("active_ratio"),
                "amplitude": day.get("amplitude"),
                "coeff_variation": day.get("coeff_variation"),
                "iqr": day.get("iqr"),
                "p90_p10_spread": day.get("p90_p10_spread"),
                "outlier_ratio": day.get("outlier_ratio"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "primary_value",
        "median",
        "mean",
        "std",
        "active_ratio",
        "amplitude",
        "coeff_variation",
        "iqr",
        "p90_p10_spread",
        "outlier_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    *,
    group_id: str,
    primary_metric: str,
    rows: list[dict[str, Any]],
    trend_data: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latest = rows[-1] if rows else None

    avg_daily_change = _to_float(trend_data.get("avg_daily_change"))
    trend_label = _trend_label(avg_daily_change)

    best_row = max(rows, key=lambda r: _to_float(r.get("primary_value")) or float("-inf")) if rows else None
    worst_row = min(rows, key=lambda r: _to_float(r.get("primary_value")) or float("inf")) if rows else None

    lines: list[str] = []
    lines.append(f"# Quick Summary - {group_id}")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append(f"Primary metric: {primary_metric}")
    lines.append(f"Days analyzed: {_fmt(trend_data.get('daily_count'))}")
    lines.append(f"Average daily change: {_fmt(avg_daily_change)} ({trend_label})")
    lines.append("")

    lines.append("## Latest Day")
    lines.append("")
    if latest is None:
        lines.append("No daily summaries available.")
    else:
        lines.append(f"- Date: {latest.get('date', '-')}")
        lines.append(f"- Primary value: {_fmt(latest.get('primary_value'))}")
        lines.append(f"- Active ratio: {_fmt(latest.get('active_ratio'))}")
        lines.append(f"- Amplitude: {_fmt(latest.get('amplitude'))}")
        lines.append(f"- Variability (CV): {_fmt(latest.get('coeff_variation'))}")
        lines.append(f"- IQR: {_fmt(latest.get('iqr'))}")
        lines.append(f"- P90-P10 spread: {_fmt(latest.get('p90_p10_spread'))}")
        lines.append(f"- Outlier ratio: {_fmt(latest.get('outlier_ratio'))}")

    lines.append("")
    lines.append("## Best/Worst Day")
    lines.append("")
    if best_row is None or worst_row is None:
        lines.append("Not available.")
    else:
        lines.append(f"- Best day: {best_row.get('date', '-')} (primary={_fmt(best_row.get('primary_value'))})")
        lines.append(f"- Worst day: {worst_row.get('date', '-')} (primary={_fmt(worst_row.get('primary_value'))})")

    lines.append("")
    lines.append("## Daily KPI Table")
    lines.append("")
    lines.append("| date | primary | median | std | active_ratio | amplitude | CV | IQR | p90-p10 | outlier_ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            f"{row.get('date', '-')} | "
            f"{_fmt(row.get('primary_value'))} | "
            f"{_fmt(row.get('median'))} | "
            f"{_fmt(row.get('std'))} | "
            f"{_fmt(row.get('active_ratio'))} | "
            f"{_fmt(row.get('amplitude'))} | "
            f"{_fmt(row.get('coeff_variation'))} | "
            f"{_fmt(row.get('iqr'))} | "
            f"{_fmt(row.get('p90_p10_spread'))} | "
            f"{_fmt(row.get('outlier_ratio'))} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_quick_summary_files(
    *,
    interday_metrics: InterDayMetrics,
    output_dir: Path,
    formats: dict[str, bool],
) -> dict[str, str]:
    """Generate quick-view summary artifacts for one group.

    Returns a dict with generated artifact paths by type.
    """
    rows = _build_rows(interday_metrics.daily_summaries)

    artifacts: dict[str, str] = {}
    if formats.get("csv", True):
        csv_path = output_dir / "quick_summary.csv"
        _write_csv(csv_path, rows)
        artifacts["csv"] = str(csv_path)

    if formats.get("markdown", True):
        md_path = output_dir / "quick_summary.md"
        _write_markdown(
            md_path,
            group_id=interday_metrics.group_id,
            primary_metric=interday_metrics.primary_metric,
            rows=rows,
            trend_data=interday_metrics.trend_data,
        )
        artifacts["markdown"] = str(md_path)

    return artifacts
