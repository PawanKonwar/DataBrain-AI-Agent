"""
Research Plotter - Sophisticated overlay visualization for research data.

Creates overlay plots from multiple files (Load vs Displacement, Time vs Value)
with automatic column detection, units, legend, and large-data downsampling.

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import BaseTool
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

MAX_ROWS = 10_000
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Fuzzy column patterns: (pattern_regex, role) - order matters for precedence
LOAD_PATTERNS = [
    r"\bload\b", r"\bforce\b", r"\bstress\b", r"\bF\b", r"\bkN\b",
    r"peak_load", r"max_load", r"load_", r"_load",
]
DISPLACEMENT_PATTERNS = [
    r"\bdisp(?:lacement)?\b", r"\bstrain\b", r"\bextension\b", r"\bdeflection\b",
    r"\bmm\b", r"\bdeformation\b",
]
TIME_PATTERNS = [
    r"\btime\b", r"\bt\b", r"\bsec\b", r"\bs\b", r"\btimestamp\b",
    r"\bdatetime\b", r"^t_", r"_time",
]
VALUE_PATTERNS = [
    r"\bvalue\b", r"\bval\b", r"\bdata\b", r"\breading\b",
    r"\bmeasurement\b", r"\bamplitude\b", r"\bvoltage\b", r"\bcurrent\b",
]


def _extract_units(col_name: str) -> Optional[str]:
    """Extract units from column name, e.g. 'Load (kN)' -> 'kN', 'Stress [MPa]' -> 'MPa'."""
    if not isinstance(col_name, str):
        return None
    m = re.search(r"\(([^)]+)\)|\[([^\]]+)\]", col_name)
    if m:
        return (m.group(1) or m.group(2) or "").strip()
    return None


def _fuzzy_match_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    """Find first column that matches any pattern (case-insensitive)."""
    cols_lower = {str(c).lower(): c for c in columns}
    for pat in patterns:
        regex = re.compile(pat, re.I)
        for lower_name, orig_name in cols_lower.items():
            if regex.search(lower_name):
                return orig_name
    return None


def _detect_axes_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect x and y columns for overlay plot.
    Returns (x_col, y_col) - e.g. (Time, Load) or (Displacement, Load).
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return None, None

    time_col = _fuzzy_match_column(numeric_cols, TIME_PATTERNS)
    load_col = _fuzzy_match_column(numeric_cols, LOAD_PATTERNS)
    disp_col = _fuzzy_match_column(numeric_cols, DISPLACEMENT_PATTERNS)
    value_col = _fuzzy_match_column(numeric_cols, VALUE_PATTERNS)

    # Prefer Time vs Load/Displacement; else Displacement vs Load
    if time_col and (load_col or disp_col):
        x_col = time_col
        y_col = load_col or disp_col
    elif disp_col and load_col:
        x_col = disp_col
        y_col = load_col
    elif time_col and value_col:
        x_col = time_col
        y_col = value_col
    else:
        # Fallback: first two numeric columns
        x_col, y_col = numeric_cols[0], numeric_cols[1]

    return x_col, y_col


def _downsample(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Downsample to max_rows if needed."""
    if len(df) <= max_rows:
        return df
    step = max(1, len(df) // max_rows)
    return df.iloc[::step].reset_index(drop=True)


def _create_overlay_plot(
    data_list: List[Tuple[str, pd.DataFrame]],
    x_col: str,
    y_col: str,
    x_units: Optional[str] = None,
    y_units: Optional[str] = None,
    title: str = "Research Data Overlay",
    output_path: Path = None,
    format: str = "html",
) -> Path:
    """Create overlay plot using Plotly (preferred) or Matplotlib."""
    try:
        import plotly.graph_objects as go
        use_plotly = True
    except ImportError:
        use_plotly = False

    x_label = f"{x_col} ({x_units})" if x_units else x_col
    y_label = f"{y_col} ({y_units})" if y_units else y_col

    if use_plotly:
        fig = go.Figure()
        for i, (filename, df) in enumerate(data_list):
            df = _downsample(df)
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=df[x_col].values,
                    y=df[y_col].values,
                    mode="lines",
                    name=filename,
                    line=dict(color=color, width=1.5),
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title="File",
            template="plotly_white",
            height=500,
        )
        ext = "html" if format == "html" else "png"
        path = output_path or Path(tempfile.gettempdir()) / f"research_overlay_{id(fig)}.{ext}"
        if format == "html":
            fig.write_html(str(path))
        else:
            try:
                fig.write_image(str(path), scale=2)
            except Exception:
                path = path.with_suffix(".html")
                fig.write_html(str(path))
        return path

    # Matplotlib fallback
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (filename, df) in enumerate(data_list):
        df = _downsample(df)
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ax.plot(df[x_col].values, df[y_col].values, label=filename, color=color, linewidth=1.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="File", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ext = "png"
    path = output_path or Path(tempfile.gettempdir()) / f"research_overlay_{id(ax)}.{ext}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _create_master_summary_plot(
    master_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Summary Across Files",
    output_path: Path = None,
    format: str = "html",
) -> Path:
    """Create bar/line plot from Master DataFrame (filename vs stat)."""
    try:
        import plotly.graph_objects as go
        use_plotly = True
    except ImportError:
        use_plotly = False

    if use_plotly:
        fig = go.Figure(data=[go.Bar(x=master_df[x_col].astype(str), y=master_df[y_col].values)])
        fig.update_layout(title=title, xaxis_tickangle=-45, height=500, xaxis_title=x_col, yaxis_title=y_col)
        ext = "html" if format == "html" else "png"
        path = output_path or Path(tempfile.gettempdir()) / f"master_plot_{id(fig)}.{ext}"
        if format == "html":
            fig.write_html(str(path))
        else:
            try:
                fig.write_image(str(path), scale=2)
            except Exception:
                path = path.with_suffix(".html")
                fig.write_html(str(path))
        return path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(master_df[x_col].astype(str), master_df[y_col].values)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = output_path or Path(tempfile.gettempdir()) / f"master_plot_{id(ax)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


class ResearchPlotterTool(BaseTool):
    """Sophisticated overlay plotter for research data from multiple files."""

    name = "research_plotter"
    description = """Create overlay plots from research data (Load vs Displacement, Time vs Value).
Use when the user wants to compare/overlay data across multiple files or plot trends.

Input options:
- folder_path: Load all files from folder, auto-detect columns, create overlay (each file = colored line).
- master_dataframe: JSON from batch_research_summarizer - plot summary stats (e.g. filename vs load_max).
- output_format: 'html' (interactive) or 'png' (high-res). Default: html.

Auto-detects Load, Displacement, Time, Value columns. Downsamples if >10k rows. Returns file path."""

    def _run(
        self,
        folder_path: str = "",
        path: str = "",
        master_dataframe: Optional[str] = None,
        output_format: str = "html",
        tool_input: Optional[object] = None,
        **kwargs: Any,
    ) -> str:
        """Create overlay or summary plot."""
        folder = folder_path or path or kwargs.get("folder_path") or kwargs.get("path", "")
        master_dataframe = master_dataframe or kwargs.get("master_dataframe") or kwargs.get("data")
        output_format = (output_format or kwargs.get("output_format") or "html").lower()
        if isinstance(tool_input, dict):
            folder = folder or tool_input.get("folder_path") or tool_input.get("path") or tool_input.get("input", "")
            master_dataframe = master_dataframe or tool_input.get("master_dataframe") or tool_input.get("data")
            output_format = (output_format or tool_input.get("output_format") or "html").lower()

        folder = str(folder or "").strip().strip('"').strip("'")
        output_format = str(output_format or "html").strip().lower()
        if output_format not in ("html", "png"):
            output_format = "html"

        # Mode 1: Master DataFrame (summary stats)
        if master_dataframe:
            try:
                if isinstance(master_dataframe, str):
                    data = json.loads(master_dataframe)
                else:
                    data = master_dataframe
                if isinstance(data, dict) and "master_dataframe" in data:
                    records = data["master_dataframe"]
                elif isinstance(data, list):
                    records = data
                else:
                    records = data if isinstance(data, list) else [data]
                master_df = pd.DataFrame(records)
                if master_df.empty or "filename" not in master_df.columns:
                    return json.dumps({"error": "Master DataFrame must have 'filename' column"})

                # Find y column: prefer *_max columns for trends
                numeric_cols = [c for c in master_df.columns if c not in ("filename", "row_count", "columns")]
                max_cols = [c for c in numeric_cols if c.endswith("_max")]
                y_col = max_cols[0] if max_cols else (numeric_cols[0] if numeric_cols else None)
                if not y_col:
                    return json.dumps({"error": "No numeric summary columns found in Master DataFrame"})

                out_path = _create_master_summary_plot(
                    master_df, x_col="filename", y_col=y_col,
                    title=f"{y_col} Across Files", format=output_format,
                )
                return json.dumps({
                    "status": "success",
                    "output_path": str(out_path.absolute()),
                    "format": output_format,
                    "type": "master_summary",
                    "message": f"Plot saved to {out_path}",
                })
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid master_dataframe JSON: {e}"})
            except Exception as e:
                logger.exception("Master plot failed")
                return json.dumps({"error": str(e)})

        # Mode 2: Folder - load files and overlay
        if not folder:
            return json.dumps({
                "error": "folder_path or master_dataframe is required",
                "hint": "Provide folder_path to overlay files, or master_dataframe (JSON) for summary plot.",
            })

        path_obj = Path(folder)
        if not path_obj.exists() or not path_obj.is_dir():
            return json.dumps({"error": f"Folder not found: {folder}"})

        try:
            from databrain_agent.backend.data_cleaner import universal_loader, SUPPORTED_RESEARCH_EXTENSIONS

            loaded = universal_loader(path_obj, combine_directories=False, health_clean=True)
            if not isinstance(loaded, dict) or not loaded:
                return json.dumps({
                    "error": f"No valid files in {folder}",
                    "hint": f"Supported: {sorted(SUPPORTED_RESEARCH_EXTENSIONS)}",
                })

            data_list: List[Tuple[str, pd.DataFrame]] = []
            x_col, y_col = None, None
            x_units, y_units = None, None

            for filename, df in loaded.items():
                if df.empty or len(df.columns) < 2:
                    continue
                data_list.append((filename, df))
                xc, yc = _detect_axes_columns(df)
                if xc and yc:
                    x_col, y_col = xc, yc
                    x_units = _extract_units(xc) or x_units
                    y_units = _extract_units(yc) or y_units

            if not data_list:
                return json.dumps({"error": "No plottable data found in folder"})
            if not x_col or not y_col:
                return json.dumps({
                    "error": "Could not auto-detect Load/Displacement or Time/Value columns",
                    "hint": "Ensure files have numeric columns matching load, displacement, time, or value.",
                })

            out_path = _create_overlay_plot(
                data_list, x_col, y_col, x_units, y_units,
                title="Research Data Overlay", format=output_format,
            )
            return json.dumps({
                "status": "success",
                "output_path": str(out_path.absolute()),
                "format": output_format,
                "type": "overlay",
                "x_column": x_col,
                "y_column": y_col,
                "files_plotted": len(data_list),
                "message": f"Plot saved to {out_path}",
            })

        except Exception as e:
            logger.exception("Research plotter failed")
            return json.dumps({"error": str(e)})

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)
