"""
Batch Research Summarizer tool for DataBrain AI Agent.

Iterates through all research files in a folder, extracts column names and
basic stats (mean, max, min) per file, and returns a Master DataFrame where
each row represents one file. Enables questions like "Which sample had the
highest displacement?" or "Plot the peak load trend across all files."

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


def _safe_stat(series: pd.Series, func: str) -> Optional[float]:
    """Compute stat (mean, max, min) safely, return None if all-NaN."""
    try:
        if func == "mean":
            val = series.mean()
        elif func == "max":
            val = series.max()
        elif func == "min":
            val = series.min()
        else:
            return None
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def _build_file_summary_row(filename: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Build one row of the Master DataFrame for a single file."""
    row: Dict[str, Any] = {
        "filename": filename,
        "row_count": len(df),
        "columns": ",".join(str(c) for c in df.columns),
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        safe_col = str(col).replace(" ", "_").replace(".", "_")
        row[f"{safe_col}_mean"] = _safe_stat(df[col], "mean")
        row[f"{safe_col}_max"] = _safe_stat(df[col], "max")
        row[f"{safe_col}_min"] = _safe_stat(df[col], "min")
    return row


class BatchResearchSummarizerTool(BaseTool):
    """Tool for batch summarizing research files in a folder."""

    name = "batch_research_summarizer"
    description = """Summarize all research files in a folder into a Master DataFrame.
Use when the user asks about comparing across many files, e.g.:
- "Which of my 100 samples had the highest displacement?"
- "Plot the peak load trend across all files"
- "Compare stats across my experiment folder"

Input: folder_path (string) - path to the folder containing research files.

For each file, extracts column names and basic stats (mean, max, min) for numeric columns.
Returns a Master DataFrame: one row per file, with columns like filename, row_count, columns,
and for each numeric column: {col}_mean, {col}_max, {col}_min.
Use this output with chart_generator or sql_executor if the Master DataFrame is loaded."""

    def _run(
        self,
        folder_path: str = "",
        path: str = "",
        tool_input: Optional[str | Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Build Master DataFrame from all files in folder."""
        # Accept folder_path or path from kwargs or tool_input
        folder = folder_path or path
        if not folder and tool_input is not None:
            if isinstance(tool_input, dict):
                folder = tool_input.get("folder_path") or tool_input.get("path") or tool_input.get("input", "")
            else:
                folder = str(tool_input)

        folder = str(folder or "").strip().strip('"').strip("'")
        if not folder:
            return json.dumps({
                "error": "Folder path is required",
                "hint": "Provide a folder path containing research files (.csv, .xlsx, .mat, .txt, .json).",
            })

        path_obj = Path(folder)
        if not path_obj.exists():
            return json.dumps({"error": f"Folder not found: {folder}", "hint": "Check the path exists."})
        if not path_obj.is_dir():
            return json.dumps({"error": f"Path is not a directory: {folder}", "hint": "Provide a folder path."})

        try:
            from databrain_agent.backend.data_cleaner import (
                universal_loader,
                SUPPORTED_RESEARCH_EXTENSIONS,
            )

            # Load each file separately (no combining) - universal_loader does health_check_and_clean
            loaded = universal_loader(path_obj, combine_directories=False, health_clean=True)
            if not isinstance(loaded, dict):
                return json.dumps({"error": "Expected per-file dict from loader", "hint": "Retry with a folder path."})

            rows: List[Dict[str, Any]] = []
            for filename, df in loaded.items():
                if df.empty:
                    continue
                rows.append(_build_file_summary_row(filename, df))

            if not rows:
                return json.dumps({
                    "error": f"No valid data found in {folder}",
                    "hint": f"Ensure folder contains supported files: {sorted(SUPPORTED_RESEARCH_EXTENSIONS)}",
                })

            # Build Master DataFrame - ensure all rows have same columns (fill missing with NaN)
            master = pd.DataFrame(rows)
            master = master.fillna(value=np.nan)

            summary = {
                "master_dataframe": master.to_dict(orient="records"),
                "file_count": len(master),
                "columns": list(master.columns),
                "row_count": len(master),
            }
            return json.dumps(summary, default=str)

        except FileNotFoundError as e:
            return json.dumps({"error": str(e), "hint": "Check the folder path exists."})
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.exception("Batch research summarizer failed")
            return json.dumps({
                "error": f"Failed to summarize folder: {str(e)}",
                "hint": f"Supported formats: {sorted(SUPPORTED_RESEARCH_EXTENSIONS)}",
            })

    async def _arun(
        self,
        folder_path: str = "",
        path: str = "",
        tool_input: Optional[str | Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Async execution."""
        return self._run(
            folder_path=folder_path,
            path=path,
            tool_input=tool_input,
            **kwargs,
        )
