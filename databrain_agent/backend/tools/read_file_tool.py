"""
Read File tool for DataBrain AI Agent.

Loads research data from a file or directory using the universal_loader.
Supports .csv, .xlsx, .xls, .json, .mat, .txt. For directories, loads every
valid file and optionally combines them.

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""
from pathlib import Path
from typing import Any, Dict, Optional

from langchain.tools import BaseTool
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class ReadFileTool(BaseTool):
    """Tool for loading research data from files or directories."""

    name = "read_file"
    description = """Load research data from a file or directory.
Use this when the user asks to read, load, or analyze data from a path.

Input: A file path or directory path (string).

- **File**: Loads the file using the appropriate format (.csv, .xlsx, .mat, .txt, .json).
- **Directory**: Loads every valid research file inside and combines them into one dataset.

Supported formats: .csv, .xlsx, .xls, .json, .mat, .txt

Returns a summary with columns, row count, and sample data (first 10 rows)."""

    def _run(
        self,
        path: str = "",
        tool_input: Optional[str | Dict[str, Any]] = None,
        combine_directories: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        """Load file(s) and return summary as JSON."""
        # Accept path from kwargs (agent passes {"path": "..."}) or tool_input
        if not path and tool_input is not None:
            if isinstance(tool_input, dict):
                path = tool_input.get("path") or tool_input.get("file_path") or tool_input.get("input", "")
                combine_directories = combine_directories or tool_input.get("combine_directories")
            else:
                path = str(tool_input)

        path = str(path or "").strip().strip('"').strip("'")
        if not path:
            return json.dumps({"error": "Path is required", "hint": "Provide a file or directory path."})

        path_obj = Path(path)
        if not path_obj.exists():
            return json.dumps({
                "error": f"Path not found: {path}",
                "hint": "Check the path exists and is accessible.",
            })

        try:
            from databrain_agent.backend.data_cleaner import universal_loader, SUPPORTED_RESEARCH_EXTENSIONS

            combine = combine_directories if combine_directories is not None else True
            if isinstance(combine, str):
                combine = str(combine).lower() in ("true", "1", "yes")
            result = universal_loader(path_obj, combine_directories=combine)

            if isinstance(result, dict):
                # Multiple DataFrames (directory, combine_directories=False)
                summary = {
                    "path": path,
                    "type": "directory",
                    "files_loaded": list(result.keys()),
                    "datasets": {},
                }
                for name, df in result.items():
                    summary["datasets"][name] = _df_summary(df)
                return json.dumps(summary, default=str)

            # Single DataFrame
            df = result
            summary = {
                "path": path,
                "type": "file" if path_obj.is_file() else "directory_combined",
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "head": df.head(10).to_dict(orient="records"),
                "dtypes": {str(c): str(d) for c, d in df.dtypes.items()},
            }
            if path_obj.is_dir():
                if "_source_file" in df.columns:
                    summary["files_combined"] = df["_source_file"].unique().tolist()
            return json.dumps(summary, default=str)

        except FileNotFoundError as e:
            return json.dumps({"error": str(e), "hint": "Check the path exists."})
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.exception("Read file tool failed")
            from databrain_agent.backend.data_cleaner import SUPPORTED_RESEARCH_EXTENSIONS
            return json.dumps({
                "error": f"Failed to load: {str(e)}",
                "hint": f"Supported formats: {sorted(SUPPORTED_RESEARCH_EXTENSIONS)}",
            })

    async def _arun(self, path: str = "", tool_input: Optional[str | Dict[str, Any]] = None, **kwargs) -> str:
        """Async execution."""
        return self._run(path=path, tool_input=tool_input, **kwargs)


def _df_summary(df: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
    """Build a compact summary of a DataFrame."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "head": df.head(max_rows).to_dict(orient="records"),
    }
