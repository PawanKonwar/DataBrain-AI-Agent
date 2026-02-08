"""
Research Data Parsing & Loading - Consolidated module.

Single source of truth for loading research data from multiple formats.
Handles .csv, .xlsx, .xls, .json, .mat, .txt with automatic format detection,
junk-header sniffing, health cleaning, and column normalization.

Key functions:
- load_any_research_file: Load a single file by extension (auto-selects loader).
- universal_loader: Load file or directory; optionally combine or return per-file.
- health_check_and_clean: Fix nulls, non-numeric strings, duplicate timestamps.
- smart_load: Text files with junk headers (lab notes, metadata) - uses sniffer.

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional: scipy for .mat files
try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

_MAT_META_KEYS = {"__header__", "__version__", "__globals__"}

# Common junk values in numeric columns (sensor glitches, lab exports)
_NUMERIC_JUNK = {
    "nan", "na", "n/a", "none", "null", "-", "--", "---", "nd", "n.d.",
    "error", "err", "#value!", "#div/0!", "#ref!", "overflow", "missing",
    "no data", "n.d.", "n.a.", "?", "*", ""
}


def sniff_data_start(
    path: str | Path,
    sample_lines: int = 50,
    max_header_candidates: int = 30,
    delimiters: str = ",\t;| ",
) -> Tuple[str, int]:
    """
    Sniff the file to find where tabular data starts and which delimiter is used.

    Iterates over candidate header rows, scores each by how well it produces a
    valid table (consistent column count, numeric content), and returns the
    best delimiter and number of rows to skip.

    Parameters
    ----------
    path : str or Path
        Path to the file.
    sample_lines : int
        Maximum number of lines to read for analysis.
    max_header_candidates : int
        Maximum number of top rows to consider as potential junk.
    delimiters : str
        Delimiter characters to try (default: comma, tab, semicolon, pipe, space).

    Returns
    -------
    tuple of (delimiter, skiprows)
        delimiter : detected delimiter string (e.g. "," or "\\t")
        skiprows : number of lines to skip before the header row
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [f.readline() for _ in range(sample_lines)]
    lines = [ln for ln in lines if ln is not None and ln.strip()]

    if not lines:
        return ",", 0

    # Detect delimiter from first non-empty lines
    try:
        sample = "\n".join(lines[: min(15, len(lines))])
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=delimiters)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if any("\t" in ln for ln in lines) else ","

    def _score_skip(skip: int) -> float:
        """Score this many header lines: higher = more consistent tabular data."""
        if skip >= len(lines):
            return -1.0
        try:
            buf = io.StringIO("".join(lines))
            df = pd.read_csv(
                buf,
                delimiter=delimiter,
                skiprows=range(skip),
                nrows=min(100, len(lines) - skip),
                encoding="utf-8",
                on_bad_lines="skip",
                engine="python",
            )
        except Exception:
            return -1.0
        if df.empty or len(df.columns) < 2:
            return -1.0
        # Prefer consistent column count and more numeric content
        numeric_ratio = df.apply(pd.to_numeric, errors="coerce").notna().values.mean()
        return float(len(df)) * (1.0 + 0.5 * numeric_ratio)

    best_skip = 0
    best_score = _score_skip(0)
    for skip in range(1, min(max_header_candidates, len(lines))):
        s = _score_skip(skip)
        if s > best_score:
            best_score = s
            best_skip = skip

    return delimiter, best_skip


def _is_timestamp_column(col: pd.Series, col_name: str) -> bool:
    """Heuristic: column looks like timestamps."""
    name_lower = str(col_name).lower()
    if any(kw in name_lower for kw in ("time", "timestamp", "date", "datetime", "t_")):
        return True
    if pd.api.types.is_datetime64_any_dtype(col):
        return True
    # Try parsing first non-null value
    sample = col.dropna().head(5)
    if len(sample) == 0:
        return False
    try:
        pd.to_datetime(sample, errors="raise")
        return True
    except Exception:
        return False


def health_check_and_clean(
    df: pd.DataFrame,
    *,
    fill_nulls: bool = True,
    coerce_numeric: bool = True,
    drop_duplicate_timestamps: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scan and fix common research data "pains".

    - **Null values** (sensor glitches): Forward-fill within numeric columns.
    - **Non-numeric strings in numeric columns**: Coerce junk values to NaN.
    - **Duplicate timestamps**: Drop duplicate rows by timestamp columns, keep first.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (typically right after loading).
    fill_nulls : bool
        If True, forward-fill nulls in numeric columns (default True).
    coerce_numeric : bool
        If True, coerce non-numeric strings to NaN in numeric-like columns (default True).
    drop_duplicate_timestamps : bool
        If True, drop rows with duplicate timestamps (default True).

    Returns
    -------
    tuple of (DataFrame, report)
        df_cleaned : cleaned DataFrame
        report : dict with keys like nulls_filled, coerced_columns, duplicates_dropped
    """
    if df.empty:
        return df.copy(), {"empty": True}

    report: Dict[str, Any] = {
        "nulls_before": int(df.isna().sum().sum()),
        "nulls_filled": 0,
        "coerced_columns": [],
        "duplicates_dropped": 0,
        "timestamp_column_used": None,
    }
    out = df.copy()

    # 1. Non-numeric strings in numeric columns -> coerce to NaN
    if coerce_numeric:
        for col in out.columns:
            s = out[col]
            if s.dtype == object or str(getattr(s.dtype, "name", "")) == "string":
                # Check if column is mostly numeric (or could be)
                numeric_count = pd.to_numeric(s, errors="coerce").notna().sum()
                total = len(s.dropna())
                if total == 0:
                    continue
                numeric_ratio = numeric_count / total
                if numeric_ratio < 0.9:
                    continue  # Likely categorical, skip
                # Coerce; values in _NUMERIC_JUNK become NaN
                def _clean_val(x):
                    if pd.isna(x):
                        return np.nan
                    xs = str(x).strip().lower()
                    if xs in _NUMERIC_JUNK:
                        return np.nan
                    return x

                cleaned = s.map(_clean_val)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
                before_na = s.isna().sum()
                after_na = out[col].isna().sum()
                if after_na > before_na:
                    report["coerced_columns"].append(col)

    # 2. Null values: forward-fill in numeric columns (sensor glitches)
    if fill_nulls:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            before = out[numeric_cols].isna().sum().sum()
            out[numeric_cols] = out[numeric_cols].ffill()
            after = out[numeric_cols].isna().sum().sum()
            report["nulls_filled"] = int(before - after)

    # 3. Duplicate timestamps
    if drop_duplicate_timestamps:
        ts_col = None
        for col in out.columns:
            if _is_timestamp_column(out[col], col):
                ts_col = col
                break
        if ts_col is not None:
            before_len = len(out)
            out = out.drop_duplicates(subset=[ts_col], keep="first").reset_index(drop=True)
            report["duplicates_dropped"] = before_len - len(out)
            report["timestamp_column_used"] = ts_col

    report["nulls_after"] = int(out.isna().sum().sum())
    return out, report


def smart_load(
    path: str | Path,
    *,
    sample_lines: int = 50,
    max_header_candidates: int = 30,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a tabular file, automatically skipping junk headers.

    Uses a sniffer to detect the delimiter and the row where the actual data
    starts (after lab notes, metadata, etc.). Works with CSV, TSV, TXT, and
    other text-based tabular formats.

    Parameters
    ----------
    path : str or Path
        Path to the file.
    sample_lines : int
        Max lines to read when sniffing (default 50).
    max_header_candidates : int
        Max rows to consider as junk header (default 30).
    **kwargs
        Passed to pandas read_csv (e.g. encoding, on_bad_lines).

    Returns
    -------
    pandas.DataFrame
        Loaded table with junk header rows skipped.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        # Excel: try header=0 first; could add Excel-specific sniffer later
        return pd.read_excel(path, **kwargs)

    # Text-based: use sniffer
    delimiter, skiprows = sniff_data_start(
        path,
        sample_lines=sample_lines,
        max_header_candidates=max_header_candidates,
    )

    return pd.read_csv(
        path,
        delimiter=delimiter,
        skiprows=range(skiprows),
        encoding=kwargs.pop("encoding", "utf-8"),
        on_bad_lines=kwargs.pop("on_bad_lines", "skip"),
        engine="python",
        **kwargs,
    )


def smart_load_csv(
    path: str | Path,
    *,
    sample_lines: int = 50,
    max_header_candidates: int = 30,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Alias for smart_load for CSV/TSV/TXT files.

    Convenience wrapper that explicitly targets text tabular files.
    """
    return smart_load(
        path,
        sample_lines=sample_lines,
        max_header_candidates=max_header_candidates,
        **kwargs,
    )


# All research file extensions the agent can load
SUPPORTED_RESEARCH_EXTENSIONS = frozenset({".csv", ".xlsx", ".xls", ".json", ".mat", ".txt"})


def _flatten_mat_value(value: Any, prefix: str = "") -> Dict[str, np.ndarray]:
    """
    Recursively flatten a .mat value into a dict of 1D arrays.
    Nested MATLAB structs become keys like "prefix_child".
    """
    out: Dict[str, np.ndarray] = {}

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return out
        if value.dtype.names:
            for name in value.dtype.names:
                col = value[name]
                if col.ndim > 1:
                    col = col.ravel()
                key = f"{prefix}_{name}" if prefix else name
                out[key] = np.asarray(col).ravel()
        elif value.ndim == 1:
            key = prefix or "value"
            out[key] = np.asarray(value).ravel()
        elif value.ndim == 2:
            for j in range(value.shape[1]):
                key = f"{prefix}_col{j}" if prefix else f"col{j}"
                out[key] = np.asarray(value[:, j]).ravel()
        else:
            key = prefix or "value"
            out[key] = np.asarray(value).ravel()
        return out

    if hasattr(value, "_fieldnames") or (
        isinstance(value, np.ndarray) and hasattr(value.dtype, "names") and value.dtype.names
    ):
        names = getattr(value.dtype, "names", None) or getattr(value, "_fieldnames", ())
        for n in names:
            if n is None:
                continue
            try:
                child = value[n] if hasattr(value, "__getitem__") else getattr(value, n, None)
            except (KeyError, IndexError):
                continue
            if child is None:
                continue
            p = f"{prefix}_{n}" if prefix else n
            out.update(_flatten_mat_value(child, p))
        return out

    if isinstance(value, dict):
        for k, v in value.items():
            if k in _MAT_META_KEYS or v is None:
                continue
            p = f"{prefix}_{k}" if prefix else k
            out.update(_flatten_mat_value(v, p))
        return out

    try:
        arr = np.atleast_1d(value)
        key = prefix or "value"
        out[key] = arr.ravel()
    except Exception:
        pass
    return out


def _load_mat(path: str | Path) -> pd.DataFrame:
    """Load a .mat file and flatten nested structure into a single DataFrame."""
    if loadmat is None:
        raise ImportError("Reading .mat files requires scipy. Install with: pip install scipy")
    raw = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    data = {k: v for k, v in raw.items() if k not in _MAT_META_KEYS}

    flat: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        flat.update(_flatten_mat_value(value, prefix=key))

    if not flat:
        return pd.DataFrame()

    max_len = max(len(v) for v in flat.values())
    for k in flat:
        v = flat[k]
        if len(v) < max_len:
            flat[k] = np.resize(v, max_len)
            flat[k][len(v) :] = np.nan

    return pd.DataFrame(flat)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names, remove Unnamed columns, reset index."""
    df = df.copy()
    df.columns = [str(c).strip() if c is not None else f"col_{i}" for i, c in enumerate(df.columns)]
    # Drop Unnamed columns (from index exports)
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed, errors="ignore")
    df = df.reset_index(drop=True)
    return df


def load_any_research_file(
    path: str | Path,
    *,
    sheet_name: int | str | None = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load any supported research file by automatically choosing the right loader.

    Handles mixed research folders: raw sensor .csv, processed .mat, summaries .xlsx, etc.
    No more "I can't read this" — the agent picks the correct tool by extension.

    Supported formats:
    - .csv, .txt : Smart loader with sniffer (skips junk headers, lab notes)
    - .xlsx, .xls: Excel via pandas
    - .mat      : MATLAB files via scipy (nested structs flattened)
    - .json     : JSON records/arrays via pandas

    Parameters
    ----------
    path : str or Path
        Path to the file.
    sheet_name : int or str or None, default 0
        Sheet for Excel files (0 = first sheet).
    **kwargs
        Passed to underlying loaders where applicable.

    Returns
    -------
    pandas.DataFrame
        Loaded table with normalized column names.

    Raises
    ------
    ValueError
        If file extension is not supported.
    FileNotFoundError
        If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_RESEARCH_EXTENSIONS:
        raise ValueError(
            f"Unsupported research file format: {ext}. "
            f"Supported: {sorted(SUPPORTED_RESEARCH_EXTENSIONS)}"
        )

    df: pd.DataFrame

    if ext == ".mat":
        df = _load_mat(path)

    elif ext in {".csv", ".txt"}:
        # Use smart loader for messy lab exports (junk headers)
        df = smart_load(
            path,
            sample_lines=kwargs.pop("sample_lines", 50),
            max_header_candidates=kwargs.pop("max_header_candidates", 30),
            **kwargs,
        )

    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)

    elif ext == ".json":
        try:
            df = pd.read_json(path, orient="records", **kwargs)
        except (ValueError, TypeError):
            df = pd.read_json(path, **kwargs)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

    else:
        raise ValueError(f"Unsupported format: {ext}")

    return _normalize_columns(df)


def universal_loader(
    path: str | Path,
    *,
    sheet_name: int | str | None = 0,
    health_clean: bool = True,
    combine_directories: bool = True,
    **kwargs: Any,
) -> pd.DataFrame | Dict[str, pd.DataFrame]:
    """
    Load research data from a file or directory.

    - **File**: Uses the appropriate loader by extension (.csv, .mat, .xlsx, etc.).
    - **Directory**: Loads every valid file (supported extensions) inside.
      Returns a single combined DataFrame (with _source_file column) or a dict
      of {filename: DataFrame} if combine_directories=False.

    Parameters
    ----------
    path : str or Path
        Path to a file or directory.
    sheet_name : int or str or None
        Sheet for Excel files (default 0).
    health_clean : bool
        If True, run health_check_and_clean on loaded data (default True).
    combine_directories : bool
        If True and path is a directory, return one combined DataFrame with
        _source_file column. If False, return dict of {filename: df}.
    **kwargs
        Passed to load_any_research_file.

    Returns
    -------
    pandas.DataFrame or dict of str -> pandas.DataFrame
        Loaded data. For a single file, returns DataFrame. For a directory,
        returns combined DataFrame or dict per combine_directories.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")

    def _load_one(p: Path) -> pd.DataFrame:
        df = load_any_research_file(p, sheet_name=sheet_name, **kwargs)
        if health_clean:
            df, _ = health_check_and_clean(df)
        return df

    if path.is_file():
        return _load_one(path)

    # Directory: collect all valid files
    valid_files = []
    for ext in SUPPORTED_RESEARCH_EXTENSIONS:
        valid_files.extend(path.glob(f"*{ext}"))
    valid_files = sorted(set(valid_files))

    if not valid_files:
        raise FileNotFoundError(
            f"No supported research files found in {path}. "
            f"Supported extensions: {sorted(SUPPORTED_RESEARCH_EXTENSIONS)}"
        )

    loaded: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    for f in valid_files:
        try:
            df = _load_one(f)
            if not df.empty:
                loaded[f.name] = df
        except Exception as e:
            errors[f.name] = str(e)
            logger.warning(f"Failed to load {f}: {e}")

    if not loaded:
        err_msg = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise ValueError(f"Could not load any files from {path}. Errors: {err_msg}")

    if combine_directories and len(loaded) > 0:
        dfs = []
        for name, df in loaded.items():
            df = df.copy()
            df["_source_file"] = name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    return loaded
