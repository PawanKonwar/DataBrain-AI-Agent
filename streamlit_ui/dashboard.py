"""
DataBrain AI Agent - Streamlit Dashboard

Connects to the FastAPI backend for dataset upload and natural language queries.
Supports all research file formats via the universal_loader.

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""
import base64
import io
import json
import sys
from pathlib import Path

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import requests
import pandas as pd
from databrain_agent.backend.data_cleaner import SUPPORTED_RESEARCH_EXTENSIONS

# Config
API_BASE = "http://localhost:8000"
SUPPORTED_TYPES = [ext.lstrip(".") for ext in sorted(SUPPORTED_RESEARCH_EXTENSIONS)]
ACCEPT_TYPES = SUPPORTED_TYPES  # e.g. ["csv", "xlsx", "xls", "json", "mat", "txt"]

st.set_page_config(
    page_title="DataBrain AI Agent",
    page_icon="🧠",
    layout="wide",
)

# Initialize persistent session state
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠 DataBrain AI Agent")
st.caption("Analyze your data with natural language. Connect to your FastAPI backend.")

# Sidebar first (so api_url is available for upload)
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input(
        "API Base URL",
        value=API_BASE,
        help="FastAPI backend URL (must be running)",
        key="api_url_input",
    )
    st.session_state.api_url = api_url

    st.divider()
    st.subheader("Dataset Selector")

    # Dataset selector
    try:
        r = requests.get(f"{api_url}/api/datasets", timeout=5)
        datasets = r.json().get("datasets", []) if r.ok else []
    except Exception:
        datasets = []

    if datasets:
        options = [d["name"] for d in datasets]
        # Default to current_dataset if it exists in options
        try:
            default_idx = options.index(st.session_state.current_dataset) if st.session_state.current_dataset in options else 0
        except (ValueError, TypeError):
            default_idx = 0
        selected = st.selectbox(
            "Active dataset",
            options,
            index=default_idx,
            key="dataset_select",
        )
        # Persist selection to session state
        st.session_state.current_dataset = selected
    else:
        selected = None
        st.session_state.current_dataset = None
        st.info("Upload a file to get started.")

    st.divider()

    # Clear Memory button
    if st.button("Clear Memory", use_container_width=True, type="secondary"):
        st.session_state.current_dataset = None
        st.session_state.chat_history = []
        st.success("Memory cleared.")
        st.rerun()

# Main area: File uploader (right below header)
st.subheader("Upload your dataset (CSV, Excel, JSON, etc.)")
uploaded_file = st.file_uploader(
    "Choose a file to analyze",
    type=ACCEPT_TYPES,
    help="Accepts: CSV, Excel (.xlsx, .xls), JSON, MATLAB (.mat), and text files",
)
dataset_name_input = st.text_input(
    "Dataset name (optional)",
    placeholder="Leave blank to use filename",
    key="dataset_name_input",
)
upload_clicked = st.button("Upload", key="upload_btn")
if upload_clicked and uploaded_file:
    with st.spinner("Uploading..."):
        try:
            api_url = st.session_state.get("api_url", API_BASE)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {}
            if dataset_name_input.strip():
                data["dataset_name"] = dataset_name_input.strip()
            r = requests.post(
                f"{api_url}/api/upload-dataset",
                files=files,
                data=data,
                timeout=60,
            )
            r.raise_for_status()
            resp = r.json()
            info = resp.get("info", {})
            ds_name = resp.get("dataset_name", "")
            st.success(
                f"✅ **{ds_name}** loaded — "
                f"{info.get('row_count', 0):,} rows × {info.get('column_count', 0)} columns"
            )
            # Show preview so user can verify column names
            try:
                prev = requests.get(
                    f"{api_url}/api/datasets/{ds_name}/preview",
                    params={"rows": 5},
                    timeout=5,
                )
                if prev.ok:
                    data = prev.json()
                    rows = data.get("rows", [])
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        st.caption("Preview (first 5 rows) — verify column names above")
            except Exception:
                pass
            st.session_state.current_dataset = ds_name
            st.rerun()
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Is it running? Check API URL in sidebar.")
        except requests.exceptions.HTTPError as e:
            err = e.response.json().get("detail", str(e))
            st.error(f"Upload failed: {err}")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# Dataset preview (when a dataset is active)
if st.session_state.current_dataset:
    try:
        api_url = st.session_state.get("api_url", API_BASE)
        prev = requests.get(
            f"{api_url}/api/datasets/{st.session_state.current_dataset}/preview",
            params={"rows": 5},
            timeout=5,
        )
        if prev.ok:
            data = prev.json()
            rows = data.get("rows", [])
            if rows:
                with st.expander("📋 Dataset preview (first 5 rows)", expanded=False):
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption("Verify column names before asking questions.")
    except Exception:
        pass

# Chat area (chat_history persists across uploads and dataset switches)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            img = msg["image"]
            st.image(io.BytesIO(img) if isinstance(img, bytes) else img, use_container_width=True)

# Chat input (use persisted current_dataset)
if st.session_state.current_dataset:
    prompt = st.chat_input("Ask a question about your data...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(
                        f"{st.session_state.get('api_url', API_BASE)}/api/query",
                        params={
                            "dataset_name": st.session_state.current_dataset,
                            "query": prompt,
                        },
                        timeout=120,
                    )
                    r.raise_for_status()
                    data = r.json()
                    text = data.get("message", data.get("answer", "No response."))

                    if data.get("tool_calls"):
                        text += "\n\n**Tools used:** " + ", ".join(
                            t.get("tool", "?") for t in data["tool_calls"]
                        )

                    st.markdown(text)

                    # Show chart if present
                    img_bytes = None
                    for tool in data.get("tool_calls", []):
                        if tool.get("tool") in ("chart_generator", "ChartGeneratorTool"):
                            try:
                                out = tool.get("output", "")
                                if isinstance(out, str):
                                    parsed = json.loads(out)
                                else:
                                    parsed = out
                                if parsed.get("image_base64"):
                                    img_bytes = base64.b64decode(parsed["image_base64"])
                                    st.image(io.BytesIO(img_bytes), use_container_width=True)
                                    break
                            except Exception:
                                pass

                    msg = {"role": "assistant", "content": text}
                    if img_bytes:
                        msg["image"] = img_bytes
                    st.session_state.chat_history.append(msg)

                except requests.exceptions.ConnectionError:
                    err = "Could not connect to backend. Ensure it's running."
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": "❌ " + err}
                    )
                except requests.exceptions.HTTPError as e:
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", str(e))
                    except Exception:
                        detail = str(e)
                    st.error(detail)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": "❌ " + detail}
                    )
                except Exception as e:
                    st.error(str(e))
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": "❌ " + str(e)}
                    )

        st.rerun()

else:
    st.info("👆 Upload a dataset above to start asking questions.")

# Footer
st.divider()
with st.expander("About supported formats"):
    st.markdown(
        """
    The system can analyze **any structured data file**. Supported formats include:

    - **.csv, .txt** – Comma- or tab-separated; smart sniffer skips junk headers (lab notes, metadata)
    - **.xlsx, .xls** – Excel workbooks
    - **.mat** – MATLAB files (via scipy)
    - **.json** – JSON records or arrays

    We have specialized loaders for research data (e.g., messy lab exports, MATLAB structs), but the same pipeline works for business data, surveys, logs, and more. Data is automatically cleaned (nulls, sensor glitches, duplicate timestamps) on upload.
    """
    )
