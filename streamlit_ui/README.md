# DataBrain Streamlit UI

Streamlit dashboard that connects to the FastAPI backend.

## Run

1. **Start the FastAPI backend** (in a separate terminal):
   ```bash
   cd databrain_agent/backend && python main.py
   ```

2. **Start the Streamlit dashboard**:
   ```bash
   streamlit run streamlit_ui/dashboard.py
   ```

3. Open http://localhost:8501 in your browser.

## Features

- File upload to `/api/upload-dataset` (supports .csv, .xlsx, .xls, .json, .mat, .txt)
- Chat interface that sends queries to `/api/query`
- Supported formats driven by `SUPPORTED_RESEARCH_EXTENSIONS` from `data_cleaner.py`
- Dataset selector and API URL configuration in the sidebar
