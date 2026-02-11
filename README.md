# DataBrain AI Agent

<div align="center">

**An intelligent data analysis agent built for researchers—handles messy lab exports, batch processing, and publication-quality visualizations.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.20-orange.svg)](https://www.langchain.com/)

</div>

---

## 🔬 Built for Research

DataBrain AI Agent is designed for scientists and engineers who work with real-world data: lab equipment exports with junk headers, mixed file formats, sensor glitches, and hundreds of samples. Ask questions in plain English and get answers, trends, and publication-ready plots.

## 📋 Table of Contents

- [Quick Start for Researchers](#-quick-start-for-researchers)
- [Research Capabilities](#-research-capabilities)
- [Recent Changes](#-recent-changes)
- [Installation](#️-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Configuration](#️-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Quick Start for Researchers

**Scenario:** You have a folder of tensile test results—some `.csv`, some `.mat`, some `.xlsx`—with lab notes at the top of each file. You want to compare peak load across all samples and create an overlay plot.

### Step 1: Start the server

```bash
git clone https://github.com/yourusername/DataBrain-AI-Agent.git
cd DataBrain-AI-Agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # Add your OPENAI_API_KEY or DEEPSEEK_API_KEY

# Start backend
cd databrain_agent/backend && python main.py
```

### Step 2: Open the UI

- Open `frontend/index.html` in your browser, or run `python -m http.server 8080` in the `frontend/` folder and go to `http://localhost:8080`

### Step 3: Upload and analyze

1. **Upload a file or folder**  
   Drag and drop a `.csv`, `.xlsx`, `.mat`, `.txt`, or `.json` file. The agent auto-detects format and cleans messy headers.

2. **Ask research questions**
   - *"Which of my 100 samples had the highest displacement?"*
   - *"Plot the peak load trend across all files"*
   - *"Create an overlay of Load vs Displacement for every test"*
   - *"Summarize all files in /path/to/my/experiment_folder"*

3. **Get publication-ready output**  
   Overlay plots, Master DataFrames, and statistical comparisons—all generated from natural language.

---

## 🧪 Research Capabilities

### Universal Data Loading

One loader for all common research formats—no more "I can't read this" errors.

| Format | Description |
|--------|-------------|
| **.csv** | Smart loader skips junk headers and lab notes automatically |
| **.txt** | Tab-delimited or comma-separated; sniffer finds where data starts |
| **.xlsx / .xls** | Excel workbooks (first sheet by default) |
| **.mat** | MATLAB files—nested structs flattened to columns via scipy |
| **.json** | JSON records or arrays |

Works with mixed folders: raw sensor `.csv`, processed `.mat`, and summary `.xlsx` in the same directory.

### Autonomous Cleaning

Data is cleaned automatically after loading:

- **Junk header detection** – Finds where the table starts (lab notes, metadata, etc.)
- **Sensor glitches** – Forward-fills null values in numeric columns
- **Non-numeric strings** – Coerces values like `"N/A"`, `"---"`, `"error"` to `NaN`
- **Duplicate timestamps** – Removes duplicate rows by time column

No manual cleaning before upload.

- **Column name stripping** – Removes leading/trailing whitespace from column names on load (fixes validation errors with columns like `amount`).

### Batch Processing

Analyze hundreds of files in one go:

- **Batch summarizer** – Iterates through every supported file in a folder
- **Master DataFrame** – One row per file: columns, row count, mean/max/min per numeric column
- **Trend questions** – *"Which sample had highest displacement?"* *"Plot peak load across all files"*
- **Downsampling** – Files with >10,000 rows are downsampled for fast plotting

### Publication-Quality Visuals

Automated overlay plots for comparative research:

- **Load vs Displacement** – Stress–strain or force–extension overlays
- **Time vs Value** – Time-series overlays from multiple files
- **Fuzzy column detection** – Finds Load, Force, Displacement, Time, Strain, etc. by name
- **Units in axes** – Extracts units from headers (e.g. `Load (kN)`)
- **Legend by filename** – Each file is a distinct colored line
- **Output formats** – Interactive HTML (Plotly) or high-res PNG

---

## 📝 Recent Changes

### Tool Validation & Reliability

- **Simple Pydantic schemas** – All analysis tools use minimal `BaseModel` input schemas (`StatsInput`, `DataManipulationInput`, `ChartInput`) with `args_schema: Type[BaseModel]`—no `ConfigDict` or aliases, fixing `pydantic.v1.errors.ConfigError`.
- **Use "column" only** – Agent is instructed to always use the key `"column"`; never abbreviate to `"col"`. Column names with spaces must match the dataset preview exactly.
- **Column input cleaning** – Tools sanitize column inputs with `column.replace("'", "").replace('"', "").strip()` to handle quoted or malformed names.
- **Column name stripping on load** – On dataset upload (`main.py`), `df.columns = [c.strip() for c in df.columns]` removes hidden spaces.
- **Handle parsing errors** – Agent executor uses `handle_parsing_errors=True` so the agent can self-correct on malformed tool inputs.
- **Universal data assistant** – System prompt: *"You are a universal data assistant. When using tools, you MUST use the key 'column'. Never abbreviate to 'col'. If the user's data has spaces in column names, use the exact name as shown in the dataset preview."*

### Files Modified

| File | Changes |
|------|---------|
| `main.py` | Column stripping on upload; column cleanup in query flow |
| `agent/orchestrator.py` | `handle_parsing_errors=True`; `QuotedKeyAgentExecutor`; key normalization |
| `agent/prompts.py` | Universal data assistant prompt; use "column" only, exact names from preview |
| `tools/stats_tool.py` | `StatsInput` schema; `_run(operation, column)`; inline column cleaning |
| `tools/data_tool.py` | `DataManipulationInput` schema; explicit params; inline column cleaning |
| `tools/chart_tool.py` | `ChartInput` schema; `column`, `x_column`, `y_column`, `group_by`; inline cleaning |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- At least one API key: [OpenAI](https://platform.openai.com/api-keys) or [DeepSeek](https://platform.deepseek.com/api_keys)

### Install

```bash
git clone https://github.com/yourusername/DataBrain-AI-Agent.git
cd DataBrain-AI-Agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=... or DEEPSEEK_API_KEY=...
```

---

## 📖 Usage

### Basic Workflow

1. **Upload** – CSV, Excel, JSON, MATLAB (.mat), or TXT
2. **Select** – Choose the dataset in the sidebar
3. **Ask** – Use natural language; the agent routes to the right tools
4. **View** – Answers, charts, and summary tables in the chat

### Example Queries

**Research & batch:**
- *"Read all files in /path/to/experiment_folder"*
- *"Which sample had the highest peak load?"*
- *"Create an overlay plot of Load vs Displacement for every file"*
- *"Summarize all files and plot the displacement_max trend"*

**General analysis:**
- *"What columns are in this dataset?"*
- *"What is the average amount?"* (or any column name)
- *"Calculate the correlation between load and displacement"*
- *"Filter rows where strain > 0.01"*
- *"Create a bar chart of mean load by sample"*

### LLM Providers

- **Auto** – Chooses the best available provider
- **OpenAI** – GPT-4 / GPT-3.5
- **DeepSeek** – OpenAI-compatible API

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload-dataset` | POST | Upload CSV, Excel, JSON, MATLAB, TXT |
| `/api/datasets` | GET | List loaded datasets |
| `/api/datasets/{name}/preview` | GET | Preview first N rows of a dataset |
| `/api/query` | POST | Query the agent (`dataset_name`, `query`, `llm_provider`) |
| `/api/llm-providers` | GET | Available LLM providers |
| `/api/cost-tracking` | GET | API usage and costs |
| `/api/datasets/{name}` | DELETE | Remove a dataset |
| `/api/test-chart` | POST | Test chart generator independently |
| `/api/test-rag-memory` | GET | Test RAG memory (schemas, context search) |

---

## 📁 Project Structure

```
DataBrain-AI-Agent/
├── databrain_agent/backend/
│   ├── main.py              # FastAPI server
│   ├── data_cleaner.py      # Universal loader, health clean, research parsing
│   ├── research_parser.py   # Compatibility shim
│   ├── agent/
│   │   ├── orchestrator.py  # Agent & tool orchestration
│   │   ├── memory.py        # RAG memory (ChromaDB)
│   │   └── prompts.py
│   ├── tools/
│   │   ├── sql_tool.py      # SQL on DataFrames
│   │   ├── chart_tool.py    # Bar, line, scatter, etc.
│   │   ├── stats_tool.py    # Statistics
│   │   ├── data_tool.py     # Filter, sort, group
│   │   ├── read_file_tool.py         # Load file or directory
│   │   ├── batch_research_summarizer.py  # Master DataFrame from folder
│   │   └── research_plotter.py       # Overlay plots
│   └── llm/
├── frontend/                # Web UI (HTML/JS)
├── streamlit_ui/            # Streamlit dashboard UI
├── tests/
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here   # Optional
```

### Server

Runs at `http://localhost:8000` by default. Use `run_server.sh` or:

```bash
cd databrain_agent/backend && python main.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">

**Built for researchers who need to analyze real data, fast.**

[Report Bug](https://github.com/yourusername/DataBrain-AI-Agent/issues) · [Request Feature](https://github.com/yourusername/DataBrain-AI-Agent/issues)

</div>
