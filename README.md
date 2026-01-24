# DataBrain AI Agent

<div align="center">

**An intelligent data analysis agent built with LangChain, featuring RAG memory, multi-LLM support, and powerful data analysis tools.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.20-orange.svg)](https://www.langchain.com/)

</div>

## 🚀 Features

- 🧠 **LangChain Agent Architecture**: Full agent implementation with intelligent tool orchestration
- 📊 **RAG Memory**: Stores dataset schemas and past analyses for context-aware responses using ChromaDB
- 🔧 **Powerful Analysis Tools**: 
  - **SQL Executor**: Run SQL queries on DataFrames using DuckDB
  - **Chart Generator**: Create visualizations (bar, line, scatter, histogram, box, heatmap)
  - **Statistics Calculator**: Compute mean, median, std, correlation, and more
  - **Data Manipulation**: Filter, sort, group, select columns, and more
- 🤖 **Multi-LLM Support**: OpenAI GPT-4/3.5 and DeepSeek with automatic routing
- 💬 **Conversation Memory**: Remembers previous questions about the same dataset
- 💰 **Cost Tracking**: Monitor API usage and costs in real-time
- 🌐 **Modern Web UI**: Clean, responsive interface for data analysis
- 📈 **Schema-Agnostic**: Works with any CSV structure without manual configuration

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Features in Detail](#features-in-detail)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Known Issues](#known-issues)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- At least one API key:
  - OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
  - DeepSeek API key ([Get one here](https://platform.deepseek.com/api_keys)) - Optional

### Step-by-Step Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/DataBrain-AI-Agent.git
cd DataBrain-AI-Agent
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your-openai-key-here
# DEEPSEEK_API_KEY=your-deepseek-key-here  # Optional
```

## 🚀 Quick Start

1. **Start the backend server:**
```bash
cd databrain_agent/backend
python main.py
```

The API will be available at `http://localhost:8000`

2. **Open the frontend:**
   - Option 1: Open `frontend/index.html` directly in your browser
   - Option 2: Serve with a simple HTTP server:
   ```bash
   cd frontend
   python -m http.server 8080
   # Then open http://localhost:8080
   ```

3. **Upload a dataset and start asking questions!**

## 📖 Usage

### Basic Workflow

1. **Upload a Dataset**: Click "Upload Dataset" and select a CSV, Excel, or JSON file
2. **Select Dataset**: Click on a dataset in the sidebar to activate it
3. **Ask Questions**: Type natural language questions about your data
4. **View Results**: See answers, charts, and analysis results in the chat interface

### Example Queries

- **Schema Questions:**
  - "What columns are in this dataset?"
  - "How many rows and columns does this dataset have?"
  - "Show me the first 5 rows"

- **Analysis Questions:**
  - "What's the average price?"
  - "Calculate the correlation between price and quantity"
  - "Show me summary statistics"

- **Visualization Requests:**
  - "Create a bar chart of sales by region"
  - "Plot a line chart showing trends over time"
  - "Generate a histogram of the age column"

- **Data Manipulation:**
  - "Filter rows where age > 30"
  - "Sort by price in descending order"
  - "Group by category and show the sum"

### LLM Provider Selection

- **Auto (Recommended)**: Automatically selects the best available LLM
- **OpenAI**: Use OpenAI GPT models (requires OPENAI_API_KEY)
- **DeepSeek**: Use DeepSeek models (requires DEEPSEEK_API_KEY)

## 🔌 API Endpoints

### Upload Dataset
```http
POST /api/upload-dataset
Content-Type: multipart/form-data

file: <file>
dataset_name: <optional-name>
```

### List Datasets
```http
GET /api/datasets
```

### Query Agent
```http
POST /api/query?dataset_name=<name>&query=<question>&llm_provider=<provider>
```

### Get LLM Providers
```http
GET /api/llm-providers
```

### Cost Tracking
```http
GET /api/cost-tracking
```

### Delete Dataset
```http
DELETE /api/datasets/{dataset_name}
```

## 🎯 Features in Detail

### RAG Memory

The agent uses ChromaDB to store:
- Dataset schemas (columns, data types, sample data)
- Past analysis results
- Query patterns and context

This enables:
- Context-aware responses based on previous interactions
- Automatic schema discovery
- Improved query understanding

### Multi-LLM Support

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **DeepSeek**: OpenAI-compatible API
- **Auto Mode**: Automatically selects the best available provider
- **Cost Tracking**: Monitor usage for both providers

### Analysis Tools

#### SQL Executor
Execute SQL queries directly on pandas DataFrames:
```sql
SELECT category, AVG(price) FROM df GROUP BY category
```

#### Chart Generator
Generate various visualizations:
- Bar charts
- Line charts
- Scatter plots
- Histograms
- Box plots
- Heatmaps (correlation matrices)

#### Statistics Calculator
Calculate statistical measures:
- Descriptive statistics (mean, median, std, min, max)
- Correlation matrices
- Column-specific statistics

#### Data Manipulator
Perform data operations:
- Filter rows by conditions
- Sort data
- Group by aggregations
- Select specific columns
- Get unique values

## 📁 Project Structure

```
DataBrain-AI-Agent/
├── databrain_agent/
│   ├── backend/
│   │   ├── main.py              # FastAPI server
│   │   ├── schemas.py           # Pydantic models
│   │   ├── compat.py            # LangChain compatibility shims
│   │   ├── agent/
│   │   │   ├── orchestrator.py  # LangChain agent setup
│   │   │   ├── memory.py        # RAG memory implementation
│   │   │   └── prompts.py       # System prompts
│   │   ├── tools/
│   │   │   ├── sql_tool.py      # SQL executor
│   │   │   ├── chart_tool.py    # Chart generator
│   │   │   ├── stats_tool.py    # Statistics calculator
│   │   │   └── data_tool.py     # Data manipulation
│   │   ├── llm/
│   │   │   ├── multi_llm.py     # LLM router
│   │   │   └── cost_tracker.py # Cost tracking
│   │   └── data/
│   │       ├── loader.py        # Data loading utilities
│   │       └── vector_store.py # ChromaDB vector store
│   └── requirements.txt
├── frontend/
│   ├── index.html               # Main UI
│   ├── style.css                # Styles
│   └── app.js                   # Frontend logic
├── tests/                       # Test files
├── docs/                        # Documentation
├── .env.example                 # Environment template
├── requirements.txt             # Main dependencies
├── run_server.sh               # Server startup script
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required: At least one API key
OPENAI_API_KEY=your_openai_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### Server Configuration

The server runs on `http://0.0.0.0:8000` by default. To change the port, modify `main.py`:

```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 🐛 Known Issues

1. **LangChain Version Compatibility**: The project uses LangChain 0.1.20. Some features may not work with older versions.

2. **Large Datasets**: Very large datasets (>100MB) may cause performance issues. Consider sampling data before upload.

3. **Chart Generation**: Some chart types may fail with certain data types. The system will provide error messages.

4. **Memory Usage**: ChromaDB stores embeddings in memory. Large numbers of datasets may increase memory usage.

5. **API Rate Limits**: Be aware of API rate limits for OpenAI and DeepSeek when processing many queries.

## 🚢 Deployment

### Local Development

```bash
# Using the startup script
./run_server.sh

# Or manually
cd databrain_agent/backend
python main.py
```

### Production Deployment

1. **Using Gunicorn:**
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker databrain_agent.backend.main:app
```

2. **Using Docker** (if Dockerfile exists):
```bash
docker build -t databrain-agent .
docker run -p 8000:8000 --env-file .env databrain-agent
```

3. **Environment Variables**: Ensure all API keys are set in production environment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest tests/

# Format code
black databrain_agent/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the agent framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [DuckDB](https://duckdb.org/) for SQL execution on DataFrames

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ for intelligent data analysis**

[Report Bug](https://github.com/yourusername/DataBrain-AI-Agent/issues) · [Request Feature](https://github.com/yourusername/DataBrain-AI-Agent/issues) · [Documentation](docs/)

</div>
