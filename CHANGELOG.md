# Changelog

All notable changes to DataBrain AI Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Research data pipeline**: data_cleaner.py with universal_loader, load_any_research_file, health_check_and_clean
- **read_file tool**: Load research data from file or directory (.csv, .xlsx, .mat, .txt, .json)
- **batch_research_summarizer tool**: Summarize all files in a folder into Master DataFrame (mean/max/min per column)
- **research_plotter tool**: Overlay plots (Load vs Displacement, Time vs Value) with fuzzy column detection, downsampling
- Dependencies: scipy (MATLAB .mat), plotly, kaleido (PNG export)
- Comprehensive GitHub preparation
- MIT License
- Enhanced README with detailed documentation
- CONTRIBUTING.md guide
- DEPLOYMENT.md guide
- Organized test files into tests/ directory
- Moved documentation files to docs/ directory

### Fixed
- Chart generation tool parameter routing
- DataManipulationTool error handling for missing operation parameter
- Tool wrapper parameter mapping for chart_generator

### Changed
- **Consolidated research parsing**: All logic moved to data_cleaner.py; research_parser.py is now a compatibility shim
- main.py upload uses universal_loader (load + health clean in one call)
- Improved .gitignore with comprehensive patterns
- Enhanced tool descriptions for better LLM routing
- Better error messages for misrouted chart requests

## [1.0.0] - 2024-01-23

### Added
- Initial release
- LangChain agent architecture
- RAG memory with ChromaDB
- Multi-LLM support (OpenAI, DeepSeek)
- SQL Executor tool
- Chart Generator tool
- Statistics Calculator tool
- Data Manipulation tool
- Cost tracking
- Web UI
- Schema-agnostic data handling
