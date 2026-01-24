# Changelog

All notable changes to DataBrain AI Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
