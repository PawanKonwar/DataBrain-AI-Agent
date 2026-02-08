"""
Research Parser - Compatibility shim.

All research parsing logic has been consolidated into data_cleaner.py.
This module re-exports the primary interface for backwards compatibility.

Use data_cleaner directly for new code:
  from databrain_agent.backend.data_cleaner import load_any_research_file, universal_loader

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""

from databrain_agent.backend.data_cleaner import (
    load_any_research_file as load_research_file,
    SUPPORTED_RESEARCH_EXTENSIONS,
)

__all__ = ["load_research_file", "SUPPORTED_RESEARCH_EXTENSIONS"]
