"""
Schema-agnostic LangChain agent orchestrator.

This module provides the main agent orchestration logic, including tool creation,
LLM routing, and query processing.

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""
from typing import List, Dict, Any, Optional, Union
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.tools import Tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field, field_validator
import logging
import pandas as pd
import json

# Try to import create_pandas_dataframe_agent for better DataFrame support
try:
    from langchain.agents import create_pandas_dataframe_agent
    HAS_PANDAS_AGENT = True
except ImportError:
    HAS_PANDAS_AGENT = False
    create_pandas_dataframe_agent = None

# LangChain 0.0.340 doesn't have Runnable interface - using direct LLM objects

from databrain_agent.backend.llm.multi_llm import MultiLLM
from databrain_agent.backend.agent.memory import RAGMemory
from databrain_agent.backend.agent.prompts import get_system_prompt
from databrain_agent.backend.tools import (
    SQLExecutorTool,
    ChartGeneratorTool,
    StatsCalculatorTool,
    DataManipulationTool
)

logger = logging.getLogger(__name__)


class QuotedKeyCleaner(BaseModel):
    """Pydantic model that automatically strips quotes from string values."""
    
    @field_validator('*', mode='before')
    @classmethod
    def clean_quoted_strings(cls, v):
        """Strip quotes from string values before validation."""
        if isinstance(v, str):
            cleaned = v.strip()
            # Remove outer quotes
            if (cleaned.startswith('"') and cleaned.endswith('"')) or \
               (cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1].strip()
            return cleaned
        elif isinstance(v, dict):
            # Recursively clean dict values
            return {k: cls.clean_quoted_strings(val) for k, val in v.items()}
        elif isinstance(v, list):
            # Clean list items
            return [cls.clean_quoted_strings(item) for item in v]
        return v


class LenientTool(Tool):
    """Tool with lenient validation that handles quoted keys."""
    
    def _run(self, tool_input: Union[str, Dict]) -> str:
        """Run tool with cleaned inputs."""
        # Clean the input before processing
        if isinstance(tool_input, dict):
            # Clean all string values in the dict
            cleaned_input = {}
            for key, value in tool_input.items():
                # Clean key
                cleaned_key = str(key).strip('"').strip("'").strip()
                # Clean value
                if isinstance(value, str):
                    cleaned_value = value.strip('"').strip("'").strip()
                else:
                    cleaned_value = value
                cleaned_input[cleaned_key] = cleaned_value
            tool_input = cleaned_input
        
        # Call the original function
        return self.func(tool_input)
    
    async def _arun(self, tool_input: Union[str, Dict]) -> str:
        """Async run with cleaned inputs."""
        return self._run(tool_input)


class QuotedKeyAgentExecutor(AgentExecutor):
    """Custom AgentExecutor that cleans quoted keys from tool calls before validation."""
    
    def _get_next_action(self, full_inputs, intermediate_steps):
        """Override to clean tool calls after parsing but before validation."""
        try:
            # Get the next action from the agent
            action = super()._get_next_action(full_inputs, intermediate_steps)
            
            # Clean quoted keys from tool input
            if hasattr(action, 'tool_input') and isinstance(action.tool_input, dict):
                cleaned_input = {}
                for key, value in action.tool_input.items():
                    # Clean key (remove quotes)
                    cleaned_key = str(key).strip('"').strip("'").strip()
                    # Clean value
                    if isinstance(value, str):
                        cleaned_value = value.strip('"').strip("'").strip()
                    else:
                        cleaned_value = value
                    cleaned_input[cleaned_key] = cleaned_value
                
                # Replace tool_input with cleaned version
                action.tool_input = cleaned_input
                logger.info(f"[AGENT_EXECUTOR] Cleaned tool input: {action.tool_input}")
            
            return action
        except Exception as e:
            error_str = str(e)
            # Check if it's a quoted key validation error
            if "missing some input keys" in error_str.lower() and '"' in error_str:
                logger.warning(f"[AGENT_EXECUTOR] Caught quoted key error: {error_str}")
                # Try to extract and clean the quoted key
                import re
                quoted_keys = re.findall(r'["\']([^"\']+)["\']', error_str)
                if quoted_keys:
                    logger.info(f"[AGENT_EXECUTOR] Detected quoted keys: {quoted_keys}")
            raise


class AgentOrchestrator:
    """Schema-agnostic orchestrator that adapts to any DataFrame structure."""
    
    def __init__(self, llm_router: MultiLLM, memory: RAGMemory, df: pd.DataFrame, dataset_name: str):
        """Initialize agent orchestrator with schema-agnostic design."""
        self.llm_router = llm_router
        self.memory = memory
        self.df = df.copy()  # Work with a copy to avoid mutations
        self.dataset_name = dataset_name
        
        # Clean DataFrame columns once at initialization
        self._clean_dataframe_columns()
        
        # Initialize tools with cleaned DataFrame
        self.tools = self._create_schema_agnostic_tools()
        
        # Create agent with error handling
        try:
            self.agent = self._create_agent()
            if self.agent is None:
                logger.warning(f"Agent creation returned None for dataset: {dataset_name}")
            else:
                logger.info(f"Schema-agnostic agent initialized for dataset: {dataset_name} with {len(self.df.columns)} columns")
        except ValueError as e:
            logger.error(f"Failed to initialize agent for {dataset_name}: {e}")
            self.agent = None
            raise
        except Exception as e:
            logger.error(f"Failed to initialize agent for {dataset_name}: {e}", exc_info=True)
            self.agent = None
    
    def _clean_dataframe_columns(self):
        """Clean DataFrame columns: strip whitespace, remove quotes, handle 'Unnamed' columns."""
        # Get current column names
        original_columns = list(self.df.columns)
        
        # Clean column names
        cleaned_columns = []
        for col in original_columns:
            if col is None:
                continue
            # Convert to string and clean
            col_str = str(col).strip()
            # Remove quotes if present
            if col_str.startswith('"') and col_str.endswith('"'):
                col_str = col_str[1:-1].strip()
            if col_str.startswith("'") and col_str.endswith("'"):
                col_str = col_str[1:-1].strip()
            # Skip 'Unnamed' columns
            if col_str.lower().startswith('unnamed'):
                continue
            if col_str:  # Only add non-empty columns
                cleaned_columns.append(col_str)
        
        # Remove duplicate columns (keep first occurrence)
        seen = set()
        unique_columns = []
        for col in cleaned_columns:
            col_lower = col.lower().strip()
            if col_lower not in seen:
                seen.add(col_lower)
                unique_columns.append(col)
        
        # Rename columns if they changed
        if len(unique_columns) != len(original_columns) or any(oc != nc for oc, nc in zip(original_columns[:len(unique_columns)], unique_columns)):
            # Create mapping for columns that exist
            column_mapping = {}
            for i, orig_col in enumerate(original_columns):
                if i < len(unique_columns):
                    column_mapping[orig_col] = unique_columns[i]
            
            # Rename columns
            self.df = self.df.rename(columns=column_mapping)
            # Select only the columns we want
            self.df = self.df[[col for col in unique_columns if col in self.df.columns]]
        
        logger.info(f"Cleaned DataFrame columns: {len(original_columns)} -> {len(self.df.columns)} columns")
    
    def clean_agent_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean agent inputs by stripping quotes from ALL dictionary keys and string values.
        This handles cases where LangChain expects clean keys but receives quoted ones.
        """
        logger.info(f"[CLEAN_AGENT_INPUTS] Cleaning agent inputs: {inputs}")
        cleaned = {}
        
        for key, value in inputs.items():
            # Clean the key itself (remove quotes)
            key_str = str(key)
            cleaned_key = key_str.strip('"').strip("'").strip()
            logger.info(f"[CLEAN_AGENT_INPUTS] Key: '{key_str}' -> '{cleaned_key}'")
            
            # Clean the value
            if isinstance(value, dict):
                # Recursively clean nested dicts
                cleaned[cleaned_key] = self.clean_agent_inputs(value)
            elif isinstance(value, str):
                # Strip quotes from string values
                cleaned_value = value.strip('"').strip("'").strip()
                logger.info(f"[CLEAN_AGENT_INPUTS] Value: '{value}' -> '{cleaned_value}'")
                cleaned[cleaned_key] = cleaned_value
            elif isinstance(value, list):
                # Clean list items
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_list.append(self.clean_agent_inputs(item))
                    elif isinstance(item, str):
                        cleaned_list.append(item.strip('"').strip("'").strip())
                    else:
                        cleaned_list.append(item)
                cleaned[cleaned_key] = cleaned_list
            else:
                cleaned[cleaned_key] = value
        
        logger.info(f"[CLEAN_AGENT_INPUTS] Cleaned inputs: {cleaned}")
        return cleaned
    
    def _resolve_column_name(self, column_input: str) -> Optional[str]:
        """
        Resolve column name with case-insensitive, whitespace-tolerant matching.
        Handles quoted names, case variations, and whitespace.
        ALWAYS strips quotes FIRST before any matching.
        """
        logger.info(f"[RESOLVE_COLUMN] Resolving column: '{column_input}' (type: {type(column_input)}, repr: {repr(column_input)})")
        
        if not column_input:
            logger.warning(f"[RESOLVE_COLUMN] Input is empty/None: {column_input}")
            return None
        
        # Convert to string and STRIP QUOTES FIRST (handles '"id"', "'id'", etc.)
        input_str = str(column_input)
        # Remove outer quotes (handles '"id"', "'id'", '"id"', etc.)
        cleaned = input_str.strip()
        # Remove quotes from both ends if present
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        logger.info(f"[RESOLVE_COLUMN] After quote stripping: '{cleaned}' (from '{column_input}')")
        
        if not cleaned:
            logger.warning(f"[RESOLVE_COLUMN] Cleaned input is empty after quote stripping")
            return None
        
        # Get actual column names
        actual_columns = list(self.df.columns)
        logger.info(f"[RESOLVE_COLUMN] Available columns: {actual_columns}")
        
        # Try exact match first
        if cleaned in actual_columns:
            logger.info(f"[RESOLVE_COLUMN] ✓ Exact match: '{cleaned}'")
            return cleaned
        
        # Try case-insensitive match
        cleaned_lower = cleaned.lower()
        logger.info(f"[RESOLVE_COLUMN] Attempting case-insensitive match: '{cleaned_lower}'")
        for col in actual_columns:
            col_str = str(col)
            if col_str.lower() == cleaned_lower:
                logger.info(f"[RESOLVE_COLUMN] ✓ Case-insensitive match: '{cleaned}' -> '{col_str}'")
                return col_str
        
        # Try case-insensitive with whitespace tolerance
        cleaned_normalized = cleaned_lower.replace(' ', '').replace('_', '')
        logger.info(f"[RESOLVE_COLUMN] Attempting normalized match: '{cleaned_normalized}'")
        for col in actual_columns:
            col_str = str(col)
            col_normalized = col_str.lower().replace(' ', '').replace('_', '')
            if col_normalized == cleaned_normalized:
                logger.info(f"[RESOLVE_COLUMN] ✓ Normalized match: '{cleaned}' -> '{col_str}'")
                return col_str
        
        # No match found
        logger.warning(f"[RESOLVE_COLUMN] ✗ No match found for '{column_input}' (cleaned: '{cleaned}')")
        logger.warning(f"[RESOLVE_COLUMN] Available columns: {actual_columns}")
        return None
    
    def _create_schema_agnostic_tools(self) -> List[Tool]:
        """Create tools that work with ANY DataFrame schema dynamically."""
        # Create tool instances with cleaned DataFrame
        sql_tool = SQLExecutorTool(df=self.df)
        chart_tool = ChartGeneratorTool(df=self.df)
        stats_tool = StatsCalculatorTool(df=self.df)
        data_tool = DataManipulationTool(df=self.df)
        
        # Get actual column names for dynamic descriptions
        actual_columns = [str(col) for col in self.df.columns]
        columns_str = ', '.join(actual_columns) if actual_columns else 'No columns available'
        
        # Create dynamic tool descriptions that include actual schema
        sql_desc = f"""Execute SQL queries on the dataset.
The dataset is available as table 'df' with these columns: {columns_str}
Use exact column names as they appear in the list above.
Returns query results as JSON."""
        
        chart_desc = f"""Generate charts and visualizations from data. USE THIS TOOL for any chart, graph, plot, or visualization requests.
Supported chart types: bar, line, scatter, histogram, box, heatmap.
Available columns: {columns_str}
Use exact column names from the list above for x_column, y_column, and group_by parameters.
Returns base64-encoded image data.
IMPORTANT: For chart requests like "show me a chart", "create a graph", "plot the data", "visualize", ALWAYS use this tool, NOT data_manipulator."""
        
        stats_desc = f"""Calculate statistical measures (mean, median, std, correlation, etc.).
Available columns: {columns_str}
Use exact column names from the list above.
Returns statistical results as JSON."""
        
        data_desc = f"""Perform data manipulation operations (filter, sort, group_by, select_columns, head, tail, unique).
Available columns: {columns_str}
Use exact column names from the list above.
Returns manipulated data as JSON.
IMPORTANT: Do NOT use this tool for chart/graph/plot/visualization requests - use chart_generator instead."""
        
        def clean_json_values(data):
            """
            Recursively clean JSON values by stripping quotes from all strings.
            Handles: {"column": "\"id\""} -> {"column": "id"}
            """
            if isinstance(data, dict):
                cleaned = {}
                for key, value in data.items():
                    # Clean key (remove quotes)
                    cleaned_key = str(key).strip('"').strip("'").strip()
                    # Recursively clean value
                    cleaned[cleaned_key] = clean_json_values(value)
                return cleaned
            elif isinstance(data, list):
                return [clean_json_values(item) for item in data]
            elif isinstance(data, str):
                # CRITICAL: Strip quotes from string values
                # Handles: '"id"' -> 'id', "\"id\"" -> 'id'
                cleaned = data.strip()
                # Remove outer quotes if present
                if (cleaned.startswith('"') and cleaned.endswith('"')) or \
                   (cleaned.startswith("'") and cleaned.endswith("'")):
                    cleaned = cleaned[1:-1].strip()
                return cleaned
            else:
                return data
        
        # Create wrapper that resolves column names dynamically
        def create_dynamic_wrapper(original_func, tool_name: str):
            def wrapper(tool_input):
                """Wrapper that resolves column names dynamically before calling tool."""
                logger.info(f"[TOOL_WRAPPER] ===== Tool '{tool_name}' called =====")
                logger.info(f"[TOOL_WRAPPER] Raw input: {tool_input} (type: {type(tool_input)}, repr: {repr(tool_input)})")
                import json as json_lib
                kwargs = {}
                
                # Parse input
                if tool_input is None:
                    logger.info(f"[TOOL_WRAPPER] Input is None, calling tool without parameters")
                    return original_func()
                
                if isinstance(tool_input, str):
                    logger.info(f"[TOOL_WRAPPER] Input is string, attempting JSON parse")
                    # Try to parse as JSON first
                    try:
                        parsed = json_lib.loads(tool_input)
                        logger.info(f"[TOOL_WRAPPER] Parsed JSON: {parsed}")
                        
                        # CRITICAL: Clean JSON values immediately after parsing
                        # This handles: {"column": "\"id\""} -> {"column": "id"}
                        if isinstance(parsed, dict):
                            kwargs = clean_json_values(parsed)
                            logger.info(f"[TOOL_WRAPPER] After JSON cleaning: {kwargs}")
                        else:
                            # Single value - clean it
                            cleaned_value = clean_json_values(parsed)
                            if tool_name == 'sql_executor':
                                kwargs = {'query': cleaned_value if isinstance(cleaned_value, str) else tool_input}
                            elif tool_name == 'chart_generator':
                                kwargs = {'chart_type': cleaned_value if isinstance(cleaned_value, str) else tool_input}
                            elif tool_name in ['stats_calculator', 'data_manipulator']:
                                kwargs = {'operation': cleaned_value if isinstance(cleaned_value, str) else tool_input}
                            else:
                                kwargs = {'operation': cleaned_value if isinstance(cleaned_value, str) else tool_input}
                            logger.info(f"[TOOL_WRAPPER] Single value after cleaning: {kwargs}")
                    except Exception as e:
                        logger.warning(f"[TOOL_WRAPPER] JSON parse failed: {e}, treating as plain string")
                        # Clean the string itself
                        cleaned_str = clean_json_values(tool_input)
                        if tool_name == 'sql_executor':
                            kwargs = {'query': cleaned_str}
                        elif tool_name == 'chart_generator':
                            kwargs = {'chart_type': cleaned_str}
                        elif tool_name in ['stats_calculator', 'data_manipulator']:
                            kwargs = {'operation': cleaned_str}
                        else:
                            kwargs = {'operation': cleaned_str}
                elif isinstance(tool_input, dict):
                    # Clean dict values immediately
                    kwargs = clean_json_values(tool_input)
                    logger.info(f"[TOOL_WRAPPER] Input is dict, cleaned: {kwargs}")
                    
                    # Special handling: If chart_generator is called but gets chart-related params,
                    # ensure chart_type is set correctly
                    if tool_name == 'chart_generator' and 'chart_type' not in kwargs:
                        # Try to infer chart_type from other parameters or set a default
                        if 'operation' in kwargs:
                            # This might be a misrouted request - try to use operation as chart_type
                            logger.warning(f"[TOOL_WRAPPER] chart_generator received 'operation' param, converting to 'chart_type'")
                            kwargs['chart_type'] = kwargs.pop('operation')
                        elif any(key in kwargs for key in ['x_column', 'y_column', 'group_by']):
                            # Chart parameters present but no chart_type - default to bar
                            logger.info(f"[TOOL_WRAPPER] Chart parameters detected but no chart_type, defaulting to 'bar'")
                            kwargs['chart_type'] = 'bar'
                    
                    # Special handling: If data_manipulator is called with chart-related params,
                    # provide helpful error message
                    if tool_name == 'data_manipulator' and ('chart_type' in kwargs or 'x_column' in kwargs or 'y_column' in kwargs):
                        logger.error(f"[TOOL_WRAPPER] Chart request misrouted to data_manipulator! Params: {kwargs}")
                        return json.dumps({
                            "error": "Chart requests should use chart_generator tool, not data_manipulator",
                            "hint": "The system will retry with the correct tool.",
                            "received_params": list(kwargs.keys())
                        })
                else:
                    logger.info(f"[TOOL_WRAPPER] Input is other type: {type(tool_input)}, converting to string")
                    cleaned_str = clean_json_values(str(tool_input))
                    if tool_name == 'sql_executor':
                        kwargs = {'query': cleaned_str}
                    elif tool_name == 'chart_generator':
                        kwargs = {'chart_type': cleaned_str}
                    elif tool_name in ['stats_calculator', 'data_manipulator']:
                        kwargs = {'operation': cleaned_str}
                    else:
                        kwargs = {'operation': cleaned_str}
                
                logger.info(f"[TOOL_WRAPPER] After JSON cleaning layer: {kwargs}")
                
                # CRITICAL: Strip quotes from BOTH keys and values BEFORE processing
                # This handles cases where LangChain passes {'"id"': value} or {key: '"id"'}
                cleaned_kwargs = {}
                for key, value in kwargs.items():
                    # Strip quotes from KEY itself (handles {'"id"': value})
                    key_str = str(key)
                    cleaned_key = key_str.strip('"').strip("'").strip()
                    logger.info(f"[TOOL_WRAPPER] Key: '{key_str}' -> '{cleaned_key}'")
                    
                    # Strip quotes from VALUE
                    if isinstance(value, str):
                        cleaned_value = value.strip('"').strip("'").strip()
                        logger.info(f"[TOOL_WRAPPER] Value: '{value}' -> '{cleaned_value}'")
                        cleaned_kwargs[cleaned_key] = cleaned_value
                    elif isinstance(value, list):
                        cleaned_list = []
                        for item in value:
                            if isinstance(item, str):
                                cleaned_item = item.strip('"').strip("'").strip()
                                cleaned_list.append(cleaned_item)
                            else:
                                cleaned_list.append(item)
                        cleaned_kwargs[cleaned_key] = cleaned_list
                        logger.info(f"[TOOL_WRAPPER] List value cleaned: {value} -> {cleaned_list}")
                    else:
                        cleaned_kwargs[cleaned_key] = value
                
                logger.info(f"[TOOL_WRAPPER] After quote stripping: {cleaned_kwargs}")
                
                # Now resolve column names
                resolved_kwargs = {}
                column_param_names = ['column', 'x_column', 'y_column', 'group_by', 'group_column', 'agg_column', 'columns']
                
                for key, value in cleaned_kwargs.items():
                    logger.info(f"[TOOL_WRAPPER] Processing parameter: key='{key}', value='{value}' (type: {type(value)})")
                    
                    # Resolve column names
                    if key.lower() in [p.lower() for p in column_param_names]:
                        logger.info(f"[TOOL_WRAPPER] This is a column parameter, resolving...")
                        if isinstance(value, str):
                            logger.info(f"[TOOL_WRAPPER] Resolving column string: '{value}'")
                            resolved = self._resolve_column_name(value)
                            if resolved:
                                resolved_kwargs[key] = resolved
                                logger.info(f"[TOOL_WRAPPER] ✓ Resolved: '{value}' -> '{resolved}'")
                            else:
                                # If resolution fails, use cleaned value
                                resolved_kwargs[key] = value
                                logger.warning(f"[TOOL_WRAPPER] ✗ Resolution failed, using cleaned value: '{value}'")
                        elif isinstance(value, list):
                            logger.info(f"[TOOL_WRAPPER] Resolving column list: {value}")
                            resolved_list = []
                            for item in value:
                                if isinstance(item, str):
                                    resolved = self._resolve_column_name(item)
                                    if resolved:
                                        resolved_list.append(resolved)
                                        logger.info(f"[TOOL_WRAPPER] ✓ Resolved list item: '{item}' -> '{resolved}'")
                                    else:
                                        resolved_list.append(item)
                                        logger.warning(f"[TOOL_WRAPPER] ✗ List item resolution failed: '{item}'")
                                else:
                                    resolved_list.append(item)
                            resolved_kwargs[key] = resolved_list
                        else:
                            resolved_kwargs[key] = value
                            logger.info(f"[TOOL_WRAPPER] Non-string column value, passing as-is: {value}")
                    else:
                        # Non-column parameter - pass as-is (already cleaned)
                        resolved_kwargs[key] = value
                        logger.info(f"[TOOL_WRAPPER] Non-column parameter: '{key}' = '{value}'")
                
                logger.info(f"[TOOL_WRAPPER] Final resolved parameters: {resolved_kwargs}")
                logger.info(f"[TOOL_WRAPPER] Calling tool '{tool_name}' with resolved parameters")
                try:
                    result = original_func(**resolved_kwargs)
                    logger.info(f"[TOOL_WRAPPER] ✓ Tool '{tool_name}' succeeded")
                    return result
                except Exception as e:
                    logger.error(f"[TOOL_WRAPPER] ✗ Tool '{tool_name}' failed with error: {e}")
                    logger.error(f"[TOOL_WRAPPER] Error type: {type(e)}")
                    logger.error(f"[TOOL_WRAPPER] Parameters passed to tool: {resolved_kwargs}")
                    import traceback
                    logger.error(f"[TOOL_WRAPPER] Traceback: {traceback.format_exc()}")
                    raise
            return wrapper
        
        # Create a wrapper function that cleans inputs, then wrap it in a Tool
        def create_cleaned_tool_func(original_func, tool_name: str):
            """Create a function that cleans inputs before calling the original."""
            
            def _clean_input(input_data):
                """Clean quoted keys and values from input."""
                if isinstance(input_data, dict):
                    cleaned_dict = {}
                    for key, value in input_data.items():
                        # Clean key (remove quotes)
                        key_str = str(key)
                        cleaned_key = key_str.strip('"').strip("'").strip()
                        logger.info(f"[CLEANED_TOOL] {tool_name} - Key: '{key_str}' -> '{cleaned_key}'")
                        
                        # Clean value (remove quotes)
                        if isinstance(value, str):
                            cleaned_value = value.strip('"').strip("'").strip()
                            logger.info(f"[CLEANED_TOOL] {tool_name} - Value: '{value}' -> '{cleaned_value}'")
                            cleaned_dict[cleaned_key] = cleaned_value
                        elif isinstance(value, dict):
                            # Recursively clean nested dicts
                            cleaned_dict[cleaned_key] = _clean_input(value)
                        elif isinstance(value, list):
                            cleaned_list = [_clean_input(item) if isinstance(item, (dict, str)) else item for item in value]
                            cleaned_dict[cleaned_key] = cleaned_list
                        else:
                            cleaned_dict[cleaned_key] = value
                    return cleaned_dict
                elif isinstance(input_data, str):
                    return input_data.strip('"').strip("'").strip()
                else:
                    return input_data
            
            def cleaned_func(tool_input):
                """Wrapper function that cleans inputs before calling original."""
                logger.info(f"[CLEANED_TOOL] {tool_name} received input: {tool_input} (type: {type(tool_input)})")
                cleaned_input = _clean_input(tool_input)
                logger.info(f"[CLEANED_TOOL] {tool_name} cleaned input: {cleaned_input}")
                return original_func(cleaned_input)
            
            return cleaned_func
        
        # Return tools with cleaned input handling using LenientTool
        # LenientTool handles quoted keys at the Pydantic validation level
        return [
            LenientTool(
                name=sql_tool.name,
                description=sql_desc,
                func=create_cleaned_tool_func(
                    create_dynamic_wrapper(sql_tool._run, sql_tool.name),
                    sql_tool.name
                )
            ),
            LenientTool(
                name=chart_tool.name,
                description=chart_desc,
                func=create_cleaned_tool_func(
                    create_dynamic_wrapper(chart_tool._run, chart_tool.name),
                    chart_tool.name
                )
            ),
            LenientTool(
                name=stats_tool.name,
                description=stats_desc,
                func=create_cleaned_tool_func(
                    create_dynamic_wrapper(stats_tool._run, stats_tool.name),
                    stats_tool.name
                )
            ),
            LenientTool(
                name=data_tool.name,
                description=data_desc,
                func=create_cleaned_tool_func(
                    create_dynamic_wrapper(data_tool._run, data_tool.name),
                    data_tool.name
                )
            )
        ]
    
    def _is_openai_compatible(self, llm) -> bool:
        """Check if LLM is OpenAI-compatible."""
        if isinstance(llm, ChatOpenAI):
            return True
        if hasattr(llm, '__class__'):
            class_name = llm.__class__.__name__
            if 'ChatOpenAI' in class_name or 'OpenAI' in class_name:
                return True
        return False
    
    def _is_actual_openai(self, llm) -> bool:
        """Check if LLM is actual OpenAI (not DeepSeek or other OpenAI-compatible APIs)."""
        if not isinstance(llm, ChatOpenAI):
            return False
        # Check if it has a custom base_url (like DeepSeek)
        if hasattr(llm, 'openai_api_base') and llm.openai_api_base:
            base_url = llm.openai_api_base
            # If base_url is not the default OpenAI URL, it's not actual OpenAI
            if 'api.openai.com' not in base_url and 'openai.com' not in base_url:
                return False
        # Also check for base_url attribute (newer LangChain versions)
        if hasattr(llm, 'base_url') and llm.base_url:
            base_url = str(llm.base_url)
            if 'api.openai.com' not in base_url and 'openai.com' not in base_url:
                return False
        return True
    
    def _ensure_runnable(self, llm):
        """For LangChain 0.0.340, just return the LLM directly (no Runnable interface)."""
        logger.info(f"Using LLM directly (LangChain 0.0.340) - type: {type(llm)}, class: {llm.__class__.__name__}")
        return llm
    
    def _create_compatible_agent(self, llm, system_prompt):
        """Create agent with compatibility handling for multiple LLMs (LangChain 0.0.340)."""
        # For LangChain 0.0.340, use LLM directly (no Runnable interface)
        logger.info(f"Creating agent with LLM - type: {type(llm)}, class: {llm.__class__.__name__}")
        logger.info(f"LLM model: {getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))}")
        if hasattr(llm, 'openai_api_base'):
            logger.info(f"LLM base_url: {llm.openai_api_base}")
        if hasattr(llm, 'base_url'):
            logger.info(f"LLM base_url (attr): {llm.base_url}")
        
        # Check if it's actual OpenAI (not DeepSeek)
        is_actual_openai = self._is_actual_openai(llm)
        is_chatopenai = isinstance(llm, ChatOpenAI)
        
        logger.info(f"Creating agent - is_actual_openai: {is_actual_openai}, is_chatopenai: {is_chatopenai}")
        
        # Try create_pandas_dataframe_agent first (better for DataFrame operations)
        if HAS_PANDAS_AGENT:
            try:
                logger.info("Attempting to create pandas dataframe agent")
                agent = create_pandas_dataframe_agent(
                    llm=llm,
                    df=self.df,
                    verbose=True,
                    return_intermediate_steps=True,
                    handle_parsing_errors=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
                )
                logger.info("Successfully created pandas dataframe agent")
                return agent
            except Exception as e:
                logger.warning(f"Failed to create pandas dataframe agent: {e}, falling back to initialize_agent")
        
        # Use REACT agent by default for ALL LLMs (works with OpenAI, DeepSeek, etc.)
        # OPENAI_FUNCTIONS only works with actual OpenAI models
        logger.info(f"Creating REACT agent (ZERO_SHOT_REACT_DESCRIPTION) - compatible with all LLMs")
        
        # Try REACT agent first (more flexible, no strict parameter validation)
        try:
            agent_kwargs = {"prefix": system_prompt}
            if hasattr(self.memory, 'conversation_memory') and self.memory.conversation_memory:
                try:
                    logger.info("Attempting to create REACT agent with memory")
                    agent = initialize_agent(
                        tools=self.tools,
                        llm=llm,  # Direct LLM object (LangChain 0.0.340)
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        return_intermediate_steps=True,
                        memory=self.memory.conversation_memory,
                        agent_kwargs=agent_kwargs,
                        handle_parsing_errors=True
                    )
                    # Patch the agent's output parser to clean quoted keys
                    if isinstance(agent, AgentExecutor) and hasattr(agent.agent, 'output_parser'):
                        original_parse = agent.agent.output_parser.parse
                        def patched_parse(text: str):
                            """Parse agent output and clean quoted keys from tool calls."""
                            result = original_parse(text)
                            # Clean quoted keys from tool input if present
                            if hasattr(result, 'tool_input') and isinstance(result.tool_input, dict):
                                cleaned_input = {}
                                for key, value in result.tool_input.items():
                                    # Clean key
                                    cleaned_key = str(key).strip('"').strip("'").strip()
                                    # Clean value
                                    if isinstance(value, str):
                                        cleaned_value = value.strip('"').strip("'").strip()
                                    else:
                                        cleaned_value = value
                                    cleaned_input[cleaned_key] = cleaned_value
                                result.tool_input = cleaned_input
                                logger.info(f"[OUTPUT_PARSER] Cleaned tool input: {result.tool_input}")
                            return result
                        agent.agent.output_parser.parse = patched_parse
                        logger.info("Patched agent output parser to clean quoted keys")
                    
                    # Wrap the agent executor to handle quoted keys
                    if isinstance(agent, AgentExecutor):
                        return QuotedKeyAgentExecutor(
                            agent=agent.agent,
                            tools=agent.tools,
                            verbose=agent.verbose,
                            return_intermediate_steps=agent.return_intermediate_steps,
                            memory=agent.memory,
                            handle_parsing_errors=agent.handle_parsing_errors
                        )
                    return agent
                except Exception as mem_error:
                    logger.warning(f"Failed to create agent with memory: {mem_error}. Trying without memory.")
            
            logger.info("Attempting to create REACT agent without memory")
            agent = initialize_agent(
                tools=self.tools,
                llm=llm,  # Direct LLM object (LangChain 0.0.340)
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                return_intermediate_steps=True,
                agent_kwargs=agent_kwargs,
                handle_parsing_errors=True
            )
            # Patch the agent's output parser to clean quoted keys
            if isinstance(agent, AgentExecutor) and hasattr(agent.agent, 'output_parser'):
                original_parse = agent.agent.output_parser.parse
                def patched_parse(text: str):
                    """Parse agent output and clean quoted keys from tool calls."""
                    result = original_parse(text)
                    # Clean quoted keys from tool input if present
                    if hasattr(result, 'tool_input') and isinstance(result.tool_input, dict):
                        cleaned_input = {}
                        for key, value in result.tool_input.items():
                            # Clean key
                            cleaned_key = str(key).strip('"').strip("'").strip()
                            # Clean value
                            if isinstance(value, str):
                                cleaned_value = value.strip('"').strip("'").strip()
                            else:
                                cleaned_value = value
                            cleaned_input[cleaned_key] = cleaned_value
                        result.tool_input = cleaned_input
                        logger.info(f"[OUTPUT_PARSER] Cleaned tool input: {result.tool_input}")
                    return result
                agent.agent.output_parser.parse = patched_parse
                logger.info("Patched agent output parser to clean quoted keys")
            
            # Wrap the agent executor to handle quoted keys
            if isinstance(agent, AgentExecutor):
                return QuotedKeyAgentExecutor(
                    agent=agent.agent,
                    tools=agent.tools,
                    verbose=agent.verbose,
                    return_intermediate_steps=agent.return_intermediate_steps,
                    memory=agent.memory,
                    handle_parsing_errors=agent.handle_parsing_errors
                )
            logger.info("Successfully created REACT agent")
            return agent
        except Exception as e:
            logger.error(f"Failed to create REACT agent: {e}", exc_info=True)
            # Fallback: Try OPENAI_FUNCTIONS ONLY for actual OpenAI (not DeepSeek)
            if is_actual_openai:
                try:
                    logger.info(f"Falling back to OPENAI_FUNCTIONS agent (OpenAI only)")
                    # For OPENAI_FUNCTIONS, system_message must be a ChatPromptTemplate or SystemMessage
                    # Create a proper ChatPromptTemplate with the system prompt
                    try:
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                            MessagesPlaceholder(variable_name="agent_scratchpad"),
                        ])
                        agent_kwargs = {"system_message": prompt_template}
                    except Exception as prompt_error:
                        # If ChatPromptTemplate creation fails, use SystemMessage directly
                        logger.warning(f"Failed to create ChatPromptTemplate: {prompt_error}, using SystemMessage")
                        agent_kwargs = {"system_message": SystemMessage(content=system_prompt)}
                    
                    agent = initialize_agent(
                        tools=self.tools,
                        llm=llm,  # Direct LLM object (LangChain 0.0.340)
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        return_intermediate_steps=True,
                        memory=self.memory.conversation_memory if hasattr(self.memory, 'conversation_memory') else None,
                        agent_kwargs=agent_kwargs,
                        handle_parsing_errors=True
                    )
                    return agent
                except Exception as e2:
                    logger.error(f"OPENAI_FUNCTIONS also failed: {e2}", exc_info=True)
                    raise ValueError(f"Failed to create agent: {str(e2)}. Please check your LLM configuration.")
            else:
                # For non-OpenAI LLMs (like DeepSeek), we can only use REACT agent
                logger.error(f"REACT agent failed for non-OpenAI LLM. Cannot use OPENAI_FUNCTIONS.")
                raise ValueError(f"Failed to create REACT agent: {str(e)}. Please check your LLM configuration.")
    
    def _create_agent(self):
        """Create schema-agnostic LangChain agent."""
        try:
            llm = self.llm_router.get_llm()
        except ValueError as e:
            logger.error(f"No LLM provider available: {e}")
            raise ValueError(f"Cannot create agent: {str(e)}. Please configure API keys.")
        
        # Ensure LLM is properly initialized (LangChain 0.0.340 uses direct LLM objects)
        if llm is None:
            raise ValueError("LLM is None. Please check your LLM configuration.")
        
        logger.info(f"Retrieved LLM: type={type(llm)}, class={llm.__class__.__name__}")
        
        # For LangChain 0.0.340, use LLM directly (no Runnable conversion needed)
        llm = self._ensure_runnable(llm)
        
        # Get context from RAG memory
        relevant_context = self.memory.get_relevant_context("", self.dataset_name)
        
        # Create dataset info from ACTUAL DataFrame (not cached schema)
        dataset_info = {
            "name": self.dataset_name,
            "columns": [str(col) for col in self.df.columns],
            "column_count": len(self.df.columns),
            "row_count": len(self.df)
        }
        
        # Update memory with actual schema
        self.memory.add_dataset_schema(self.dataset_name, dataset_info)
        
        # Generate dynamic system prompt
        system_prompt = get_system_prompt(dataset_info, relevant_context)
        
        # Create agent
        try:
            executor = self._create_compatible_agent(llm, system_prompt)
        except Exception as e:
            logger.error(f"Failed to create agent: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize agent: {str(e)}. Please check your API keys and configuration.")
        
        return executor
    
    def query(self, user_query: str, llm_provider: Optional[str] = None) -> Dict[str, Any]:
        """Process user query through schema-agnostic agent."""
        # Get relevant context
        relevant_context = self.memory.get_relevant_context(user_query, self.dataset_name)
        
        # Always use ACTUAL DataFrame columns (never cached/stale)
        dataset_info = {
            "name": self.dataset_name,
            "columns": [str(col) for col in self.df.columns],
            "column_count": len(self.df.columns),
            "row_count": len(self.df)
        }
        
        logger.info(f"Query on '{self.dataset_name}': {len(self.df.columns)} columns, {len(self.df)} rows")
        
        # Generate dynamic prompt with actual schema
        system_prompt = get_system_prompt(dataset_info, relevant_context)
        
        # Ensure agent is initialized
        if not hasattr(self, 'agent') or self.agent is None:
            try:
                llm = self.llm_router.get_llm(llm_provider)
                if llm is None:
                    raise ValueError("LLM is None. Please check your LLM configuration.")
                
                # For LangChain 0.0.340, use LLM directly (no Runnable conversion needed)
                llm = self._ensure_runnable(llm)
                logger.info(f"Using LLM provider: {llm_provider}, type: {type(llm)}")
                
                self.agent = self._create_compatible_agent(llm, system_prompt)
                if self.agent is None:
                    raise ValueError("Failed to initialize agent. Please check your LLM API keys.")
            except Exception as e:
                logger.error(f"Failed to initialize agent: {e}", exc_info=True)
                raise ValueError(f"Agent initialization failed: {str(e)}. Please check your LLM API keys.")
        
        # Execute query with detailed logging and error handling
        try:
            logger.info(f"[QUERY] ===== Executing query =====")
            logger.info(f"[QUERY] Query: {user_query}")
            logger.info(f"[QUERY] Dataset: {self.dataset_name}")
            logger.info(f"[QUERY] Columns: {list(self.df.columns)}")
            
            # Log agent type and tools
            if hasattr(self.agent, 'tools'):
                logger.info(f"[QUERY] Agent has {len(self.agent.tools)} tools")
                for tool in self.agent.tools:
                    logger.info(f"[QUERY] Tool: {tool.name} - {tool.description[:100]}")
            
            # Prepare agent inputs
            raw_inputs = {"input": user_query}
            logger.info(f"[QUERY] Agent inputs before cleaning: {raw_inputs}")
            
            # CRITICAL: Clean agent inputs before invoke to remove quoted keys
            cleaned_inputs = self.clean_agent_inputs(raw_inputs)
            logger.info(f"[QUERY] Agent inputs after cleaning: {cleaned_inputs}")
            
            # Wrap agent.invoke to catch and handle validation errors
            try:
                result = self.agent.invoke(cleaned_inputs)
                logger.info(f"[QUERY] ✓ Query executed successfully")
            except ValueError as ve:
                error_str = str(ve)
                # Check if it's the quoted key validation error
                if "missing some input keys" in error_str.lower() and '"' in error_str:
                    logger.warning(f"[QUERY] Caught validation error with quoted keys: {error_str}")
                    # Try to extract the quoted key
                    import re
                    quoted_keys = re.findall(r'["\']([^"\']+)["\']', error_str)
                    if quoted_keys:
                        logger.info(f"[QUERY] Detected quoted keys in error: {quoted_keys}")
                        # The issue is that LangChain is validating tool inputs with quoted keys
                        # We can't intercept at this level, but we can provide a better error message
                        # and suggest the user try again (our CleanedTool should handle it on retry)
                        available_cols = ', '.join(self.df.columns.tolist())
                        raise ValueError(
                            f"Tool validation error detected. This is a known issue with quoted column names. "
                            f"Available columns: {available_cols}. "
                            f"Please try your query again - the system will automatically clean quoted keys."
                        )
                    raise
                else:
                    raise
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[QUERY] ✗ Query execution error: {e}")
            logger.error(f"[QUERY] Error type: {type(e)}")
            logger.error(f"[QUERY] Error repr: {repr(e)}")
            import traceback
            logger.error(f"[QUERY] Traceback: {traceback.format_exc()}")
            
            # Provide helpful error messages
            if "missing some input keys" in error_msg.lower():
                # Extract the missing key from error message
                import re
                missing_keys = re.findall(r"['\"]([^'\"]+)['\"]", error_msg)
                available_cols = ', '.join(self.df.columns.tolist())
                
                logger.error(f"[QUERY] Missing keys detected: {missing_keys}")
                logger.error(f"[QUERY] Available columns: {available_cols}")
                
                # Try to resolve the missing key
                if missing_keys:
                    resolved_key = self._resolve_column_name(missing_keys[0])
                    if resolved_key:
                        logger.info(f"[QUERY] ✓ Resolved missing key '{missing_keys[0]}' -> '{resolved_key}'")
                    else:
                        logger.warning(f"[QUERY] ✗ Could not resolve missing key '{missing_keys[0]}'")
                
                raise ValueError(
                    f"Tool validation error. Missing key: {missing_keys[0] if missing_keys else 'unknown'}. "
                    f"Available columns: {available_cols}. "
                    f"Please try your query again - column names are resolved automatically."
                )
            elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ValueError("Invalid or missing API key. Please check your LLM API keys in .env file.")
            elif "rate limit" in error_msg.lower():
                raise ValueError("API rate limit exceeded. Please try again later.")
            else:
                raise ValueError(f"Query execution failed: {error_msg}")
        
        # Extract tool calls from intermediate steps
        tool_calls = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                tool_calls.append({
                    "tool": step[0].tool,
                    "input": str(step[0].tool_input),
                    "output": str(step[1])[:500]  # Limit output length
                })
        
        # Extract final answer
        answer = result.get("output", "No answer generated")
        
        # Store analysis in memory
        try:
            self.memory.add_analysis(
                self.dataset_name,
                user_query,
                answer,
                {"tool_calls": tool_calls}
            )
        except Exception as e:
            logger.warning(f"Failed to store analysis in memory: {e}")
        
        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "dataset_info": dataset_info
        }
