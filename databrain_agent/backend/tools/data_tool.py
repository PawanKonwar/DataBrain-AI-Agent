"""
Data manipulation tool for DataBrain AI Agent.

Provides functionality to filter, sort, group, and manipulate DataFrames.

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""
from typing import Optional, Dict, Any, List
from langchain.tools import BaseTool
from pydantic import Field
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class DataManipulationTool(BaseTool):
    """Tool for basic data manipulation operations."""
    
    name = "data_manipulator"
    description = """Perform data manipulation operations.
    Available operations: filter, sort, group_by, select_columns, head, tail, unique.
    Returns manipulated data as JSON.
    Available columns: Use the actual column names from the dataset."""
    
    df: pd.DataFrame = Field(description="The DataFrame to manipulate")
    
    def _validate_column(self, column: str) -> str:
        """
        Validate and normalize column name - schema-agnostic.
        Handles case-insensitive matching, whitespace, and quoted names.
        ALWAYS strips quotes FIRST before any matching.
        """
        logger.info(f"[VALIDATE_COLUMN] ===== Validating column =====")
        logger.info(f"[VALIDATE_COLUMN] Input: '{column}' (type: {type(column)}, repr: {repr(column)})")
        
        if column is None:
            logger.warning("[VALIDATE_COLUMN] Column input is None")
            return None
        
        available_columns = list(self.df.columns)
        logger.info(f"[VALIDATE_COLUMN] Available columns: {available_columns}")
        
        # CRITICAL: Strip quotes FIRST before any processing
        input_str = str(column)
        cleaned_input = input_str.strip()
        # Remove outer quotes if present (handles '"id"', "'id'", etc.)
        if (cleaned_input.startswith('"') and cleaned_input.endswith('"')) or \
           (cleaned_input.startswith("'") and cleaned_input.endswith("'")):
            cleaned_input = cleaned_input[1:-1].strip()
        logger.info(f"[VALIDATE_COLUMN] After quote stripping: '{cleaned_input}' (from '{column}')")
        
        # Try exact match first
        if cleaned_input in available_columns:
            logger.info(f"[VALIDATE_COLUMN] ✓ Exact match found: '{cleaned_input}'")
            return cleaned_input
        
        # Try case-insensitive match
        column_lower = cleaned_input.lower()
        logger.info(f"[VALIDATE_COLUMN] Attempting case-insensitive match with: '{column_lower}'")
        for col in available_columns:
            col_str = str(col)
            if col_str.lower() == column_lower:
                logger.info(f"[VALIDATE_COLUMN] ✓ Case-insensitive match: '{column}' -> '{col_str}'")
                return col_str
        
        # Try with whitespace normalization
        column_normalized = column_lower.replace(' ', '').replace('_', '')
        logger.info(f"[VALIDATE_COLUMN] Attempting normalized match with: '{column_normalized}'")
        for col in available_columns:
            col_str = str(col)
            col_normalized = col_str.lower().replace(' ', '').replace('_', '')
            if col_normalized == column_normalized:
                logger.info(f"[VALIDATE_COLUMN] ✓ Normalized match: '{column}' -> '{col_str}'")
                return col_str
        
        # Column doesn't exist
        available = ', '.join([str(c) for c in available_columns])
        logger.error(f"[VALIDATE_COLUMN] ✗ Column '{column}' (cleaned: '{cleaned_input}') not found.")
        logger.error(f"[VALIDATE_COLUMN] Available columns: {available}")
        logger.error(f"[VALIDATE_COLUMN] Input type: {type(column)}, Input repr: {repr(column)}")
        raise ValueError(f"Column '{column}' not found. Available columns: {available}")
    
    def _run(self, operation: str = None, **kwargs) -> str:
        """Perform data manipulation."""
        try:
            # Handle missing operation parameter - it might be passed as keyword arg
            if operation is None:
                # Try to get operation from kwargs if it wasn't passed as positional
                operation = kwargs.pop('operation', None)
                if operation is None:
                    logger.error("DataManipulationTool called without 'operation' parameter")
                    logger.error(f"Received kwargs: {kwargs}")
                    
                    # Check if this looks like a chart request that was misrouted
                    chart_keywords = ['chart', 'graph', 'plot', 'visualize', 'bar', 'line', 'scatter', 'histogram']
                    received_keys = [str(k).lower() for k in kwargs.keys()]
                    received_values = [str(v).lower() for v in kwargs.values() if isinstance(v, str)]
                    all_text = ' '.join(received_keys + received_values)
                    
                    if any(keyword in all_text for keyword in chart_keywords) or \
                       any(key in ['chart_type', 'x_column', 'y_column', 'group_by'] for key in kwargs.keys()):
                        return json.dumps({
                            "error": "Chart request detected but wrong tool was used",
                            "hint": "Please use the chart_generator tool for chart requests, not data_manipulator",
                            "received_params": list(kwargs.keys()),
                            "suggestion": "Retry the request with chart_generator tool"
                        })
                    
                    return json.dumps({
                        "error": "Missing required parameter 'operation'. Available operations: filter, sort, group_by, select_columns, head, tail, unique",
                        "hint": "If you're trying to generate a chart, use the chart_generator tool instead.",
                        "received_params": list(kwargs.keys())
                    })
            if operation == "filter":
                # Filter by column and value
                column = kwargs.get("column")
                if not column:
                    return f"Error: 'column' parameter required. Available columns: {', '.join(self.df.columns.tolist())}"
                column = self._validate_column(column)
                value = kwargs.get("value")
                operator = kwargs.get("operator", "==")
                
                if operator == "==":
                    result = self.df[self.df[column] == value]
                elif operator == ">":
                    result = self.df[self.df[column] > value]
                elif operator == "<":
                    result = self.df[self.df[column] < value]
                elif operator == ">=":
                    result = self.df[self.df[column] >= value]
                elif operator == "<=":
                    result = self.df[self.df[column] <= value]
                else:
                    return f"Error: Unsupported operator '{operator}'"
                
                return result.to_json(orient="records")
            
            elif operation == "sort":
                column = kwargs.get("column")
                if column:
                    column = self._validate_column(column)
                else:
                    column = self.df.columns[0]
                ascending = kwargs.get("ascending", True)
                result = self.df.sort_values(by=column, ascending=ascending)
                return result.to_json(orient="records")
            
            elif operation == "group_by":
                group_column = kwargs.get("group_column")
                agg_column = kwargs.get("agg_column")
                if not group_column or not agg_column:
                    return f"Error: 'group_column' and 'agg_column' required. Available columns: {', '.join(self.df.columns.tolist())}"
                group_column = self._validate_column(group_column)
                agg_column = self._validate_column(agg_column)
                agg_func = kwargs.get("agg_func", "sum")
                
                grouped = self.df.groupby(group_column)[agg_column].agg(agg_func)
                return grouped.to_json()
            
            elif operation == "select_columns":
                columns = kwargs.get("columns", [])
                if isinstance(columns, str):
                    columns = [col.strip() for col in columns.split(",")]
                if not columns:
                    return f"Error: 'columns' parameter required. Available columns: {', '.join(self.df.columns.tolist())}"
                # Validate all columns
                validated_columns = []
                for col in columns:
                    validated_columns.append(self._validate_column(col))
                result = self.df[validated_columns]
                return result.to_json(orient="records")
            
            elif operation == "head":
                n = kwargs.get("n", 5)
                return self.df.head(n).to_json(orient="records")
            
            elif operation == "tail":
                n = kwargs.get("n", 5)
                return self.df.tail(n).to_json(orient="records")
            
            elif operation == "unique":
                column = kwargs.get("column")
                if column:
                    column = self._validate_column(column)
                    result = self.df[column].unique().tolist()
                else:
                    result = {col: self.df[col].unique().tolist() for col in self.df.columns}
                return json.dumps(result)
            
            else:
                return f"Error: Unsupported operation '{operation}'. Use: filter, sort, group_by, select_columns, head, tail, unique"
        
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error performing data manipulation: {str(e)}"
    
    async def _arun(self, operation: str, **kwargs) -> str:
        """Async execution."""
        return self._run(operation, **kwargs)
