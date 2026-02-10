"""
Data manipulation tool for DataBrain AI Agent.

Provides functionality to filter, sort, group, and manipulate DataFrames.

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""
from typing import Optional, Dict, Any, List, Union, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


def _strip_column_quotes(value) -> Optional[str]:
    """Strip quotes from column names the LLM may send."""
    if value is None:
        return None
    s = str(value).replace("'", "").replace('"', "").strip()
    return s if s else None


class DataManipulationInput(BaseModel):
    """Input schema for data_manipulator."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    operation: str = Field(description="filter, sort, group_by, select_columns, head, tail, unique")
    column: Optional[str] = Field(default=None, description="Column name", alias="col")
    group_column: Optional[str] = Field(default=None, description="Group by column", alias="group_col")
    agg_column: Optional[str] = Field(default=None, description="Aggregation column", alias="agg_col")
    columns: Optional[Union[List[str], str]] = Field(default=None, description="Column list", alias="cols")
    value: Optional[Any] = None
    operator: Optional[str] = None
    ascending: Optional[bool] = None
    agg_func: Optional[str] = None
    n: Optional[int] = None


class DataManipulationTool(BaseTool):
    """Tool for basic data manipulation operations."""

    name = "data_manipulator"
    description = """Perform data manipulation operations.
    Available operations: filter, sort, group_by, select_columns, head, tail, unique.
    Returns manipulated data as JSON.
    Available columns: Use the actual column names from the dataset."""

    df: pd.DataFrame = Field(description="The DataFrame to manipulate")
    args_schema: Type[BaseModel] = DataManipulationInput

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
    
    def _run(self, operation: str, column: Optional[str] = None, group_column: Optional[str] = None,
             agg_column: Optional[str] = None, columns: Optional[Union[List[str], str]] = None,
             value: Optional[Any] = None, operator: Optional[str] = None, ascending: Optional[bool] = None,
             agg_func: Optional[str] = None, n: Optional[int] = None) -> str:
        """Perform data manipulation."""
        try:
            column = _strip_column_quotes(column) if column else None
            group_column = _strip_column_quotes(group_column) if group_column else None
            agg_column = _strip_column_quotes(agg_column) if agg_column else None

            if operation is None:
                return json.dumps({
                    "error": "Missing required parameter 'operation'. Available operations: filter, sort, group_by, select_columns, head, tail, unique",
                    "received_params": ["operation"]
                })
            if operation == "filter":
                if not column:
                    return "Please specify a column."
                column = self._validate_column(column)
                op_val = operator or "=="
                
                if op_val == "==":
                    result = self.df[self.df[column] == value]
                elif op_val == ">":
                    result = self.df[self.df[column] > value]
                elif op_val == "<":
                    result = self.df[self.df[column] < value]
                elif op_val == ">=":
                    result = self.df[self.df[column] >= value]
                elif op_val == "<=":
                    result = self.df[self.df[column] <= value]
                else:
                    return f"Error: Unsupported operator '{op_val}'"
                
                return result.to_json(orient="records")
            
            elif operation == "sort":
                if column:
                    column = self._validate_column(column)
                else:
                    column = self.df.columns[0]
                asc = ascending if ascending is not None else True
                result = self.df.sort_values(by=column, ascending=asc)
                return result.to_json(orient="records")
            
            elif operation == "group_by":
                if not group_column or not agg_column:
                    return f"Error: 'group_column' and 'agg_column' required. Available columns: {', '.join(self.df.columns.tolist())}"
                group_column = self._validate_column(group_column)
                agg_column = self._validate_column(agg_column)
                af = agg_func or "sum"
                
                grouped = self.df.groupby(group_column)[agg_column].agg(af)
                return grouped.to_json()
            
            elif operation == "select_columns":
                cols = columns or []
                if isinstance(cols, str):
                    cols = [c.strip() for c in cols.split(",")]
                cols = [c for c in (_strip_column_quotes(col) or (str(col).strip() if col else None) for col in cols) if c]
                if not cols:
                    return f"Error: 'columns' parameter required. Available columns: {', '.join(self.df.columns.tolist())}"
                validated_columns = [self._validate_column(col) for col in cols]
                result = self.df[validated_columns]
                return result.to_json(orient="records")
            
            elif operation == "head":
                rows = n if n is not None else 5
                return self.df.head(rows).to_json(orient="records")
            
            elif operation == "tail":
                rows = n if n is not None else 5
                return self.df.tail(rows).to_json(orient="records")
            
            elif operation == "unique":
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
    
    async def _arun(self, operation: str, column: Optional[str] = None, group_column: Optional[str] = None,
                    agg_column: Optional[str] = None, columns: Optional[Union[List[str], str]] = None,
                    value: Optional[Any] = None, operator: Optional[str] = None, ascending: Optional[bool] = None,
                    agg_func: Optional[str] = None, n: Optional[int] = None) -> str:
        """Async execution."""
        return self._run(operation, column, group_column, agg_column, columns,
                        value, operator, ascending, agg_func, n)
