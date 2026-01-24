"""Statistical analysis tool."""
from typing import Optional, Dict, Any
from langchain.tools import BaseTool
from pydantic import Field
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class StatsCalculatorTool(BaseTool):
    """Tool for calculating statistical measures."""
    
    name = "stats_calculator"
    description = """Calculate statistical measures on data.
    Available operations: describe, correlation, mean, median, std, min, max, count.
    Returns statistical results as JSON.
    Available columns: Use the actual column names from the dataset."""
    
    df: pd.DataFrame = Field(description="The DataFrame to analyze")
    
    def _validate_column(self, column: str) -> str:
        """Validate and normalize column name."""
        if column is None:
            return None
        
        from databrain_agent.backend.data.loader import normalize_column_name
        
        normalized_input = normalize_column_name(str(column))
        actual_columns_normalized = {normalize_column_name(str(col)): col for col in self.df.columns}
        
        if normalized_input in actual_columns_normalized:
            matched_column = actual_columns_normalized[normalized_input]
            logger.debug(f"Column validation: '{column}' -> '{normalized_input}' -> '{matched_column}'")
            return matched_column
        
        available = ', '.join(self.df.columns.tolist())
        logger.warning(f"Column '{column}' (normalized: '{normalized_input}') not found. Available: {available}")
        raise ValueError(f"Column '{column}' not found. Available columns: {available}")
    
    def _run(self, operation: str, column: Optional[str] = None) -> str:
        """Calculate statistics."""
        try:
            numeric_df = self.df.select_dtypes(include=[np.number])
            
            if operation == "describe":
                result = self.df.describe().to_dict()
                return json.dumps(result, default=str)
            
            elif operation == "correlation":
                if numeric_df.empty:
                    return "Error: No numeric columns for correlation"
                result = numeric_df.corr().to_dict()
                return json.dumps(result, default=str)
            
            elif operation == "mean":
                if column:
                    column = self._validate_column(column)
                    result = {"column": column, "mean": self.df[column].mean()}
                else:
                    result = numeric_df.mean().to_dict()
                return json.dumps(result, default=str)
            
            elif operation == "median":
                if column:
                    column = self._validate_column(column)
                    result = {"column": column, "median": self.df[column].median()}
                else:
                    result = numeric_df.median().to_dict()
                return json.dumps(result, default=str)
            
            elif operation == "std":
                if column:
                    column = self._validate_column(column)
                    result = {"column": column, "std": self.df[column].std()}
                else:
                    result = numeric_df.std().to_dict()
                return json.dumps(result, default=str)
            
            elif operation == "min":
                if column:
                    column = self._validate_column(column)
                    result = {"column": column, "min": self.df[column].min()}
                else:
                    result = numeric_df.min().to_dict()
                return json.dumps(result, default=str)
            
            elif operation == "max":
                if column:
                    column = self._validate_column(column)
                    result = {"column": column, "max": self.df[column].max()}
                else:
                    result = numeric_df.max().to_dict()
                return json.dumps(result, default=str)
            
            elif operation == "count":
                if column:
                    column = self._validate_column(column)
                    result = {"column": column, "count": len(self.df[column].dropna())}
                else:
                    result = {"total_rows": len(self.df), "columns": {col: len(self.df[col].dropna()) for col in self.df.columns}}
                return json.dumps(result, default=str)
            
            else:
                return f"Error: Unsupported operation '{operation}'. Use: describe, correlation, mean, median, std, min, max, count"
        
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error calculating statistics: {str(e)}"
    
    async def _arun(self, operation: str, column: Optional[str] = None) -> str:
        """Async execution."""
        return self._run(operation, column)
