"""SQL execution tool for DataFrames."""
from typing import Optional, Dict, Any
from langchain.tools import BaseTool
from pydantic import Field
import pandas as pd

# Try to import DuckDB, fallback to pandasql
try:
    import duckdb
    USE_DUCKDB = True
except ImportError:
    USE_DUCKDB = False
    try:
        from pandasql import sqldf
        USE_PANDASQL = True
    except ImportError:
        USE_PANDASQL = False


class SQLExecutorTool(BaseTool):
    """Tool for executing SQL queries on pandas DataFrames."""
    
    name = "sql_executor"
    description = """Execute SQL queries on loaded datasets. 
    Use this tool to query, filter, aggregate, and join data.
    The dataset is available as a table named 'df'.
    Returns the query results as JSON."""
    
    df: pd.DataFrame = Field(description="The DataFrame to query")
    
    def _run(self, query: str) -> str:
        """Execute SQL query on DataFrame."""
        try:
            if USE_DUCKDB:
                # Use DuckDB for SQL execution on pandas DataFrames
                conn = duckdb.connect()
                conn.register('df', self.df)
                result = conn.execute(query).fetchdf()
                conn.close()
            elif USE_PANDASQL:
                # Fallback to pandasql
                # pandasql expects a dict with DataFrame variables
                result = sqldf(query, {'df': self.df})
            else:
                return "Error: No SQL engine available. Please install duckdb or pandasql."
            
            # Convert to JSON for easy parsing
            return result.to_json(orient="records")
        except Exception as e:
            return f"Error executing SQL: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async execution."""
        return self._run(query)
