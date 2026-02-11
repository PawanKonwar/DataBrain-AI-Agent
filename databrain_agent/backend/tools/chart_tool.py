"""
Chart generation tool for DataBrain AI Agent.

Provides functionality to generate various data visualizations using matplotlib
and seaborn.

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""
from typing import Optional, Dict, Any, List, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import json
import logging

logger = logging.getLogger(__name__)


class ChartInput(BaseModel):
    """Input schema for chart_generator."""
    chart_type: str = Field(description="bar, line, scatter, histogram, box, heatmap")
    column: Optional[str] = Field(default=None, description="The name of the column to analyze")
    x_column: Optional[str] = Field(default=None, description="X-axis column")
    y_column: Optional[str] = Field(default=None, description="Y-axis column")
    group_by: Optional[str] = Field(default=None, description="Group by column")
    title: Optional[str] = Field(default=None, description="Chart title")


class ChartGeneratorTool(BaseTool):
    """Tool for generating data visualizations."""

    name = "chart_generator"
    description = """Generate charts and visualizations from data.
    Supported chart types: bar, line, scatter, histogram, box, heatmap.
    Returns base64-encoded image data.
    Available columns: Use the actual column names from the dataset."""

    df: pd.DataFrame = Field(description="The DataFrame to visualize")
    args_schema: Type[BaseModel] = ChartInput

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
    
    def _run(self, chart_type: str, column: Optional[str] = None, x_column: Optional[str] = None,
             y_column: Optional[str] = None, group_by: Optional[str] = None, title: Optional[str] = None) -> str:
        """Generate chart from DataFrame."""
        try:
            if column:
                column = column.replace("'", "").replace('"', "").strip() or None
            if x_column:
                x_column = x_column.replace("'", "").replace('"', "").strip() or None
            if y_column:
                y_column = y_column.replace("'", "").replace('"', "").strip() or None
            if group_by:
                group_by = group_by.replace("'", "").replace('"', "").strip() or None
            # Use column as x_column fallback when x_column not provided
            if not x_column and column:
                x_column = column
            if not chart_type:
                return json.dumps({"error": "Chart type is required"})
            
            chart_type = chart_type.lower().strip()
            
            # Check if DataFrame is empty
            if self.df.empty:
                return json.dumps({"error": "DataFrame is empty"})
            
            # Set up figure
            plt.figure(figsize=(10, 6))
            plt.clf()  # Clear any previous plots
            
            # Generate chart based on type
            if chart_type == "bar":
                if group_by:
                    group_by = self._validate_column(group_by)
                    y_col = self._validate_column(y_column) if y_column else self.df.columns[0]
                    data = self.df.groupby(group_by)[y_col].sum()
                    data.plot(kind='bar', rot=45)
                    plt.ylabel(y_col)
                    plt.xlabel(group_by)
                else:
                    x_col = self._validate_column(x_column) if x_column else self.df.columns[0]
                    self.df[x_col].value_counts().head(20).plot(kind='bar', rot=45)
                    plt.ylabel('Count')
                    plt.xlabel(x_col)
            
            elif chart_type == "line":
                if x_column and y_column:
                    x_col = self._validate_column(x_column)
                    y_col = self._validate_column(y_column)
                    self.df.plot(x=x_col, y=y_col, kind='line', marker='o')
                    plt.ylabel(y_col)
                    plt.xlabel(x_col)
                else:
                    # Plot all numeric columns
                    numeric_cols = self.df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        self.df[numeric_cols].plot(kind='line')
                    else:
                        return json.dumps({"error": "No numeric columns found for line chart"})
            
            elif chart_type == "scatter":
                x_col = self._validate_column(x_column) if x_column else self.df.columns[0]
                y_col = self._validate_column(y_column) if y_column else (self.df.columns[1] if len(self.df.columns) > 1 else self.df.columns[0])
                self.df.plot(x=x_col, y=y_col, kind='scatter', alpha=0.6)
                plt.ylabel(y_col)
                plt.xlabel(x_col)
            
            elif chart_type == "histogram":
                col = self._validate_column(x_column) if x_column else self.df.columns[0]
                self.df[col].hist(bins=30, edgecolor='black')
                plt.ylabel('Frequency')
                plt.xlabel(col)
            
            elif chart_type == "box":
                col = self._validate_column(y_column) if y_column else self.df.columns[0]
                self.df.boxplot(column=col)
                plt.ylabel(col)
            
            elif chart_type == "heatmap":
                numeric_df = self.df.select_dtypes(include=['number'])
                if numeric_df.empty:
                    return json.dumps({"error": "No numeric columns found for heatmap"})
                if len(numeric_df.columns) < 2:
                    return json.dumps({"error": "Need at least 2 numeric columns for heatmap"})
                sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
            
            else:
                return json.dumps({
                    "error": f"Unsupported chart type '{chart_type}'. Supported types: bar, line, scatter, histogram, box, heatmap"
                })
            
            # Add title
            if title:
                plt.title(title, fontsize=14, fontweight='bold')
            else:
                plt.title(f"{chart_type.capitalize()} Chart", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            logger.info(f"Successfully generated {chart_type} chart")
            
            return json.dumps({
                "chart_type": chart_type,
                "image_base64": image_base64,
                "title": title or f"{chart_type.capitalize()} Chart",
                "status": "success"
            })
        
        except ValueError as e:
            logger.error(f"Chart generation ValueError: {e}")
            return json.dumps({"error": str(e), "status": "error"})
        except Exception as e:
            logger.error(f"Chart generation error: {e}", exc_info=True)
            return json.dumps({"error": f"Error generating chart: {str(e)}", "status": "error"})
    
    async def _arun(self, chart_type: str, column: Optional[str] = None, x_column: Optional[str] = None,
                    y_column: Optional[str] = None, group_by: Optional[str] = None, title: Optional[str] = None) -> str:
        """Async execution."""
        return self._run(chart_type, column, x_column, y_column, group_by, title)
