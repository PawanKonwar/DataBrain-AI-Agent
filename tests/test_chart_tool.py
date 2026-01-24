"""Standalone test script for Chart Generator tool."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from databrain_agent.backend.tools.chart_tool import ChartGeneratorTool

def test_chart_generator():
    """Test the Chart Generator tool with sample data."""
    print("=" * 60)
    print("Testing Chart Generator Tool")
    print("=" * 60)
    
    # Create sample DataFrame
    sample_data = {
        'category': ['A', 'B', 'C', 'D', 'E'] * 4,
        'value': [10, 20, 15, 25, 30] * 4,
        'price': [100, 200, 150, 250, 300] * 4,
        'quantity': [1, 2, 1.5, 2.5, 3] * 4
    }
    df = pd.DataFrame(sample_data)
    
    print(f"\nSample DataFrame:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    # Initialize tool
    chart_tool = ChartGeneratorTool(df=df)
    print(f"\n✓ Chart Generator Tool initialized")
    
    # Test 1: Bar chart
    print("\n" + "-" * 60)
    print("Test 1: Bar Chart")
    print("-" * 60)
    try:
        result = chart_tool._run(
            chart_type="bar",
            x_column="category",
            title="Category Distribution"
        )
        import json
        if isinstance(result, str):
            try:
                result_data = json.loads(result)
            except:
                result_data = {"error": result}
        else:
            result_data = result
        if isinstance(result_data, dict) and "image_base64" in result_data:
            print(f"✓ Bar chart generated successfully")
            print(f"  Chart type: {result_data.get('chart_type')}")
            print(f"  Image size: {len(result_data.get('image_base64', ''))} bytes")
        else:
            print(f"✗ Error: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Scatter plot
    print("\n" + "-" * 60)
    print("Test 2: Scatter Plot")
    print("-" * 60)
    try:
        result = chart_tool._run(
            chart_type="scatter",
            x_column="value",
            y_column="price",
            title="Value vs Price"
        )
        import json
        if isinstance(result, str):
            try:
                result_data = json.loads(result)
            except:
                result_data = {"error": result}
        else:
            result_data = result
        if isinstance(result_data, dict) and "image_base64" in result_data:
            print(f"✓ Scatter plot generated successfully")
            print(f"  Chart type: {result_data.get('chart_type')}")
        else:
            print(f"✗ Error: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Histogram
    print("\n" + "-" * 60)
    print("Test 3: Histogram")
    print("-" * 60)
    try:
        result = chart_tool._run(
            chart_type="histogram",
            x_column="value",
            title="Value Distribution"
        )
        import json
        if isinstance(result, str):
            try:
                result_data = json.loads(result)
            except:
                result_data = {"error": result}
        else:
            result_data = result
        if isinstance(result_data, dict) and "image_base64" in result_data:
            print(f"✓ Histogram generated successfully")
            print(f"  Chart type: {result_data.get('chart_type')}")
        else:
            print(f"✗ Error: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Heatmap
    print("\n" + "-" * 60)
    print("Test 4: Heatmap (Correlation)")
    print("-" * 60)
    try:
        result = chart_tool._run(
            chart_type="heatmap",
            title="Correlation Heatmap"
        )
        import json
        if isinstance(result, str):
            try:
                result_data = json.loads(result)
            except:
                result_data = {"error": result}
        else:
            result_data = result
        if isinstance(result_data, dict) and "image_base64" in result_data:
            print(f"✓ Heatmap generated successfully")
            print(f"  Chart type: {result_data.get('chart_type')}")
        else:
            print(f"✗ Error: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 5: Bar chart with group_by
    print("\n" + "-" * 60)
    print("Test 5: Bar Chart with Group By")
    print("-" * 60)
    try:
        result = chart_tool._run(
            chart_type="bar",
            y_column="value",
            group_by="category",
            title="Value by Category"
        )
        import json
        if isinstance(result, str):
            try:
                result_data = json.loads(result)
            except:
                result_data = {"error": result}
        else:
            result_data = result
        if isinstance(result_data, dict) and "image_base64" in result_data:
            print(f"✓ Grouped bar chart generated successfully")
            print(f"  Chart type: {result_data.get('chart_type')}")
        else:
            print(f"✗ Error: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Chart Generator Tool Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_chart_generator()
