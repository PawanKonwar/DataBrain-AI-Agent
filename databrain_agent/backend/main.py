"""
DataBrain AI Agent - FastAPI Backend Server

This module provides the main FastAPI application for the DataBrain AI Agent,
including endpoints for dataset upload, query processing, and cost tracking.

Copyright (c) 2024 DataBrain AI Agent Contributors
License: MIT
"""
# Import compatibility shim FIRST before any LangChain imports
try:
    from databrain_agent.backend import compat  # noqa: F401
except ImportError:
    pass  # Compatibility shim not critical if it fails

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import pandas as pd
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
# Try loading from project root first, then backend directory
project_root = Path(__file__).parent.parent.parent
env_paths = [
    project_root / ".env",
    Path(__file__).parent / ".env",
    Path.cwd() / ".env"
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        env_loaded = True
        logging.info(f"Loaded .env file from: {env_path}")
        break

if not env_loaded:
    # Try default load_dotenv() which looks in current directory and parent directories
    load_dotenv()
    logging.warning("No .env file found in expected locations. Using default load_dotenv() behavior.")

# Validate API keys on startup
def validate_api_keys():
    """Validate that at least one API key is configured."""
    openai_key = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not openai_key and not deepseek_key:
        error_msg = """
╔════════════════════════════════════════════════════════════════╗
║  ERROR: No API Keys Configured                                ║
╠════════════════════════════════════════════════════════════════╣
║  The DataBrain AI Agent requires at least one API key:        ║
║                                                                ║
║  1. Create a .env file in the project root                    ║
║  2. Add at least one of the following:                        ║
║     - OPENAI_API_KEY=your_key_here                            ║
║     - DEEPSEEK_API_KEY=your_key_here                           ║
║                                                                ║
║  Get API keys from:                                            ║
║  - OpenAI: https://platform.openai.com/api-keys                ║
║  - DeepSeek: https://platform.deepseek.com/api_keys            ║
║                                                                ║
║  See .env.example for a template.                             ║
╚════════════════════════════════════════════════════════════════╝
"""
        logging.error(error_msg)
        return False
    
    configured = []
    if openai_key and openai_key.strip() and openai_key != "your_openai_api_key_here":
        configured.append("OpenAI")
    if deepseek_key and deepseek_key.strip() and deepseek_key != "your_deepseek_api_key_here":
        configured.append("DeepSeek")
    
    if not configured:
        error_msg = """
╔════════════════════════════════════════════════════════════════╗
║  ERROR: API Keys Not Properly Configured                     ║
╠════════════════════════════════════════════════════════════════╣
║  API keys found in .env but they appear to be placeholders.   ║
║  Please replace the placeholder values with your actual keys. ║
╚════════════════════════════════════════════════════════════════╝
"""
        logging.error(error_msg)
        return False
    
    logging.info(f"✓ API keys configured: {', '.join(configured)}")
    return True

# Validate API keys on startup
api_keys_valid = validate_api_keys()

from databrain_agent.backend.data.loader import DataLoader
from databrain_agent.backend.data.vector_store import DatasetVectorStore
from databrain_agent.backend.llm.multi_llm import MultiLLM
from databrain_agent.backend.agent.memory import RAGMemory
from databrain_agent.backend.agent.orchestrator import AgentOrchestrator
from databrain_agent.backend.schemas import DatasetInfo, AgentResponse

app = FastAPI(title="DataBrain AI Agent API")

# Check API keys on startup
if not api_keys_valid:
    logging.warning("⚠️  Server starting without valid API keys. Some features may not work.")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
datasets: Dict[str, pd.DataFrame] = {}
orchestrators: Dict[str, AgentOrchestrator] = {}
vector_store = DatasetVectorStore()
llm_router = MultiLLM()
data_loader = DataLoader()


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    global vector_store, llm_router
    vector_store = DatasetVectorStore()
    llm_router = MultiLLM()


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...), dataset_name: Optional[str] = None):
    """Upload and load a dataset."""
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load dataset based on file type
        if suffix == ".csv":
            df = data_loader.load_csv(tmp_path)
        elif suffix in [".xlsx", ".xls"]:
            df = data_loader.load_excel(tmp_path)
        elif suffix == ".json":
            df = data_loader.load_json(tmp_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Generate dataset name if not provided
        if not dataset_name:
            dataset_name = Path(file.filename).stem
        
        # DataFrame columns are already normalized by the loader
        # Log them for debugging
        logging.info(f"Dataset '{dataset_name}' loaded with {len(df.columns)} columns: {list(df.columns)}")
        
        # Store dataset
        datasets[dataset_name] = df
        
        # Get dataset info
        dataset_info = data_loader.get_dataset_info(df, dataset_name)
        
        # Log column names for debugging
        logging.info(f"Dataset '{dataset_name}' loaded with columns: {list(df.columns)}")
        
        # Clear old schema for this dataset before storing new one
        try:
            vector_store.collection.delete(ids=[f"schema_{dataset_name}"])
            logging.info(f"Cleared old schema for dataset '{dataset_name}'")
        except Exception as e:
            logging.debug(f"No old schema to clear for '{dataset_name}': {e}")
        
        # Store schema in vector store (this will clean and update the schema)
        memory = RAGMemory(vector_store)
        # Schema columns are already normalized from DataFrame
        cleaned_schema = dataset_info["schema"].copy()
        if "columns" in cleaned_schema:
            cleaned_schema["columns"] = [
                col for col in cleaned_schema["columns"]
                if col and str(col).strip() and not str(col).startswith("unnamed")
            ]
        memory.add_dataset_schema(dataset_name, cleaned_schema)
        logging.info(f"Stored schema for '{dataset_name}' with columns: {cleaned_schema.get('columns', [])}")
        
        # Create orchestrator for this dataset with error handling
        try:
            orchestrators[dataset_name] = AgentOrchestrator(
                llm_router=llm_router,
                memory=memory,
                df=df,
                dataset_name=dataset_name
            )
        except Exception as e:
            # Log error but don't fail the upload
            logging.error(f"Failed to create orchestrator for {dataset_name}: {e}")
            # Store dataset anyway, orchestrator can be created later
            # Don't raise - allow dataset to be uploaded even if agent init fails
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "info": dataset_info
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/datasets")
async def list_datasets():
    """List all loaded datasets."""
    dataset_list = []
    for name, df in datasets.items():
        info = data_loader.get_dataset_info(df, name)
        dataset_list.append({
            "name": name,
            "row_count": info["row_count"],
            "column_count": info["column_count"],
            "columns": info["columns"]
        })
    
    return {"datasets": dataset_list}


@app.post("/api/query")
async def query_agent(dataset_name: str, query: str, llm_provider: Optional[str] = None):
    """Query the agent about a dataset."""
    # Check API keys first
    openai_key = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    
    has_openai = bool(openai_key and openai_key.strip() and openai_key != "your_openai_api_key_here")
    has_deepseek = bool(deepseek_key and deepseek_key.strip() and deepseek_key != "your_deepseek_api_key_here")
    
    if not has_openai and not has_deepseek:
        error_detail = (
            "No API keys configured. Please set at least one API key in your .env file:\n\n"
            "1. Create a .env file in the project root directory\n"
            "2. Add at least one of:\n"
            "   - OPENAI_API_KEY=your_key_here\n"
            "   - DEEPSEEK_API_KEY=your_key_here\n\n"
            "Get API keys from:\n"
            "- OpenAI: https://platform.openai.com/api-keys\n"
            "- DeepSeek: https://platform.deepseek.com/api_keys\n\n"
            "See .env.example for a template."
        )
        raise HTTPException(
            status_code=400,
            detail=error_detail
        )
    
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    # Ensure dataset is clean before creating orchestrator
    if dataset_name in datasets:
        df = datasets[dataset_name]
        # Clean columns one more time (safety check)
        df.columns = [col.strip() if isinstance(col, str) else str(col).strip() 
                     for col in df.columns]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
        df = df.reset_index(drop=True)
        datasets[dataset_name] = df
        logging.info(f"Cleaned dataset '{dataset_name}' columns: {list(df.columns)}")
    
    if dataset_name not in orchestrators:
        # Create orchestrator if it doesn't exist
        try:
            memory = RAGMemory(vector_store)
            orchestrators[dataset_name] = AgentOrchestrator(
                llm_router=llm_router,
                memory=memory,
                df=datasets[dataset_name],
                dataset_name=dataset_name
            )
        except ValueError as e:
            # ValueError means API key or configuration issue
            logging.error(f"Failed to create orchestrator (configuration error): {e}")
            error_msg = str(e)
            if "api key" in error_msg.lower() or "no llm providers" in error_msg.lower():
                # Provide helpful error message about API keys
                api_key_help = (
                    "\n\nAPI Key Configuration Help:\n"
                    "1. Create a .env file in the project root\n"
                    "2. Add your API keys:\n"
                    "   OPENAI_API_KEY=your_key_here\n"
                    "   DEEPSEEK_API_KEY=your_key_here\n"
                    "3. Restart the server\n\n"
                    "Get keys from:\n"
                    "- OpenAI: https://platform.openai.com/api-keys\n"
                    "- DeepSeek: https://platform.deepseek.com/api_keys"
                )
                error_msg = error_msg + api_key_help
                detail = f"LLM configuration error: {error_msg}. Please check your API keys in .env file."
            else:
                detail = f"Agent initialization failed: {error_msg}. Please check your LLM configuration."
            raise HTTPException(status_code=500, detail=detail)
        except Exception as e:
            # Other errors
            logging.error(f"Failed to create orchestrator: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize agent: {str(e)}. Please check your LLM API keys and configuration."
            )
    
    try:
        # Verify orchestrator exists and is ready
        if dataset_name not in orchestrators:
            raise HTTPException(
                status_code=500,
                detail="Agent not initialized for this dataset. Please try uploading the dataset again."
            )
        
        orchestrator = orchestrators[dataset_name]
        
        # Verify agent is initialized
        if not hasattr(orchestrator, 'agent') or orchestrator.agent is None:
            raise HTTPException(
                status_code=500,
                detail="Agent not properly initialized. Please check your LLM API keys and try again."
            )
        
        result = orchestrator.query(query, llm_provider)
        
        # Handle result dict - it returns "answer" not "response"
        answer = result.get("answer", result.get("response", "No response generated"))
        tool_calls = result.get("tool_calls", [])
        
        return AgentResponse(
            message=answer,
            tool_calls=tool_calls,
            dataset_context=DatasetInfo(**data_loader.get_dataset_info(datasets[dataset_name], dataset_name))
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # User-facing errors (like query execution failures, API key errors)
        error_msg = str(e)
        logging.error(f"Query validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except KeyError as e:
        # Handle missing keys in response dict
        error_msg = str(e)
        logging.error(f"Query response key error: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Query failed: Invalid response format. Missing key: {error_msg}. Please try again."
        )
    except AttributeError as e:
        # Handle attribute errors (like missing 'response' attribute)
        error_msg = str(e)
        logging.error(f"Query attribute error: {error_msg}", exc_info=True)
        if "'response'" in error_msg or "'answer'" in error_msg:
            raise HTTPException(
                status_code=500,
                detail="Query failed: Agent response format error. Please check your LLM configuration and try again."
            )
        raise HTTPException(status_code=500, detail=f"Query failed: {error_msg}")
    except Exception as e:
        # Unexpected errors
        error_msg = str(e)
        logging.error(f"Query error: {error_msg}", exc_info=True)
        
        # Provide more helpful error messages
        if "api key" in error_msg.lower() or "authentication" in error_msg.lower():
            detail = "Invalid or missing API key. Please check your LLM API keys in .env file."
        elif "No LLM providers" in error_msg:
            detail = "No LLM providers configured. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env file."
        elif "requests" in error_msg.lower() and "module" in error_msg.lower():
            detail = "Missing 'requests' module. Please install it: pip install requests"
        elif "'response'" in error_msg or "'answer'" in error_msg:
            detail = "Query failed: Agent response format error. Please check your LLM configuration."
        else:
            detail = f"Query failed: {error_msg}"
        
        raise HTTPException(status_code=500, detail=detail)


@app.get("/api/llm-providers")
async def get_llm_providers():
    """Get available LLM providers."""
    providers = llm_router.get_available_providers()
    # Add "Auto" as an option
    if "Auto" not in providers:
        providers = ["Auto"] + providers
    return {
        "providers": providers,
        "default": llm_router.default_provider
    }


@app.get("/api/cost-tracking")
async def get_cost_tracking():
    """Get cost tracking information."""
    return {
        "total_cost": llm_router.cost_tracker.get_total_cost(),
        "by_provider": llm_router.cost_tracker.get_cost_by_provider()
    }


@app.delete("/api/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset."""
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    del datasets[dataset_name]
    if dataset_name in orchestrators:
        del orchestrators[dataset_name]
    
    return {"status": "success", "message": f"Dataset '{dataset_name}' deleted"}


@app.post("/api/test-chart")
async def test_chart(dataset_name: str, chart_type: str, 
                     x_column: Optional[str] = None,
                     y_column: Optional[str] = None,
                     group_by: Optional[str] = None,
                     title: Optional[str] = None):
    """
    Test endpoint for Chart Generator tool (STEP 1: Independent testing).
    
    This endpoint allows testing the chart generator independently before
    full agent integration.
    
    Example:
    POST /api/test-chart
    {
        "dataset_name": "movies",
        "chart_type": "bar",
        "x_column": "genre",
        "title": "Movies by Genre"
    }
    """
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    try:
        from databrain_agent.backend.tools.chart_tool import ChartGeneratorTool
        
        df = datasets[dataset_name]
        
        # Log chart request
        logging.info(f"Generating {chart_type} chart for dataset '{dataset_name}'")
        logging.info(f"Parameters: x_column={x_column}, y_column={y_column}, group_by={group_by}, title={title}")
        
        chart_tool = ChartGeneratorTool(df=df)
        
        result = chart_tool._run(
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            group_by=group_by,
            title=title
        )
        
        # Parse JSON result
        import json
        if isinstance(result, str):
            try:
                result_data = json.loads(result)
            except json.JSONDecodeError:
                # If it's not JSON, it's an error message
                return {"error": result, "status": "error"}
        else:
            result_data = result
        
        # Check for errors
        if "error" in result_data or result_data.get("status") == "error":
            error_msg = result_data.get("error", "Unknown error")
            logging.error(f"Chart generation failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        logging.info(f"Chart '{chart_type}' generated successfully")
        
        return {
            "status": "success",
            "chart_type": result_data.get("chart_type"),
            "title": result_data.get("title"),
            "image_base64": result_data.get("image_base64"),
            "message": f"Chart '{chart_type}' generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Chart generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@app.get("/api/test-rag-memory")
async def test_rag_memory(dataset_name: Optional[str] = None, query: Optional[str] = None):
    """
    Test endpoint for RAG Memory (STEP 2: Independent testing).
    
    This endpoint allows testing RAG memory independently:
    - Store/retrieve schemas
    - Store/retrieve analysis results
    - Search for relevant context
    
    Example:
    GET /api/test-rag-memory?dataset_name=movies&query=columns
    """
    try:
        memory = RAGMemory(vector_store)
        
        results = {
            "status": "success",
            "tests": {}
        }
        
        # Test 1: Get dataset schema
        if dataset_name:
            schema = memory.get_dataset_schema(dataset_name)
            results["tests"]["schema_retrieval"] = {
                "success": schema is not None,
                "columns": schema.get("columns", []) if schema else [],
                "message": f"Schema {'found' if schema else 'not found'} for '{dataset_name}'"
            }
        
        # Test 2: Search context
        if query and dataset_name:
            context = memory.get_relevant_context(query, dataset_name)
            results["tests"]["context_search"] = {
                "success": len(context) > 0,
                "context_length": len(context),
                "context_preview": context[:200] if context else "",
                "message": f"Found {len(context)} characters of context" if context else "No context found"
            }
        
        # Test 3: List all stored schemas
        try:
            # Get all schemas from collection
            all_data = vector_store.collection.get()
            schema_count = sum(1 for m in (all_data.get('metadatas', []) or []) 
                             if m and m.get('type') == 'schema')
            analysis_count = sum(1 for m in (all_data.get('metadatas', []) or []) 
                               if m and m.get('type') == 'analysis')
            
            results["tests"]["storage_stats"] = {
                "schemas_stored": schema_count,
                "analyses_stored": analysis_count,
                "total_items": len(all_data.get('ids', []))
            }
        except Exception as e:
            results["tests"]["storage_stats"] = {
                "error": str(e)
            }
        
        return results
        
    except Exception as e:
        logging.error(f"RAG Memory test error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG Memory test failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "DataBrain AI Agent API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
