#!/usr/bin/env python3
"""Test script to diagnose query endpoint issues."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("=" * 60)
print("DataBrain AI Agent - Query Endpoint Diagnostic")
print("=" * 60)

# Check API keys
print("\n1. Checking API Keys...")
openai_key = os.getenv("OPENAI_API_KEY")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")

if openai_key:
    print(f"  ✅ OPENAI_API_KEY: {'*' * 20}{openai_key[-4:] if len(openai_key) > 4 else '***'}")
else:
    print("  ❌ OPENAI_API_KEY: Not set")

if deepseek_key:
    print(f"  ✅ DEEPSEEK_API_KEY: {'*' * 20}{deepseek_key[-4:] if len(deepseek_key) > 4 else '***'}")
else:
    print("  ❌ DEEPSEEK_API_KEY: Not set")

if not openai_key and not deepseek_key:
    print("\n  ⚠️  WARNING: No API keys configured!")
    print("  Set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env file")

# Test LLM initialization
print("\n2. Testing LLM Initialization...")
try:
    from databrain_agent.backend.llm.multi_llm import MultiLLM
    llm_router = MultiLLM()
    providers = llm_router.get_available_providers()
    print(f"  ✅ Available providers: {providers}")
    
    if providers:
        try:
            llm = llm_router.get_llm()
            print(f"  ✅ Default LLM: {type(llm).__name__}")
        except Exception as e:
            print(f"  ❌ Failed to get LLM: {e}")
    else:
        print("  ❌ No LLM providers available")
except Exception as e:
    print(f"  ❌ Failed to initialize MultiLLM: {e}")
    import traceback
    traceback.print_exc()

# Test agent creation
print("\n3. Testing Agent Creation...")
try:
    from databrain_agent.backend.data.vector_store import DatasetVectorStore
    from databrain_agent.backend.agent.memory import RAGMemory
    from databrain_agent.backend.agent.orchestrator import AgentOrchestrator
    import pandas as pd
    
    # Create a test DataFrame
    test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    vector_store = DatasetVectorStore()
    memory = RAGMemory(vector_store)
    
    orchestrator = AgentOrchestrator(
        llm_router=llm_router,
        memory=memory,
        df=test_df,
        dataset_name="test_dataset"
    )
    
    if orchestrator.agent is None:
        print("  ❌ Agent is None - initialization failed")
    else:
        print(f"  ✅ Agent created: {type(orchestrator.agent).__name__}")
        
        # Test a simple query
        print("\n4. Testing Query Execution...")
        try:
            result = orchestrator.query("What columns are in this dataset?")
            print(f"  ✅ Query successful!")
            print(f"  Response: {result['response'][:100]}...")
        except Exception as e:
            print(f"  ❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"  ❌ Failed to create orchestrator: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)
