"""Standalone test script for RAG Memory (ChromaDB)."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from databrain_agent.backend.data.vector_store import DatasetVectorStore
from databrain_agent.backend.agent.memory import RAGMemory

def test_rag_memory():
    """Test RAG Memory with sample data."""
    print("=" * 60)
    print("Testing RAG Memory (ChromaDB)")
    print("=" * 60)
    
    # Initialize vector store and memory
    vector_store = DatasetVectorStore(persist_directory="./test_chroma_db")
    memory = RAGMemory(vector_store)
    
    print("\n✓ RAG Memory initialized")
    
    # Test 1: Store dataset schema
    print("\n" + "-" * 60)
    print("Test 1: Store Dataset Schema")
    print("-" * 60)
    try:
        sample_schema = {
            "columns": ["id", "name", "age", "salary"],
            "dtypes": {"id": "int64", "name": "object", "age": "int64", "salary": "float64"},
            "shape": [100, 4],
            "null_counts": {"id": 0, "name": 0, "age": 2, "salary": 1},
            "sample_data": [
                {"id": 1, "name": "Alice", "age": 30, "salary": 50000},
                {"id": 2, "name": "Bob", "age": 25, "salary": 45000}
            ]
        }
        
        memory.add_dataset_schema("test_dataset", sample_schema)
        print("✓ Schema stored successfully")
        
        # Retrieve it
        retrieved = memory.get_dataset_schema("test_dataset")
        if retrieved:
            print(f"✓ Schema retrieved: {len(retrieved.get('columns', []))} columns")
        else:
            print("✗ Schema retrieval failed")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Store analysis result
    print("\n" + "-" * 60)
    print("Test 2: Store Analysis Result")
    print("-" * 60)
    try:
        memory.add_analysis(
            dataset_name="test_dataset",
            query="What is the average salary?",
            result="The average salary is $47,500",
            tool_used="stats_calculator"
        )
        print("✓ Analysis result stored successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Search for relevant context
    print("\n" + "-" * 60)
    print("Test 3: Search Relevant Context")
    print("-" * 60)
    try:
        context = memory.get_relevant_context("average salary", "test_dataset")
        if context:
            print(f"✓ Context retrieved: {len(context)} characters")
            print(f"  Preview: {context[:100]}...")
        else:
            print("⚠ No context found (this is OK for first test)")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Conversation memory
    print("\n" + "-" * 60)
    print("Test 4: Conversation Memory")
    print("-" * 60)
    try:
        memory.add_conversation_message("user", "What columns are in the dataset?")
        memory.add_conversation_message("assistant", "The dataset has columns: id, name, age, salary")
        
        history = memory.get_conversation_history()
        print(f"✓ Conversation history: {len(history)} messages")
        
        # Clear conversation
        memory.clear_conversation()
        history_after_clear = memory.get_conversation_history()
        print(f"✓ Conversation cleared: {len(history_after_clear)} messages remaining")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("RAG Memory Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_memory()
