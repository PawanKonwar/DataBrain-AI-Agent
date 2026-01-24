"""RAG memory for datasets and conversation history."""
from typing import List, Dict, Any, Optional
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from databrain_agent.backend.data.vector_store import DatasetVectorStore
from databrain_agent.backend.schemas import ConversationMessage, AnalysisResult


class RAGMemory:
    """RAG-based memory for dataset schemas and past analyses."""
    
    def __init__(self, vector_store: DatasetVectorStore):
        """Initialize RAG memory."""
        self.vector_store = vector_store
        self.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    
    def add_dataset_schema(self, dataset_name: str, schema: Dict[str, Any]):
        """Store dataset schema in vector store."""
        self.vector_store.add_dataset_schema(dataset_name, schema)
    
    def add_analysis(self, dataset_name: str, query: str, result: Any, tool_used: str):
        """Store analysis result."""
        self.vector_store.add_analysis(dataset_name, query, result, tool_used)
    
    def get_relevant_context(self, query: str, dataset_name: Optional[str] = None) -> str:
        """Retrieve relevant context from vector store."""
        contexts = self.vector_store.search_relevant_context(query, dataset_name, n_results=3)
        
        if not contexts:
            return ""
        
        context_parts = []
        for ctx in contexts:
            context_parts.append(ctx["document"])
        
        return "\n\n".join(context_parts)
    
    def get_dataset_schema(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get stored dataset schema."""
        return self.vector_store.get_dataset_schema(dataset_name)
    
    def add_conversation_message(self, role: str, content: str, tool_calls: List[Dict] = None):
        """Add message to conversation history."""
        if role == "user":
            self.conversation_memory.chat_memory.add_user_message(content)
        elif role == "assistant":
            self.conversation_memory.chat_memory.add_ai_message(content)
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get conversation history."""
        return self.conversation_memory.chat_memory.messages
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_memory.clear()
