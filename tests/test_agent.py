import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing DataBrain AI Agent...")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file")
    print("Create a .env file with: OPENAI_API_KEY=your-key-here")
    exit(1)
else:
    print(f"✅ OPENAI_API_KEY found (first 10 chars): {api_key[:10]}...")

# Create test dataframe
test_data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'score': [85, 92, 78, 88, 95]
}
df = pd.DataFrame(test_data)
print(f"\n✅ Created test dataframe: {len(df)} rows, {len(df.columns)} columns")

# Initialize LLM
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    print("✅ ChatOpenAI initialized")
    
    # Create agent
    agent = create_pandas_daagent(llm, df, verbose=False)
    print("✅ Pandas dataframe agent created")
    
    # Test query
    print("\n🤖 Testing query: 'How many rows are in the data?'")
    result = agent.run("How many rows are in the data?")
    print(f"✅ Agent response: {result}")
    
    print("\n🎉 All tests passed! DataBrain AI Agent is working!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Check your OpenAI API key and internet connection")
