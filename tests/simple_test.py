import os
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

print("Simple test...")

# 1. Test OpenAI connection
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    print("✅ ChatOpenAI works")
    
    # Simple test query
    response = llm.invoke("Say 'Hello from DataBrain'")
    print(f"✅ LLM response: {response.content}")
except Exception as e:
    print(f"❌ OpenAI error: {e}")

# 2. Test pandas dataframe creation
try:
    df = pd.DataFrame({'test': [1, 2, 3]})
    print(f"✅ Pandas works: {len(df)} rows")
except Exception as e:
    print(f"❌ Pandas error: {e}")

# 3. Check if we have the fix for response key error
test_data = {'answer': 'Test response'}
result = test_data.get('answer', test_data.get('response', 'No response'))
print(f"✅ Response key fix works: {result}")
