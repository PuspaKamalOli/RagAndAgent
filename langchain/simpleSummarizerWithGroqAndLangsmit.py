from dotenv import load_dotenv        
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load .env variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
langsmith = os.getenv("LANGSMITH_API_KEY")

# Setup LangChain environment
os.environ['LANGSMITH_API_KEY'] = langsmith
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model='llama3-8b-8192', temperature=0.1, max_tokens=2000)


# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that summarizes content."),
    ("user", "Summarize the following content in 5 points: {content}")
])

# Create the chain
chain = prompt | llm | StrOutputParser()

# Invoke with correct variable name
response = chain.invoke({
    "content": "The quick brown fox jumps over the lazy dog. The fox is very clever and quick, while the dog is slow and lazy. They both live in a beautiful forest with lots of trees and flowers. The fox loves to play tricks on the dog, but they are still good friends."
})

print(response)
