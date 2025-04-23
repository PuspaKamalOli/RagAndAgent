from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langserve import add_routes
import os
import uvicorn

# Load environment variables from .env
()



# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192",
    temperature=0.2,
    max_tokens=2000
)

# Define the summarization prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that summarizes content."),
    ("user", "Summarize the following content in 5 points:\n\n{content}")
])

# Chain
chain = prompt | llm | StrOutputParser()

# Create FastAPI app and expose the chain
app = FastAPI(title="Groq Summarizer API")
add_routes(app,chain, path="/summarizer")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

