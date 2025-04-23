#step-1
import os
# Disable parallelism warning from tokenizer libraries (especially HuggingFace tokenizers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

#step-2
# --- Document Loading ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# Load a plain text file (alternatively you can load a PDF, web page, etc.)
loader = TextLoader("/Users/puspakamaloli/Desktop/langchain /speech.txt")  
# Load the contents into LangChain document format
docs = loader.load()

#step-3
# --- Text Splitting into Chunks ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Split text into chunks of 1000 characters with 20 characters overlap between chunks
# Overlap helps maintain context between adjacent chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# Split all documents and store them for embedding
documents = text_splitter.split_documents(docs)

#step-4
# --- Embedding Model Initialization ---
from langchain_community.embeddings import OllamaEmbeddings
# Use Ollama to generate vector embeddings for each text chunk (based on LLaMA3 1B model)
embedding_model = OllamaEmbeddings(model="llama3.2:1b")

# Alternative: Use HuggingFace embeddings instead of Ollama
# from langchain_community.embeddings import HuggingFaceEmbeddings
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#step-5
# --- Vectorstore Creation using FAISS ---
from langchain_community.vectorstores import FAISS, Chroma
# Create a FAISS vector store using the first 30 embedded chunks
# FAISS allows fast similarity search over dense vectors
db = FAISS.from_documents(documents[:30], embedding_model)

# Alternative: Use Chroma instead of FAISS
# db = Chroma.from_documents(documents[:30], embedding_model)

# Optional: Use OpenAI Embeddings instead of Ollama
# from langchain_community.embeddings import OpenAIEmbeddings
# db = FAISS.from_documents(documents[:30], OpenAIEmbeddings())

# --- Perform Similarity Search (Optional) ---
# This shows how you can query the vector store directly
# query = "An attention function can be described as mapping a query"
# result = db.similarity_search(query)
# print(result)

#step-6
# --- LLM Initialization ---
from langchain_community.llms import Ollama
# Load a local Ollama LLM instance with LLaMA3 model (same model used for embeddings)
llm = Ollama(model="llama3.2:1b")

# Alternative: Use Groq API with Qwen LLM
# from langchain_groq import ChatGroq
# groq_api_key = os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     print("Error: GROQ_API_KEY not found.")
#     exit(1)
# llm = ChatGroq(groq_api_key=groq_api_key, model="qwen-qwq-32b")

#step-7
# --- Prompt Template Setup ---
from langchain_core.prompts import ChatPromptTemplate
# Prompt instructs the LLM to answer only using the retrieved context
# Structured to encourage step-by-step reasoning
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

#step-8
# --- Document Chain Creation ---
from langchain.chains.combine_documents import create_stuff_documents_chain
# Combines context documents with the LLM and prompt to form a "stuff" document chain
# The "stuff" chain simply stuffs the retrieved documents into the prompt as-is
document_chain = create_stuff_documents_chain(llm, prompt)

#step-9
# --- Retriever Setup ---
# Turn the vectorstore into a retriever, retrieving top 3 most similar chunks per query
retriever = db.as_retriever(search_kwargs={"k": 3})

#step-10
# --- RAG Chain Construction ---
from langchain.chains import create_retrieval_chain
# Combines retriever and document chain into a RAG pipeline
# Retrieves relevant chunks and feeds them into LLM with the prompt to generate an answer
retrieval_chain = create_retrieval_chain(retriever, document_chain)

#step-11
# --- Run the Chain and Get Answer ---
try:
    # Invoke the full RAG chain with a user query
    response = retrieval_chain.invoke({"input": "what is the speech about"})
    print("\nAnswer:\n", response["answer"])
except Exception as e:
    print(f"Error invoking chain: {str(e)}")

