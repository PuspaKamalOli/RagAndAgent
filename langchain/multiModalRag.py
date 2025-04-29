# Import necessary libraries
import os # For interacting with the operating system (like listing files)
import streamlit as st # For building the web application UI
from langchain_core.prompts import ChatPromptTemplate # To create a template for the chat prompt
from langchain_core.vectorstores import InMemoryVectorStore # A simple in-memory vector store for storing document embeddings
from langchain_ollama import OllamaEmbeddings # To generate embeddings using an Ollama model
from langchain_ollama.llms import OllamaLLM # To use an Ollama model as a Language Model
from langchain_text_splitters import RecursiveCharacterTextSplitter # To split text into smaller chunks
# Used for partitioning PDFs, specifically extracting text, images, and tables
from unstructured.partition.pdf import partition_pdf
# Constants for partitioning strategies
from unstructured.partition.utils.constants import PartitionStrategy

# Define the template for the question-answering prompt
# This template instructs the LLM on how to answer based on the provided context
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# Define directories for storing uploaded PDFs and extracted figures
pdfs_directory = 'multi-modal-rag/pdfs/'
figures_directory = 'multi-modal-rag/figures/'

# Ensure the directories exist
os.makedirs(pdfs_directory, exist_ok=True)
os.makedirs(figures_directory, exist_ok=True)


# Initialize the embedding model from Ollama
# This model is used to convert text into numerical vectors (embeddings)
embeddings = OllamaEmbeddings(model="llama3.2")
# Initialize the in-memory vector store with the embedding model
# This store will hold the embeddings of the document chunks
vector_store = InMemoryVectorStore(embeddings)

# Initialize the Language Model (LLM) from Ollama
# This model will be used for generating answers and describing images
model = OllamaLLM(model="gemma3:27b")

# Function to handle the upload of a PDF file
def upload_pdf(file):
    # Open the file in binary write mode and save it to the specified directory
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())
    st.success(f"Uploaded {file.name}") # Display a success message in the Streamlit UI

# Function to load and process a PDF file
def load_pdf(file_path):
    # Partition the PDF using the unstructured library
    # - strategy=PartitionStrategy.HI_RES uses a high-resolution strategy
    # - extract_image_block_types specifies which types of elements to extract as images
    # - extract_image_block_output_dir specifies where to save the extracted images
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=figures_directory
    )

    # Filter out image and table elements, keeping only text elements
    text_elements = [element.text for element in elements if element.category not in ["Image", "Table"]]

    # Iterate through the extracted figures directory
    for file in os.listdir(figures_directory):
        # For each image file, extract text description using the LLM
        extracted_text = extract_text(figures_directory + file)
        # Append the extracted text description to the list of text elements
        text_elements.append(extracted_text)

    # Join all text elements (original text and image descriptions) into a single string
    return "\n\n".join(text_elements)

# Function to extract text (description) from an image file using the LLM
def extract_text(file_path):
    # Bind the image file path to the model context
    # This prepares the model to process the image
    model_with_image_context = model.bind(images=[file_path])
    # Invoke the model with a prompt asking it to describe the picture
    return model_with_image_context.invoke("Tell me what do you see in this picture.")

# Function to split the combined text into smaller chunks
def split_text(text):
    # Initialize a RecursiveCharacterTextSplitter
    # - chunk_size: maximum size of each chunk
    # - chunk_overlap: number of characters to overlap between chunks
    # - add_start_index: adds the starting character index to the metadata of each chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    # Split the input text into chunks
    return text_splitter.split_text(text)

# Function to index the text chunks into the vector store
def index_docs(texts):
    # Add the list of text chunks to the vector store
    vector_store.add_texts(texts)
    st.success("Document indexed successfully!") # Display a success message

# Function to retrieve relevant documents (text chunks) based on a query
def retrieve_docs(query):
    # Perform a similarity search in the vector store using the query
    # This finds chunks whose embeddings are most similar to the query's embedding
    return vector_store.similarity_search(query)

# Function to answer a question using retrieved documents
def answer_question(question, documents):
    # Combine the content of the retrieved documents into a single context string
    context = "\n\n".join([doc.page_content for doc in documents])
    # Create a chat prompt from the predefined template
    prompt = ChatPromptTemplate.from_template(template)
    # Create a processing chain: prompt -> model
    # The prompt is formatted with the question and context, then passed to the model
    chain = prompt | model

    # Invoke the chain with the question and context and return the model's response
    return chain.invoke({"question": question, "context": context})

# --- Streamlit UI Layout ---

st.title("Multi-Modal RAG with Ollama") # Set the title of the Streamlit app

# File uploader widget for the user to upload a PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf", # Accept only PDF files
    accept_multiple_files=False # Allow only one file upload at a time
)

# --- Main Application Logic ---

# Check if a file has been uploaded
if uploaded_file:
    # If a file is uploaded, process it
    upload_pdf(uploaded_file) # Save the uploaded file
    # Load and process the PDF, including image description
    text = load_pdf(pdfs_directory + uploaded_file.name)
    # Split the processed text into chunks
    chunked_texts = split_text(text)
    # Index the chunks into the vector store
    index_docs(chunked_texts)

    # Create a chat input box for the user to type questions
    question = st.chat_input("Ask a question about the document...")

    # Check if the user has entered a question
    if question:
        # Display the user's question in the chat interface
        st.chat_message("user").write(question)
        # Retrieve relevant document chunks based on the question
        related_documents = retrieve_docs(question)
        # Get the answer from the LLM using the question and retrieved documents
        answer = answer_question(question, related_documents)
        # Display the assistant's answer in the chat interface
        st.chat_message("assistant").write(answer)

