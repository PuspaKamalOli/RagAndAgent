RagAndAgent
RagAndAgent is a collection of AI-driven applications leveraging LangChain and Phidata for text summarization, Retrieval-Augmented Generation (RAG), and multimodal content analysis. The project includes a Streamlit-based web application for analyzing videos, audio, PDFs, and YouTube links, as well as LangChain scripts for summarization, RAG, and API services using Groq and Ollama models.
Project Structure
RagAndAgent/
├── langchain/
│   ├── LangserveForApi.py              # FastAPI service for text summarization using Groq
│   ├── mcp.py                          # RAG pipeline with Groq and FAISS using MCP
│   ├── RAG.py                          # RAG pipeline with Ollama and FAISS for text analysis
│   ├── simpleSummarizerWithGroqAndLangsmit.py  # Simple text summarization with Groq and LangSmith
│   └── simpleSummarizerWithOllamaStreamlitAndLangsmit.py  # Streamlit app for text queries with Ollama
├── video_summarizer_with_phidata/
│   ├── app.py                          # Streamlit app for multimodal analysis (video, audio, PDF, YouTube)
│   └── requirements.txt                # Project dependencies
├── .env                                # Environment variables (not tracked)
├── .gitignore                          # Git ignore file
└── agent/                              # Virtual environment

Components

LangChain Subfolder:

LangserveForApi.py: A FastAPI service exposing a text summarization endpoint using Groq (LLaMA3-8b).
mcp.py: A RAG pipeline with Groq (DeepSeek-Coder:7b) and FAISS, using Memory-Augmented Context Prompting (MCP).
RAG.py: A RAG pipeline with Ollama (LLaMA3.2:1b) and FAISS for analyzing text files (e.g., speech.txt).
simpleSummarizerWithGroqAndLangsmit.py: A script for summarizing text in 5 points using Groq and LangSmith tracing.
simpleSummarizerWithOllamaStreamlitAndLangsmit.py: A Streamlit app for answering user queries with Ollama and LangSmith.


Video Summarizer with Phidata:

app.py: A Streamlit web app that analyzes uploaded videos, audio, PDFs, or YouTube links using Phidata’s agent framework, Google’s Gemini model, and tools like DuckDuckGo and YouTubeTools. Supports PDF text extraction with PyPDF2 and audio transcription (requires openai-whisper).



Prerequisites

Python 3.8+
Git
ffmpeg (for audio transcription in the Streamlit app)
Windows: Install via choco install ffmpeg (with Chocolatey) or download from FFmpeg.
macOS: brew install ffmpeg
Linux: sudo apt-get install ffmpeg
Verify: ffmpeg -version


API keys:
Google API key (for Gemini in app.py)
Groq API key (for LangChain scripts)
LangSmith API key (for tracing in LangChain scripts)



Setup

Clone the Repository:
git clone <repository-url>
cd RagAndAgent


Set Up the Virtual Environment:
python -m venv agent
source agent/bin/activate  # Linux/macOS
agent\Scripts\activate     # Windows


Install Dependencies:
pip install -r video_summarizer_with_phidata/requirements.txt

Note: The requirements.txt includes dependencies for both subfolders:

langchain_groq, langchain_community, langchain_core, langchain-ollama
streamlit, fastapi, uvicorn, langserve
sentence-transformers, faiss-cpu, pydantic
phidata, google-generativeai, duckduckgo-search, youtube-transcript-api, py2pdf
whisper (ensure openai-whisper is installed)


Install openai-whisper for Audio Transcription:
pip uninstall whisper  # Remove any conflicting whisper package
pip install openai-whisper


Create a .env File:Copy video_summarizer_with_phidata/.env.example (if provided) to .env in the root directory and fill in your API keys:
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name

The .env file is excluded from version control via .gitignore.

Verify ffmpeg:Ensure ffmpeg is installed and accessible:
ffmpeg -version



Running the Applications
1. Multimodal Streamlit App (video_summarizer_with_phidata/app.py)

Purpose: Analyze uploaded videos, audio, PDFs, or YouTube links with Phidata and Gemini.
Run:cd video_summarizer_with_phidata
streamlit run app.py


Usage:
Open http://localhost:8501 in your browser.
Upload a file (.mp4, .mov, .avi, .mp3, .wav, .pdf) or enter a YouTube link.
Enter a query (e.g., “Summarize the video” or “What are the lyrics?”).
Click “Analyze” to view results.


Notes:
Audio transcription requires openai-whisper and ffmpeg.
PDF analysis uses PyPDF2 for text extraction.
YouTube analysis uses YouTubeTools and DuckDuckGo for web research.



2. LangChain Scripts (langchain/)
a. FastAPI Summarization Service (LangserveForApi.py)

Purpose: Expose a text summarization endpoint using Groq (LLaMA3-8b).
Run:cd langchain
python LangserveForApi.py


Usage:
Access http://localhost:8000/docs for the FastAPI Swagger UI.
Use the /summarizer endpoint to summarize text.


Notes: Requires GROQ_API_KEY in .env.

b. RAG with MCP (mcp.py)

Purpose: Run a RAG pipeline with Groq (DeepSeek-Coder:7b) and FAISS, using MCP.
Run:cd langchain
python mcp.py


Usage: Outputs the answer to “What is LangChain used for?” based on the FAISS index.
Notes:
Requires a pre-existing faiss_index/ directory.
Ensure GROQ_API_KEY is set.



c. R'inc'AG Pipeline (RAG.py)

Purpose: Analyze text files (e.g., speech.txt) using Ollama (LLaMA3.2:1b) and FAISS.
Run:cd langchain
python RAG.py


Usage: Outputs the answer to “What is the speech about?” based on speech.txt.
Notes:
Requires speech.txt in the specified path (update the path in the script).
Uses Ollama locally; ensure it’s running (ollama serve).



d. Simple Summarizer with Groq (simpleSummarizerWithGroqAndLangsmit.py)

Purpose: Summarize text in 5 points using Groq and LangSmith tracing.
Run:cd langchain
python simpleSummarizerWithGroqAndLangsmit.py


Usage: Outputs a summary of hardcoded text.
Notes: Requires GROQ_API_KEY and LANGSMITH_API_KEY.

e. Streamlit Query App with Ollama (simpleSummarizerWithOllamaStreamlitAndLangsmit.py)

Purpose: Answer user queries via a Streamlit interface using Ollama.
Run:cd langchain
streamlit run simpleSummarizerWithOllamaStreamlitAndLangsmit.py


Usage:
Open http://localhost:8501.
Enter a query (e.g., “What is AI?”).
View the response.


Notes: Requires Ollama running locally (ollama serve) and LANGSMITH_API_KEY.

Troubleshooting
Whisper Error (module 'whisper' has no attribute 'load_model')

Cause: Incorrect whisper package installed.
Fix:pip uninstall whisper
pip install openai-whisper


Ensure ffmpeg is installed (see Prerequisites).
Test Whisper:import whisper
model = whisper.load_model("base")
print(model)





Missing speech.txt in RAG.py

Update the file path in RAG.py to point to your text file:loader = TextLoader("path/to/speech.txt")



FAISS Index Missing in mcp.py

Generate the faiss_index/ directory by running a script to create and save the index:from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Assume documents is a list of LangChain documents
vectorstore = FAISS.from_documents(documents, embedding)
vectorstore.save_local("faiss_index")



API Key Errors

Ensure all required keys are in .env (see Setup).
If keys are missing, obtain them from:
Google Cloud Console (for GOOGLE_API_KEY)
Groq Console (for GROQ_API_KEY)
LangSmith Dashboard (for LANGSMITH_API_KEY, LANGCHAIN_API_KEY)



Contributing

Fork the repository and create a pull request with your changes.
Ensure code follows PEP 8 style guidelines.
Add tests for new features.

License
This project is licensed under the MIT License. See the LICENSE file for details (if applicable).
Acknowledgments

Built with LangChain, Phidata, Streamlit, and FastAPI.
Thanks to Groq, Ollama, and Google for providing AI models and APIs.

