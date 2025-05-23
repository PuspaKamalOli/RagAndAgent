# RagAndAgent

## Overview
RagAndAgent is a versatile AI project combining **LangChain** for text summarization and Retrieval-Augmented Generation (RAG) with a **Streamlit-based multimodal AI agent** for analyzing videos, audio, PDFs, and YouTube links. The project leverages models like Groq, Ollama, and Google’s Gemini, along with tools such as FAISS, HuggingFace embeddings, and Phidata, to provide robust text and multimedia processing capabilities. It supports applications like content summarization, question answering, and contextual analysis, with LangSmith integration for tracing.

## Features
- Text summarization and RAG pipelines using Groq and Ollama models.
- Multimodal analysis of videos, audio, PDFs, and YouTube links with Phidata and Gemini.
- FastAPI service for exposing summarization endpoints.
- Streamlit web interfaces for interactive query answering and file analysis.
- Support for FAISS vector stores and HuggingFace embeddings.
- Audio transcription (via `openai-whisper`) and PDF text extraction (via PyPDF2).
- YouTube video analysis with transcript extraction and web research (DuckDuckGo).

## Project Structure
```
RagAndAgent/
├── langchain/
│   ├── LangserveForApi.py              # FastAPI service for text summarization (Groq)
│   ├── mcp.py                          # RAG with Groq and FAISS (MCP-enabled)
│   ├── RAG.py                          # RAG with Ollama and FAISS for text files
│   ├── simpleSummarizerWithGroqAndLangsmit.py  # Text summarization with Groq and LangSmith
│   └── simpleSummarizerWithOllamaStreamlitAndLangsmit.py  # Streamlit app for queries (Ollama)
├── video_summarizer_with_phidata/
│   ├── app.py                          # Streamlit app for multimodal analysis
│   └── requirements.txt                # Project dependencies
├── .env                                # Environment variables (not tracked)
├── .gitignore                          # Git ignore file
├── agent/                              # Virtual environment
└── README.md                           # This file
```

## Requirements
- Python 3.8+
- Git
- `ffmpeg` (for audio transcription in `app.py`)
- API keys for Google, Groq, and LangSmith

Install dependencies:
```bash
pip install -r video_summarizer_with_phidata/requirements.txt
```

Install `ffmpeg`:
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Use `choco install ffmpeg` (with Chocolatey) or download from [FFmpeg](https://ffmpeg.org/download.html)
- Verify: `ffmpeg -version`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd RagAndAgent
   ```

2. **Set Up the Virtual Environment**:
   ```bash
   python -m venv agent
   source agent/bin/activate  # Linux/macOS
   agent\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r video_summarizer_with_phidata/requirements.txt
   ```

4. **Install `openai-whisper` for Audio Transcription**:
   ```bash
   pip uninstall whisper  # Remove conflicting packages
   pip install openai-whisper
   ```

5. **Configure Environment Variables**:
   Create a `.env` file in the root directory with the following:
   ```plaintext
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=your_project_name
   ```
   Note: The `.env` file is excluded from version control via `.gitignore`.

## Usage

### 1. Multimodal Streamlit App (`video_summarizer_with_phidata/app.py`)
- **Purpose**: Analyze videos, audio, PDFs, or YouTube links using Phidata and Gemini.
- **Run**:
  ```bash
  cd video_summarizer_with_phidata
  streamlit run app.py
  ```
- **Usage**:
  - Open `http://localhost:8501` in your browser.
  - Upload a file (`.mp4`, `.mov`, `.avi`, `.mp3`, `.wav`, `.pdf`) or enter a YouTube link.
  - Enter a query (e.g., “Summarize the video” or “Extract lyrics from the audio”).
  - Click “Analyze” to view the response.
- **Notes**:
  - Requires `GOOGLE_API_KEY` for Gemini.
  - Audio transcription uses `openai-whisper` and `ffmpeg`.
  - PDFs are processed with PyPDF2; YouTube links use YouTubeTools and DuckDuckGo.

### 2. LangChain Scripts (`langchain/`)

#### a. FastAPI Summarization Service (`LangserveForApi.py`)
- **Purpose**: Expose a text summarization API using Groq (LLaMA3-8b).
- **Run**:
  ```bash
  cd langchain
  python LangserveForApi.py
  ```
- **Usage**:
  - Access `http://localhost:8000/docs` for the FastAPI Swagger UI.
  - Send POST requests to `/summarizer` with text to summarize.
- **Notes**: Requires `GROQ_API_KEY`.

#### b. RAG with MCP (`mcp.py`)
- **Purpose**: Run a RAG pipeline with Groq (DeepSeek-Coder:7b) and FAISS, using MCP.
- **Run**:
  ```bash
  cd langchain
  python mcp.py
  ```
- **Usage**: Outputs the answer to “What is LangChain used for?” using a FAISS index.
- **Notes**:
  - Requires a pre-existing `faiss_index/` directory.
  - Set `GROQ_API_KEY`.

#### c. RAG Pipeline (`RAG.py`)
- **Purpose**: Analyze text files (e.g., `speech.txt`) using Ollama (LLaMA3.2:1b) and FAISS.
- **Run**:
  ```bash
  cd langchain
  python RAG.py
  ```
- **Usage**: Outputs the answer to “What is the speech about?” based on `speech.txt`.
- **Notes**:
  - Update the file path in `RAG.py` to your `speech.txt` location.
  - Requires Ollama running (`ollama serve`).

#### d. Simple Summarizer (`simpleSummarizerWithGroqAndLangsmit.py`)
- **Purpose**: Summarize text in 5 points using Groq and LangSmith.
- **Run**:
  ```bash
  cd langchain
  python simpleSummarizerWithGroqAndLangsmit.py
  ```
- **Usage**: Outputs a summary of hardcoded text.
- **Notes**: Requires `GROQ_API_KEY` and `LANGSMITH_API_KEY`.

#### e. Streamlit Query App (`simpleSummarizerWithOllamaStreamlitAndLangsmit.py`)
- **Purpose**: Answer queries via a Streamlit interface using Ollama.
- **Run**:
  ```bash
  cd langchain
  streamlit run simpleSummarizerWithOllamaStreamlitAndLangsmit.py
  ```
- **Usage**:
  - Open `http://localhost:8501`.
  - Enter a query (e.g., “Explain RAG”).
  - View the response.
- **Notes**: Requires Ollama (`ollama serve`) and `LANGSMITH_API_KEY`.

## Configuration
- **Environment Variables**: Configure API keys and LangSmith settings in `.env` (see Installation).
- **Ollama**: Ensure the Ollama server is running for `RAG.py` and `simpleSummarizerWithOllamaStreamlitAndLangsmit.py`:
  ```bash
  ollama serve
  ```
- **FAISS Index**: For `mcp.py`, generate a `faiss_index/` directory if missing (see Troubleshooting).
- **Text Files**: Update the path to `speech.txt` in `RAG.py` to match your file location.

## Models
- **Streamlit App**: Uses Google’s Gemini (gemini-2.0-flash-exp) for multimodal analysis.
- **LangChain Scripts**:
  - Groq: LLaMA3-8b (`LangserveForApi.py`, `simpleSummarizerWithGroqAndLangsmit.py`), DeepSeek-Coder:7b (`mcp.py`).
  - Ollama: LLaMA3.2:1b (`RAG.py`, `simpleSummarizerWithOllamaStreamlitAndLangsmit.py`).
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) and Ollama (LLaMA3.2:1b).


## Contributing
- Fork the repository and submit pull requests.
- Follow PEP 8 style guidelines.
- Add tests for new features or scripts.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (if included).

## Acknowledgments
- [LangChain](https://www.langchain.com/) for text processing and RAG pipelines.
- [Phidata](https://www.phidata.ai/) for multimodal agent framework.
- [Streamlit](https://streamlit.io/) and [FastAPI](https://fastapi.tiangolo.com/) for web and API interfaces.
- [Groq](https://groq.com/), [Ollama](https://ollama.com/), and [Google](https://cloud.google.com/) for AI models.
- [HuggingFace](https://huggingface.co/) for embeddings.
- [Speechocean762](https://www.openslr.org/101/) dataset for inspiration (not used directly).

## Contact
For issues or suggestions, open an issue on GitHub or contact `olicodes12@gmail.com`.
