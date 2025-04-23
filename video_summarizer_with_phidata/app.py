import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.youtube_tools import YouTubeTools
from google.generativeai import upload_file, get_file
import google.generativeai as genai
from phi.storage.agent.sqlite import SqlAgentStorage
import PyPDF2

import os
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Load API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Streamlit page config
st.set_page_config(
    page_title="Multimodal AI Agent",
    layout="wide"
)

st.title("Phidata Multimodal AI Agent")

# Initialize Phidata agent
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Multimodal AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo(), YouTubeTools()],
        storage=SqlAgentStorage(table_name="agent_sessions", db_file="tmp/agent_storage.db"),
        add_history_to_messages=True,
        num_history_responses=5,
        markdown=True,
        show_tool_calls=True
    )

multimodal_Agent = initialize_agent()

# File uploader for videos, audio, and documents
uploaded_file = st.file_uploader(
    "Upload a file (video, audio, or document)", 
    type=['mp4', 'mov', 'avi', 'mp3', 'wav', 'pdf'],
    help="Supported formats: Videos (.mp4, .mov, .avi), Audio (.mp3, .wav), Documents (.pdf)"
)
# YouTube link input
youtube_link = st.text_input("Or enter a YouTube video link")

# User query
user_query = st.text_area(
    "Ask a Question",
    placeholder="Ask anything about the uploaded file or YouTube video.",
    help="Provide specific questions or instructions."
)

# Button to trigger analysis
if st.button("🔍 Analyze", key="analyze_button"):
    if not user_query and not uploaded_file and not youtube_link:
        st.warning("Please upload a file or provide a YouTube link along with a question.")
    else:
        try:
            with st.spinner("Analyzing... please wait"):
                # If YouTube link is provided
                if youtube_link:
                    yt_prompt = f"""
                    You are a helpuful YouTube agent.
                    Analyze every informations and details about this video and answer to any  questions asked in : {youtube_link}
                    Question: {user_query}
                    """
                    response = multimodal_Agent.run(yt_prompt)

                # If a file is uploaded
                elif uploaded_file:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    # Video processing
                    if file_extension in ['mp4', 'mov', 'avi']:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                            temp_file.write(uploaded_file.read())
                            file_path = temp_file.name

                        st.video(file_path, format=f"video/{file_extension}", start_time=0)

                        # Upload and process video with Gemini
                        processed_file = upload_file(file_path)
                        while processed_file.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_file = get_file(processed_file.name)

                        analysis_prompt = f"""
                        You are a helpful video analysis agent.
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}
                        Provide a detailed, user-friendly, and actionable response.
                        """
                        response = multimodal_Agent.run(analysis_prompt, videos=[processed_file])
                        
                        # Clean up temporary file
                        Path(file_path).unlink(missing_ok=True)

                    # Audio processing
                    elif file_extension in ['mp3', 'wav']:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                            temp_file.write(uploaded_file.read())
                            file_path = temp_file.name

                        # Upload and process audio with Gemini
                        processed_file = upload_file(file_path)
                        while processed_file.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_file = get_file(processed_file.name)

                        analysis_prompt = f"""
                        Analyze the uploaded audio file for content, transcription, or context.
                        Respond to the following query using audio insights and supplementary web research:
                        {user_query}
                        Provide a detailed, user-friendly, and actionable response.
                        """
                        response = multimodal_Agent.run(analysis_prompt, audios=[processed_file])
                        
                        # Clean up temporary file
                        Path(file_path).unlink(missing_ok=True)

                    # PDF document processing
                    elif file_extension == 'pdf':
                        # Extract text from PDF
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        pdf_text = ""
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text() or ""

                        if not pdf_text.strip():
                            st.warning("No text could be extracted from the PDF.")
                            response = None
                        else:
                            analysis_prompt = f"""
                            You are a helpful document analysis agent.
                            Analyze the following text extracted from a PDF document in:
                            {pdf_text[:2000]}  # Limit to avoid token overflow
                            
                            Respond to the following query using the document content and supplementary web research:
                            {user_query}
                            Provide a detailed, user-friendly, and actionable response.
                            """
                            response = multimodal_Agent.run(analysis_prompt)
                else:
                    # Fallback if only a question is asked (no file, no YouTube link)
                     general_prompt = f"""
                     You are a helpful general AI assistant.
                     Answer the following user question thoroughly:
                     {user_query} """
                     response = multimodal_Agent.run(general_prompt)


                # Display results
                if response:
                    st.subheader("Analysis Result")
                    st.markdown(response.content)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Styling
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

