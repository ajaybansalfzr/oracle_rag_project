# scripts/ui_app.py

import streamlit as st
import subprocess
import pandas as pd
from pathlib import Path
import sys
import time
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Oracle RAG Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path & System Configuration ---
python_executable = sys.executable
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_DIR / "logs"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OLLAMA_URL = "http://localhost:11434/api/generate"

for d in [DATA_DIR, OUTPUT_DIR, LOG_DIR, EMBEDDINGS_DIR]:
    d.mkdir(exist_ok=True)


# --- Helper Functions ---
# def parse_rag_output(stdout: str) -> tuple[str, str]:
#     """Parses the stdout from the RAG script to separate the answer and sources."""
#     try:
#         answer_marker = "Oracle RAG v3 Answer:\n\n"
#         sources_marker = "\n" + "=" * 60 + "\nSources:\n"
        
#         answer_start_index = stdout.find(answer_marker)
#         if answer_start_index == -1: return "Could not parse answer (missing start marker).", ""
        
#         sources_start_index = stdout.find(sources_marker)
#         if sources_start_index == -1: return "Could not parse answer (missing end marker).", ""
        
#         answer_text_start = answer_start_index + len(answer_marker)
#         answer = stdout[answer_text_start:sources_start_index].strip()
        
#         sources_text = stdout[sources_start_index + len(sources_marker):].strip()
#         return answer, sources_text
#     except Exception:
#         return "Could not parse the answer from the script output.", ""

def parse_rag_output(stdout: str) -> tuple[str, str]:
    """
    Parses the stdout from the rag_query_cli_thinking.py script.
    It looks for the 'Final Answer:' and 'Sources:' markers.
    """
    try:
        # Define the markers based on the actual output of the script
        answer_marker = "Final Answer:"
        sources_marker = "Sources:"
        
        # Find the start of the answer
        answer_start_pos = stdout.find(answer_marker)
        if answer_start_pos == -1:
            return "Could not find the answer marker in the script output.", ""
            
        # Find the start of the sources
        sources_start_pos = stdout.find(sources_marker, answer_start_pos)
        
        # Extract the answer text
        answer_text_start = answer_start_pos + len(answer_marker)
        answer = stdout[answer_text_start:sources_start_pos].strip()
        
        # Extract the sources text
        sources = ""
        if sources_start_pos != -1:
            sources_text_start = sources_start_pos + len(sources_marker)
            sources = stdout[sources_text_start:].strip().replace("=","") # Clean up trailing ====
            
        return answer, sources
    except Exception as e:
        logger.error(f"Error parsing RAG output: {e}", exc_info=True)
        return "An error occurred while parsing the script's response.", ""

def llama3_call(prompt: str, system: str) -> str:
    """Lightweight LLM call for the query condensing step."""
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": "llama3", "prompt": prompt, "system": system, "stream": False
        }, timeout=30)
        res.raise_for_status()
        return res.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        return f"[LLM Condensing Error: {e}]"

def get_standalone_question(chat_history: list) -> str:
    """Uses an LLM to condense chat history into a new, standalone question."""
    if len(chat_history) < 2: # First question is always standalone
        return chat_history[-1]['content']

    # Format the history for the LLM
    formatted_history = ""
    for msg in chat_history[:-1]: # All but the latest message
        formatted_history += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    last_user_question = chat_history[-1]['content']

    system_prompt = """You are a query rewriting expert. Your task is to rephrase a follow-up question into a self-contained, standalone question based on the provided chat history. The new question must be understandable without needing to read the previous conversation.
- If the follow-up question is already standalone, just return it as is.
- Otherwise, incorporate the necessary context from the history.
- Output ONLY the rephrased question."""

    user_prompt = f"""Chat History:
{formatted_history}
Follow-up Question: {last_user_question}

---
Standalone Question:"""

    return llama3_call(user_prompt, system_prompt)

# def run_rag_pipeline(query: str, persona: str) -> tuple[str, str]:
#     """Executes the backend RAG script and returns the answer and sources."""
#     try:
#         cmd = [python_executable, "-m", "scripts.rag_query_cli_thinking", "--query", query, "--persona", persona.lower()]
#         result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, check=True, timeout=120)
#         return parse_rag_output(result.stdout)
#     except subprocess.CalledProcessError as e:
#         error_message = f"The RAG pipeline failed.\nError: {e.stderr}"
#         return error_message, ""
#     except subprocess.TimeoutExpired:
#         return "The RAG pipeline timed out. The request took too long to process.", ""
#     except Exception as e:
#         return f"An unknown error occurred while running the RAG pipeline: {str(e)}", ""

def run_rag_pipeline(query: str, persona: str) -> tuple[str, str]:
    """Executes the backend RAG script and returns the answer and sources."""
    
    # FR-6: Map user-friendly persona names to the script's expected arguments
    persona_map = {
        "Consultant": "consultant_answer",
        "Developer": "developer_answer",
        "User": "user_answer"
    }
    persona_arg = persona_map.get(persona, "user_answer") # Default to user_answer

    try:
        cmd = [
            python_executable, "-m", "scripts.rag_query_cli_thinking",
            "--query", query,
            "--persona", persona_arg,
            "--ensemble"  # Always run in ensemble mode from the UI for best results
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, check=True, timeout=120)
        return parse_rag_output(result.stdout)
    except subprocess.CalledProcessError as e:
        error_message = f"The RAG pipeline failed.\n\n**Error Details:**\n```\n{e.stderr}\n```"
        return error_message, ""
    except subprocess.TimeoutExpired:
        return "The RAG pipeline timed out. The request took too long to process.", ""
    except Exception as e:
        return f"An unknown error occurred while running the RAG pipeline: {str(e)}", ""

# --- Streamlit UI ---
st.title("💬 Oracle RAG Chat Assistant")

# --- Sidebar for Document Management ---
with st.sidebar:
    st.header("⚙️ Document Management")
    with st.expander("Upload & Process Documents", expanded=False):
        uploaded_file = st.file_uploader("Upload an Oracle PDF", type="pdf")
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Running extraction... This may take a few minutes."):
                    file_path = DATA_DIR / uploaded_file.name
                    with open(file_path, 'wb') as f: f.write(uploaded_file.getbuffer())
                    cmd = [python_executable, "-m", "scripts.extract_pdf_v6_7"]
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                    if result.returncode == 0:
                        st.success("Extraction complete!")
                        st.code(result.stdout, language="bash")
                    else:
                        st.error("Extraction failed.")
                        st.code(result.stderr, language="bash")

        if st.button("Generate & Store Embeddings"):
            with st.spinner("Generating embeddings for all processed documents..."):
                cmd = [python_executable, "-m", "scripts.rag_embed_store"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                if result.returncode == 0:
                    st.success("Embedding store created/updated!")
                    st.code(result.stdout, language="bash")
                else:
                    st.error("Embedding generation failed.")
                    st.code(result.stderr, language="bash")
    
    # --- Persona Selection ---
    st.header("🤖 AI Persona")
    st.session_state.persona = st.selectbox(
        "Select a Persona",
        ["Consultant", "Developer", "User"],
        help="The persona influences the tone and focus of the answer."
    )

# --- Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your Oracle documents today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_prompt := st.chat_input("Ask a follow-up question or start a new topic..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Condense chat history to a standalone question
            standalone_question = get_standalone_question(st.session_state.messages)
            
            # Optional: Show the condensed question for debugging
            with st.expander("🔍 See RAG Input"):
                st.write("The following standalone question was sent to the RAG pipeline:")
                st.info(standalone_question)

            # 2. Run the full RAG pipeline with the standalone question
            answer, sources = run_rag_pipeline(standalone_question, st.session_state.persona)

            # 3. Format and display the final response
            final_response = answer
            if sources:
                final_response += f"\n\n---\n**📚 Cited Sources:**\n\n{sources}"
                
            st.markdown(final_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})