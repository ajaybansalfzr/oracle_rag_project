# scripts/ui_app.py

import os
import streamlit as st
import subprocess
import pandas as pd
from pathlib import Path
import sys
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Oracle RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

python_executable = sys.executable

# --- Path Definitions ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_DIR / "logs"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for d in [DATA_DIR, OUTPUT_DIR, LOG_DIR, EMBEDDINGS_DIR]:
    d.mkdir(exist_ok=True)


# --- Helper Functions ---
def parse_rag_output(stdout: str) -> tuple[str, list[str]]:
    """Parses the stdout from the RAG script using string splitting."""
    try:
        answer_marker = "Oracle RAG v3 Answer:\n\n"
        sources_marker = "\n" + "=" * 60 + "\nSources:\n"
        answer_start_index = stdout.find(answer_marker)
        if answer_start_index == -1: return "Could not parse answer (missing start marker).", []
        sources_start_index = stdout.find(sources_marker)
        if sources_start_index == -1: return "Could not parse answer (missing end marker).", []
        answer_text_start = answer_start_index + len(answer_marker)
        answer = stdout[answer_text_start:sources_start_index].strip()
        sources_text = stdout[sources_start_index + len(sources_marker):].strip()
        sources = [line.strip() for line in sources_text.split('\n') if line.strip()]
        return answer, sources
    except Exception:
        return "Could not parse the answer from the script output.", []

def stop_process():
    """Safely stops the active process."""
    if 'active_process' in st.session_state and st.session_state.active_process:
        st.session_state.active_process.terminate()
        st.warning("Process stopped by user.")
    st.session_state.active_process = None
    st.session_state.process_type = None
    time.sleep(1)

# --- Initialize Session State ---
# Using a single dictionary for process management
if 'process_info' not in st.session_state:
    st.session_state.process_info = {
        "process": None,
        "type": None,      # e.g., 'extraction', 'embedding', 'query'
        "output": "",
        "error": ""
    }
if 'rag_answer' not in st.session_state: st.session_state.rag_answer = ""
if 'rag_sources' not in st.session_state: st.session_state.rag_sources = []

# --- UI Layout ---
st.title("📘 Oracle RAG - AI Document QA Assistant")
st.markdown("A unified interface to upload, process, and query Oracle documentation using a local LLaMA 3 model.")

# --- Sidebar for Control Panel ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.subheader("1. Ingest Document")
    uploaded_file = st.file_uploader("Upload an Oracle PDF document", type="pdf")

    if uploaded_file:
        if st.button("Process Document", disabled=(st.session_state.process_info["process"] is not None)):
            st.session_state.process_info["type"] = "extraction" # <<< Set process type
            with st.spinner("Starting extraction..."):
                file_path = DATA_DIR / uploaded_file.name
                with open(file_path, 'wb') as f: f.write(uploaded_file.getbuffer())
                cmd = [python_executable, "-m", "scripts.extract_pdf_v6_7"]
                st.session_state.process_info["process"] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=PROJECT_ROOT)
                st.rerun()

    st.subheader("2. Build Vector Store")
    if st.button("Generate & Store Embeddings", disabled=(st.session_state.process_info["process"] is not None)):
        st.session_state.process_info["type"] = "embedding" # <<< Set process type
        with st.spinner("Starting embedding generation..."):
            cmd = [python_executable, "-m", "scripts.rag_embed_store"]
            st.session_state.process_info["process"] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=PROJECT_ROOT)
            st.rerun()

    # --- Active Process Monitor ---
    if st.session_state.process_info["process"]:
        st.sidebar.subheader("⏳ Process Running...")
        process = st.session_state.process_info["process"]
        with st.sidebar:
            with st.spinner("Working..."):
                if process.poll() is not None: # Check if process has finished
                    stdout, stderr = process.communicate()
                    # <<< CORE LOGIC FIX: Handle output based on process type
                    if st.session_state.process_info["type"] == "query":
                        st.session_state.rag_answer, st.session_state.rag_sources = parse_rag_output(stdout)
                        if "Could not parse" in st.session_state.rag_answer:
                             st.session_state.process_info["error"] = stdout or stderr # Show raw output on error
                    else: # For 'extraction' or 'embedding'
                        st.session_state.process_info["output"] = stdout
                        st.session_state.process_info["error"] = stderr
                    
                    st.session_state.process_info["process"] = None # Clear the process
                    st.session_state.process_info["type"] = None
                    st.rerun()
                else: # Process still running
                    if st.button("Stop Process", type="primary"):
                        stop_process()
                        st.rerun()
                    time.sleep(2)
                    st.rerun()

    # Display results for background processes in the sidebar
    if st.session_state.process_info["output"] or st.session_state.process_info["error"]:
        st.sidebar.subheader("📋 Background Process Results")
        if st.session_state.process_info["error"]:
            st.sidebar.error("Process failed or returned errors:")
            st.sidebar.code(st.session_state.process_info["error"], language="bash")
        if st.session_state.process_info["output"]:
            st.sidebar.success("Process completed successfully!")
            st.sidebar.code(st.session_state.process_info["output"], language="bash")
        # Clear after displaying
        st.session_state.process_info["output"] = ""
        st.session_state.process_info["error"] = ""

# --- Main Panel ---
tab1, tab2, tab3 = st.tabs(["💬 Ask Oracle AI", "📊 Review Extracted Data", "🧾 View Logs"])

with tab1:
    st.header("❓ Ask a Question")
    col1, col2 = st.columns([1, 1])

    with col1:
        persona = st.selectbox("Select a Persona", ["Consultant", "Developer", "User"], help="The persona influences the tone and focus of the answer.")
        query = st.text_area("Enter your question:", height=150, placeholder="e.g., How do I configure Smart View for EPM Cloud?")
        
        if st.button("Generate Answer", type="primary", disabled=(st.session_state.process_info["process"] is not None)):
            if query:
                if not (EMBEDDINGS_DIR / "oracle_index.faiss").exists():
                    st.error("FAISS index not found. Please generate embeddings first.")
                else:
                    st.session_state.process_info["type"] = "query" # <<< Set process type
                    st.session_state.rag_answer, st.session_state.rag_sources = "", [] # Clear old answer
                    cmd = [python_executable, "-m", "scripts.rag_query_cli_thinking", "--query", query, "--persona", persona.lower()]
                    st.session_state.process_info["process"] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=PROJECT_ROOT)
                    st.rerun()
            else:
                st.warning("Please enter a question.")

    with col2:
        st.subheader("🧠 Answer from Oracle AI")
        if st.session_state.process_info["process"] and st.session_state.process_info["type"] == "query":
             st.info("Generating answer... Stop button is available in the sidebar.")
        elif st.session_state.rag_answer:
            if "Could not parse" in st.session_state.rag_answer:
                 st.error("The backend script ran, but the UI could not parse its output.")
                 st.code(st.session_state.process_info["error"], language="bash")
            else:
                st.markdown(st.session_state.rag_answer)
                st.subheader("📚 Cited Sources")
                if st.session_state.rag_sources:
                    for source in st.session_state.rag_sources:
                        st.info(source, icon="📄")
                else:
                    st.info("No sources were cited for this answer.")
        else:
             st.info("The generated answer and its sources will appear here.")

with tab2:
    st.header("📊 Review Extracted Data")
    st.info("This section displays the structured data from the most recently processed PDF.")
    latest_csv = max(OUTPUT_DIR.glob("Oracle_Data_*.csv"), key=os.path.getctime, default=None)
    if latest_csv and latest_csv.exists():
        st.success(f"Displaying data from: `{latest_csv.name}`")
        df = pd.read_csv(latest_csv)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No extracted CSV found. Please upload and process a PDF from the sidebar.")

with tab3:
    st.header("🧾 View Logs")
    st.info("Logs provide detailed, timestamped information about the document processing steps.")
    log_files = sorted(LOG_DIR.glob("*.log"), key=os.path.getmtime, reverse=True)
    if log_files:
        selected_log = st.selectbox("Choose a log file to view", log_files, format_func=lambda p: p.name)
        if selected_log:
            with open(selected_log, "r", encoding="utf-8") as f:
                log_content = f.read()
            st.text_area("Log Contents", log_content, height=400, key=f"log_{selected_log.name}")
    else:
        st.warning("No log files available yet. Process a document to generate logs.")