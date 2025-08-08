# scripts/ui_app.py
import sys
from pathlib import Path


# --- CRITICAL FIX FOR IMPORTS ---
# This block makes the entire 'scripts' package available for absolute imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- END OF FIX ---

import streamlit as st
# import subprocess
# import pandas as pd
# import traceback
# import time
# import requests
import time

# --- Correctly use the project's modules ---
# from scripts.query_handler import query_rag_pipeline,load_resources
from scripts import config
from scripts.utils.utils import get_logger
# from scripts.tasks import get_rag_answer
from celery.result import AsyncResult
from scripts.tasks import get_rag_answer, run_pipeline_task, reload_models_task


logger = get_logger(__name__)

# --- Path & System Configuration ---
python_executable = sys.executable


# --- Page Configuration ---
st.set_page_config(
    page_title="Oracle RAG Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Best Practice: Centralized Session State Initialization ---
if 'persona' not in st.session_state:
    st.session_state.persona = "User"
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your Oracle documents today?"}]
if 'pipeline_task_id' not in st.session_state:
    st.session_state.pipeline_task_id = None
if 'pipeline_task_result' not in st.session_state:
    st.session_state.pipeline_task_result = None
if 'last_run_script' not in st.session_state:
    st.session_state.last_run_script = None
# --- End of Initialization ---


# --- Sidebar for Document Management ---
with st.sidebar:
    st.header("⚙️ Document Management")
    #  # Session state to track our background tasks
    # if 'pipeline_task_id' not in st.session_state:
    #     st.session_state.pipeline_task_id = None
    # if 'pipeline_task_result' not in st.session_state:
    #     st.session_state.pipeline_task_result = None

    with st.expander("Upload & Process Documents", expanded=True):
        uploaded_file = st.file_uploader("Upload an Oracle PDF", type="pdf")
        
        if uploaded_file:
            # Use config.DATA_DIR from the imported config module
            file_path = config.DATA_DIR / uploaded_file.name
            # --- START OF GRACEFUL ERROR HANDLING ---
            try:
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved '{uploaded_file.name}' to the data folder.")
            except PermissionError:
                st.error(
                    f"Permission Denied: Could not save the file to '{file_path}'. "
                    "Please check that the application has write permissions for the 'data' directory "
                    "and that the file is not locked by another program (like Antivirus or a PDF reader)."
                )
            except Exception as e:
                st.error(f"An unexpected error occurred while saving the file: {e}")
            # --- END OF GRACEFUL ERROR HANDLING ---
            # with open(file_path, 'wb') as f:
            #     f.write(uploaded_file.getbuffer())
            # st.success(f"Saved '{uploaded_file.name}' to the data folder.")

        st.info("Run the 3 pipeline steps in order for new documents.")

        # # --- Button Logic Refactored ---
        # if st.button("1. Extract & Structure"):
        #     with st.spinner("Submitting extraction task..."):
        #         task = run_pipeline_task.delay("scripts.extractor_for_pdf")
        #         st.session_state.pipeline_task_id = task.id
        #         st.session_state.pipeline_task_result = None # Clear old results
        #         st.success(f"✅ Extraction task submitted! (ID: {task.id})")

        # if st.button("2. Summarize & Chunk"):
        #     with st.spinner("Submitting summarization task..."):
        #         task = run_pipeline_task.delay("scripts.process_and_summarize")
        #         st.session_state.pipeline_task_id = task.id
        #         st.session_state.pipeline_task_result = None
        #         st.success(f"✅ Summarization task submitted! (ID: {task.id})")

        # if st.button("3. Generate & Store Embeddings"):
        #     with st.spinner("Submitting embedding task..."):
        #         task = run_pipeline_task.delay("scripts.create_vector_store")
        #         st.session_state.pipeline_task_id = task.id
        #         st.session_state.pipeline_task_result = None
        #         st.success(f"✅ Embedding task submitted! (ID: {task.id})")
        if st.button("1. Extract & Structure", key="extract_btn"):
            with st.spinner("Submitting task..."):
                task = run_pipeline_task.delay("scripts.extractor_for_pdf")
                st.session_state.pipeline_task_id = task.id
                st.session_state.pipeline_task_result = None
                st.session_state.last_run_script = "scripts.extractor_for_pdf"
                st.success(f"✅ Task submitted!")
                st.rerun() # UX Improvement: Auto-refresh to show "in progress"

        if st.button("2. Summarize & Chunk", key="summarize_btn"):
            with st.spinner("Submitting task..."):
                task = run_pipeline_task.delay("scripts.process_and_summarize")
                st.session_state.pipeline_task_id = task.id
                st.session_state.pipeline_task_result = None
                st.session_state.last_run_script = "scripts.process_and_summarize"
                st.success(f"✅ Task submitted!")
                st.rerun()

        if st.button("3. Generate & Store Embeddings", key="embed_btn"):
            with st.spinner("Submitting task..."):
                task = run_pipeline_task.delay("scripts.create_vector_store")
                st.session_state.pipeline_task_id = task.id
                st.session_state.pipeline_task_result = None
                st.session_state.last_run_script = "scripts.create_vector_store"
                st.success(f"✅ Task submitted!")
                st.rerun()
                
                # --- Section to Check and Display Task Results ---
    if st.session_state.pipeline_task_id:
        result = AsyncResult(st.session_state.pipeline_task_id, app=run_pipeline_task)
        # --- NEW LOGIC TO HANDLE PROGRESS ---
        if result.state == 'PROGRESS':
            # Display a progress bar and status text
            progress_meta = result.info
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Polling loop to update the progress bar
            while result.state == 'PROGRESS':
                progress_meta = result.info
                percent_complete = int((progress_meta.get('current', 0) / progress_meta.get('total', 100)) * 100)
                status_message = progress_meta.get('status', 'Processing...')
                
                progress_bar.progress(percent_complete)
                # Display the raw tqdm status message for detailed feedback
                status_text.text(f"Status: {status_message}")
                
                # Wait a short time before polling again
                time.sleep(0.5)
            
            # After the loop, the task is finished. We need to get the final result.
            st.session_state.pipeline_task_result = result.get()
            st.rerun() # Rerun the script to now display the final result.

        elif result.ready():
            if st.session_state.pipeline_task_result is None:
                 st.session_state.pipeline_task_result = result.get()
            
            st.header("Pipeline Task Result")
            result_data = st.session_state.pipeline_task_result
            if result_data and result_data.get('status') == 'SUCCESS':
                st.success("Task completed successfully!")
                st.code(result_data.get('output', ''), language="bash")
                
                if st.session_state.last_run_script == "scripts.create_vector_store":
                    with st.spinner("Broadcasting model reload signal to workers..."):
                        reload_models_task.apply_async()
                        st.success("✅ Workers have been instructed to reload data on the next query.")
                    st.session_state.last_run_script = None
            else:
                st.error("Task failed!")
                st.code(result_data.get('output', 'No output received.'), language="bash")
            
            if st.button("Clear Result"):
                st.session_state.pipeline_task_id = None
                st.session_state.pipeline_task_result = None
                st.session_state.last_run_script = None
                st.rerun()
        else:
            st.info("A pipeline task is running in the background...")
            time.sleep(1)
            st.rerun()
            # st.button("Refresh Status") # Allows manual refresh if needed

    st.header("🤖 AI Persona")
    st.selectbox(
        "Select a Persona",
        ["Consultant", "Developer", "User"],
        key='persona', # Binds this widget directly to the session state key
    )


# --- Chat Interface ---
st.title("💬 Oracle RAG Chat Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        persona_map = {"Consultant": "consultant_answer", "Developer": "developer_answer", "User": "user_answer"}
        persona_arg = persona_map.get(st.session_state.persona, "user_answer")

        task = get_rag_answer.delay(
            query=user_prompt, 
            chat_history=st.session_state.messages, # Pass the history
            persona=persona_arg, 
            # top_k=config.top_k, 
            ensemble=True
        )

        with st.spinner("Thinking... Your request is being processed. Please wait."):
            while True:
                result = AsyncResult(task.id, app=get_rag_answer)
                if result.ready():
                    if result.successful():
                        response_data = result.get()
                        answer = response_data.get('answer', 'Error: No answer found in response.')
                        sources = response_data.get('sources', [])
                        formatted_sources = "\n".join([f"- {s}" for s in sources])
                        final_response = f"{answer}\n\n---\n**📚 Cited Sources:**\n{formatted_sources}"
                    else:
                        final_response = "I'm sorry, an error occurred. Please check the Celery worker logs for details."
                    
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    break
                
                time.sleep(1)


#         # --- Section to Check and Display Task Results ---
#     if st.session_state.pipeline_task_id:
#         result = AsyncResult(st.session_state.pipeline_task_id, app=run_pipeline_task)
#         if result.ready():
#             # Task is finished, store and display the result
#             if st.session_state.pipeline_task_result is None:
#                  st.session_state.pipeline_task_result = result.get()
            
#             st.header("Pipeline Task Result")
#             result_data = st.session_state.pipeline_task_result
#             if result_data['status'] == 'SUCCESS':
#                 st.success("Task completed successfully!")
#                 st.code(result_data['output'], language="bash")
#             else:
#                 st.error("Task failed!")
#                 st.code(result_data['output'], language="bash")
            
#             if st.button("Clear Result"):
#                 st.session_state.pipeline_task_id = None
#                 st.session_state.pipeline_task_result = None
#                 st.rerun()
#         else:
#             # Task is still running
#             # st.header("Pipeline Task in Progress...")
#             st.info("A pipeline task is currently running in the background. The result will appear here when complete.")
#             st.button("Refresh Status")
#                 #  st.rerun()

#         # if st.button("1. Extract & Structure"):
#         #     with st.spinner("Running extraction..."):
#         #         cmd = [python_executable, "-m", "scripts.extractor_for_pdf"]
#         #         result = subprocess.run(cmd, capture_output=True, text=True, cwd=config.PROJECT_ROOT)
#         #         if result.returncode == 0:
#         #             st.success("Extraction complete!")
#         #             st.code(result.stdout, language="bash")
#         #         else:
#         #             st.error("Extraction failed:")
#         #             st.code(result.stderr, language="bash")

#         # if st.button("2. Summarize & Chunk"):
#         #     with st.spinner("Running summarization and chunking..."):
#         #         cmd = [python_executable, "-m", "scripts.process_and_summarize"]
#         #         result = subprocess.run(cmd, capture_output=True, text=True, cwd=config.PROJECT_ROOT)
#         #         if result.returncode == 0:
#         #             st.success("Summarization complete!")
#         #             st.code(result.stdout, language="bash")
#         #         else:
#         #             st.error("Summarization failed:")
#         #             st.code(result.stderr, language="bash")

#         # if st.button("3. Generate & Store Embeddings"):
#         #     with st.spinner("Generating embeddings..."):
#         #         cmd = [python_executable, "-m", "scripts.create_vector_store"]
#         #         result = subprocess.run(cmd, capture_output=True, text=True, cwd=config.PROJECT_ROOT)
#         #         if result.returncode == 0:
#         #             st.success("Embedding store created/updated!")
#         #             st.code(result.stdout, language="bash")
#         #         else:
#         #             st.error("Embedding generation failed:")
#         #             st.code(result.stderr, language="bash")

# #     # --- Persona Selection ---
#     st.header("🤖 AI Persona")
#     st.session_state.persona = st.selectbox(
#         "Select a Persona",
#         ["Consultant", "Developer", "User"],
#         help="The persona influences the tone and focus of the answer."
#         )

# # --- Chat Interface ---
# st.title("💬 Oracle RAG Chat Assistant")

# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your Oracle documents today?"}]

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # if user_prompt := st.chat_input("Ask a question..."):
# #     st.session_state.messages.append({"role": "user", "content": user_prompt})
# #     with st.chat_message("user"):
# #         st.markdown(user_prompt)

# #     with st.chat_message("assistant"):
# #         with st.spinner("Thinking..."):
# #             standalone_question = get_standalone_question(st.session_state.messages)
            
# #             with st.expander("🔍 See RAG Input"):
# #                 st.info(standalone_question)

# #             answer, sources = run_rag_pipeline(standalone_question, st.session_state.persona)

# #             final_response = answer
# #             if sources:
# #                 final_response += f"\n\n---\n**📚 Cited Sources:**\n{sources}"
            
# #             st.markdown(final_response)
    
# #     st.session_state.messages.append({"role": "assistant", "content": final_response})

# if user_prompt := st.chat_input("Ask a question..."):
#     st.session_state.messages.append({"role": "user", "content": user_prompt})
#     with st.chat_message("user"):
#         st.markdown(user_prompt)

#     with st.chat_message("assistant"):
#         # --- The New Asynchronous Flow ---
        
#         # 1. Get the persona argument
#         persona_map = {"Consultant": "consultant_answer", "Developer": "developer_answer", "User": "user_answer"}
#         persona_arg = persona_map.get(st.session_state.persona, "user_answer")

#         # 2. SUBMIT the job to the background worker. This is non-blocking.
#         task = get_rag_answer.delay(
#             query=user_prompt,
#             persona=persona_arg,
#             top_k=config.top_k,
#             ensemble=True
#         )

#         # 3. Show a spinner and POLL for the result.
#         with st.spinner("Thinking... Your request is being processed by a dedicated worker. Please wait."):
#             while True:
#                 result = AsyncResult(task.id, app=get_rag_answer)
#                 if result.ready():
#                     # Job is done!
#                     if result.successful():
#                         response_data = result.get()
#                         answer = response_data['answer']
#                         sources = response_data['sources']
#                         formatted_sources = "\n".join([f"- {s}" for s in sources])
                        
#                         final_response = answer
#                         if sources:
#                             final_response += f"\n\n---\n**📚 Cited Sources:**\n{formatted_sources}"
#                     else:
#                         # The task failed in the worker
#                         final_response = "I'm sorry, an error occurred while processing your request in the background. Please check the worker logs."
                    
#                     st.markdown(final_response)
#                     st.session_state.messages.append({"role": "assistant", "content": final_response})
#                     break # Exit the polling loop
                
#                 # Wait for 2 seconds before checking again
#                 time.sleep(2)

# #     st.header("🤖 AI Persona")
# #     st.session_state.persona = st.selectbox(
# #         "Select a Persona",
# #         ["Consultant", "Developer", "User"],
#         help="The persona influences the tone and focus of the answer."
#     )

# # --- Chat Interface ---

# # Initialize chat history in session state
# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your Oracle documents today?"}]

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if user_prompt := st.chat_input("Ask a follow-up question or start a new topic..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": user_prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(user_prompt)

#     # Display assistant response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             # 1. Condense chat history to a standalone question
#             standalone_question = get_standalone_question(st.session_state.messages)
            
#             # Optional: Show the condensed question for debugging
#             with st.expander("🔍 See RAG Input"):
#                 st.write("The following standalone question was sent to the RAG pipeline:")
#                 st.info(standalone_question)

#             # 2. Run the full RAG pipeline with the standalone question
#             answer, sources = run_rag_pipeline(standalone_question, st.session_state.persona)

#             # 3. Format and display the final response
#             final_response = answer
#             if sources:
#                 final_response += f"\n\n---\n**📚 Cited Sources:**\n\n{sources}"
                
#             st.markdown(final_response)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": final_response})


# # --- Add this cached function right after imports ---
# @st.cache_resource
# def initialize_rag_models():
#     """Loads all models into memory and caches them for the app's lifecycle."""
#     # Ensure directories exist before loading models that might need them
#     logger.info("Attempting to initialize and cache RAG models for the UI.")
#     for d in [config.DATA_DIR, config.OUTPUT_DIR, config.LOG_DIR, config.EMBEDDINGS_DIR]:
#         d.mkdir(exist_ok=True)
#     # load_resources()
#      # This function from query_handler now handles the actual loading.
#     load_resources()
#     logger.info("Model loading complete. Resources are now cached.")
#     return True

# # --- Call the function once at the start of the app ---
# initialize_rag_models()

# def llama3_call(prompt: str, system: str) -> str:
#     """Lightweight LLM call for the query condensing step."""
#     try:
#         res = requests.post(config.OLLAMA_URL, json={
#             "model": config.LLM_MODEL, "prompt": prompt, "system": system, "stream": False
#         }, timeout=30)
#         res.raise_for_status()
#         return res.json()["response"].strip()
#     except requests.exceptions.RequestException as e:
#         return f"[LLM Condensing Error: {e}]"

# def get_standalone_question(chat_history: list) -> str:
#     """Uses an LLM to condense chat history into a new, standalone question."""
#     if len(chat_history) <= 2: # First question is always standalone
#         return chat_history[-1]['content']

#     # Format the history for the LLM
#     # formatted_history = ""
#     # for msg in chat_history[:-1]: # All but the latest message
#     #     formatted_history += f"{msg['role'].capitalize()}: {msg['content']}\n"
#     formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[:-1]])
#     last_user_question = chat_history[-1]['content']
    
#     # last_user_question = chat_history[-1]['content']

#     system_prompt = """You are a query rewriting expert. Your task is to rephrase a follow-up question into a self-contained, standalone question based on the provided chat history. The new question must be understandable without needing to read the previous conversation.
# - If the follow-up question is already standalone, just return it as is.
# - Otherwise, incorporate the necessary context from the history.
# - Output ONLY the rephrased question."""

#     user_prompt = f"""Chat History:
# {formatted_history}
# Follow-up Question: {last_user_question}

# ---
# Standalone Question:"""

#     return llama3_call(user_prompt, system_prompt)

# --- START CORRECTION ---
# This function now has the correct logic to handle the first query.

# def get_standalone_question(chat_history: list) -> str:
#     """
#     Uses an LLM to condense chat history into a new, standalone question.
#     If it's the first question, it is used directly without modification.
#     """
#     # The history starts with one assistant message. When the user asks the first question,
#     # the length becomes 2. This is the first turn.
#     if len(chat_history) <= 2:
#         logger.info("First user query. Using it directly without rephrasing.")
#         return chat_history[-1]['content']

#     logger.info("Follow-up question detected. Rephrasing with chat history.")
#     formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[:-1]])
#     last_user_question = chat_history[-1]['content']

#     system_prompt = """You are a query rewriting expert. Your task is to rephrase a follow-up question into a self-contained, standalone question based on the provided chat history. The new question must be understandable without needing to read the previous conversation.
# - If the follow-up question is already standalone, just return it as is.
# - Otherwise, incorporate the necessary context from the history.
# - Output ONLY the rephrased question."""

#     user_prompt = f"""Chat History:
# {formatted_history}
# Follow-up Question: {last_user_question}

# ---
# Standalone Question:"""

#     return llama3_call(user_prompt, system_prompt)
# # --- END CORRECTION ---


# def run_rag_pipeline(query: str, persona: str) -> tuple[str, str]:
#     """Executes the backend RAG pipeline via direct function calls."""
    
#     persona_map = {
#         "Consultant": "consultant_answer",
#         "Developer": "developer_answer",
#         "User": "user_answer"
#     }
#     persona_arg = persona_map.get(persona, "user_answer")

#     try:
#         # Direct, efficient function call. Models are already in memory.
#         answer, sources = query_rag_pipeline(
#             query=query,
#             persona=persona_arg,
#             top_k=config.top_k, # Or make this configurable in the UI
#             ensemble=True # Always use ensemble for best UI results
#         )
        
#         # Format sources for display
#         formatted_sources = "\n".join([f"- {s}" for s in sources])
#         return answer, formatted_sources

#     except Exception as e:
#         logger.error(f"Error running RAG pipeline directly: {e}", exc_info=True)
#         # Create a detailed, user-friendly error message with a traceback
#         error_details = "".join(traceback.format_exception(type(e), e, e.__traceback__))
#         error_message = (
#             "An unexpected error occurred while processing your request. "
#             "Please check the application logs for more details.\n\n"
#             f"```\n{error_details}\n```"
#         )
#         return error_message, ""
#         # return "An unexpected error occurred while processing your request.", ""

# def parse_rag_output(stdout: str) -> tuple[str, str]:
#     """
#     Parses the stdout from the query_handler.py script.
#     It looks for the 'Final Answer:' and 'Sources:' markers.
#     """
#     try:
#         # Define the markers based on the actual output of the script
#         answer_marker = "Final Answer:"
#         sources_marker = "Sources:"
        
#         # Find the start of the answer
#         answer_start_pos = stdout.find(answer_marker)
#         if answer_start_pos == -1:
#             return "Could not find the answer marker in the script output.", ""
            
#         # Find the start of the sources
#         sources_start_pos = stdout.find(sources_marker, answer_start_pos)
        
#         # Extract the answer text
#         answer_text_start = answer_start_pos + len(answer_marker)
#         answer = stdout[answer_text_start:sources_start_pos].strip()
        
#         # Extract the sources text
#         sources = ""
#         if sources_start_pos != -1:
#             sources_text_start = sources_start_pos + len(sources_marker)
#             sources = stdout[sources_text_start:].strip().replace("=","") # Clean up trailing ====
            
#         return answer, sources
#     except Exception as e:
#         logger.error(f"Error parsing RAG output: {e}", exc_info=True)
#         return "An error occurred while parsing the script's response.", ""

# --- Streamlit UI ---
# st.title("💬 Oracle RAG Chat Assistant")

# # --- Sidebar for Document Management ---
# with st.sidebar:
#     st.header("⚙️ Document Management")
#     with st.expander("Upload & Process Documents", expanded=False):
#         uploaded_file = st.file_uploader("Upload an Oracle PDF", type="pdf")
#         if uploaded_file:
#             if st.button("Process Document"):
#                 with st.spinner("Running extraction... This may take a few minutes."):
#                     file_path = config.DATA_DIR / uploaded_file.name
#                     with open(file_path, 'wb') as f: f.write(uploaded_file.getbuffer())
#                     cmd = [config.python_executable, "-m", "scripts.extractor_for_pdf"]
#                     result = subprocess.run(cmd, capture_output=True, text=True, cwd=config.PROJECT_ROOT)
#                     if result.returncode == 0:
#                         st.success("Extraction complete!")
#                         st.code(result.stdout, language="bash")
#                     else:
#                         st.error("Extraction failed.")
#                         st.code(result.stderr, language="bash")

#         if st.button("Generate & Store Embeddings"):
#             with st.spinner("Generating embeddings for all processed documents..."):
#                 cmd = [config.python_executable, "-m", "scripts.create_vector_store"]
#                 result = subprocess.run(cmd, capture_output=True, text=True, cwd=config.PROJECT_ROOT)
#                 if result.returncode == 0:
#                     st.success("Embedding store created/updated!")
#                     st.code(result.stdout, language="bash")
#                 else:
#                     st.error("Embedding generation failed.")
#                     st.code(result.stderr, language="bash")

# # --- START OF CRITICAL FIX: Initialize all session state variables ---
# # This ensures that these values always exist, preventing AttributeError crashes.
# if 'persona' not in st.session_state:
#     st.session_state.persona = "User" # Default to User persona
# if 'messages' not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your Oracle documents today?"}]
# if 'pipeline_task_id' not in st.session_state:
#     st.session_state.pipeline_task_id = None
# if 'pipeline_task_result' not in st.session_state:
#     st.session_state.pipeline_task_result = None
# # --- END OF CRITICAL FIX ---
