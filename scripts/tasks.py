import io
import subprocess
import sys
from pathlib import Path

from celery import Celery

# --- Add project root to path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End of fix ---

from scripts.query_handler import MODELS  # <-- Add MODELS to your imports
from scripts.query_handler import query_rag_pipeline

# Configure Celery to use Redis as the message broker and result backend
celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

# @celery_app.task(bind=True)
# def get_rag_answer(self, query: str, persona: str, top_k: int, ensemble: bool):
#     """
#     This is the Celery task that will run in the background.
#     It wraps your existing pipeline function.
#     """
#     # The heavy lifting happens here, in the worker process.
#     answer, sources = query_rag_pipeline(query, persona, top_k, ensemble)
#     return {"answer": answer, "sources": sources}

# @celery_app.task(bind=True)
# def run_pipeline_task(script_name: str):
#     """
#     Executes a given pipeline script as a subprocess in the background.

#     Args:
#         script_name: The module name of the script to run (e.g., 'scripts.extractor_for_pdf').
#     """
#     # Get the project root from the current file's path to ensure correct execution context
#     project_root = Path(__file__).resolve().parents[1]
#     python_executable = sys.executable  # Use the same python as the worker

#     try:
#         # We run the script as a module from the project root directory
#         cmd = [python_executable, "-m", script_name]
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             cwd=project_root,
#             check=True  # This will raise an exception if the script returns a non-zero exit code
#         )
#         # On success, return the standard output
#         return {"status": "SUCCESS", "output": result.stdout}
#     except subprocess.CalledProcessError as e:
#         # On failure, return the standard error
#         error_output = f"--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}"
#         return {"status": "FAILURE", "output": error_output}
#     except Exception as e:
#         return {"status": "FAILURE", "output": f"An unexpected error occurred: {str(e)}"}

# @celery_app.task(bind=True)
# def reload_models_task():
#     """
#     A special task that tells a worker to clear its in-memory model cache.
#     """
#     if MODELS:
#         MODELS.clear()
#         return "SUCCESS: In-memory model cache has been cleared."
#     return "INFO: Model cache was already empty."

# --- START CORRECTION ---


@celery_app.task(bind=True)
def get_rag_answer(self, query: str, chat_history: list, persona: str, ensemble: bool):
    """
    Celery task for the RAG pipeline. The 'self' argument is now correctly included.
    """
    answer, sources = query_rag_pipeline(query, chat_history, persona, ensemble)
    return {"answer": answer, "sources": sources}


# @celery_app.task(bind=True)
# def run_pipeline_task(self, script_name: str):
#     """
#     Executes a pipeline script. The 'self' argument is now correctly included.
#     """
#     project_root = Path(__file__).resolve().parents[1]
#     python_executable = sys.executable

#     try:
#         cmd = [python_executable, "-u","-m", script_name]
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             cwd=project_root,
#             check=True
#         )
#         return {"status": "SUCCESS", "output": result.stdout}
#     except subprocess.CalledProcessError as e:
#         error_output = f"--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}"
#         return {"status": "FAILURE", "output": error_output}
#     except Exception as e:
#         return {"status": "FAILURE", "output": f"An unexpected error occurred: {str(e)}"}

# @celery_app.task(bind=True)
# def run_pipeline_task(self, script_name: str):
#     """
#     Executes a pipeline script as a subprocess and reports progress in real-time.
#     """
#     project_root = Path(__file__).resolve().parents[1]
#     python_executable = sys.executable

#     try:
#         cmd = [python_executable, "-u", "-m", script_name] # The "-u" flag is for unbuffered output

#         # Use Popen for real-time output streaming
#         process = subprocess.Popen(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=False,
#             cwd=project_root,
#             # bufsize=1 # Line-buffered
#         )

#         # Initialize state for progress reporting
#         self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Starting process...'})

#         full_stdout = ""
#         # Read stdout line by line as it is produced
#         for line in iter(process.stdout.readline, ''):
#             full_stdout += line

#             # --- Progress Parsing Logic ---
#             # We'll look for the percentage output from tqdm, which is a common pattern.
#             # Example tqdm output: "  - Scanning Pages for my_doc:  50%|█████     | 5/10 [00:01<00:01,  4.88it/s]"
#             if '%' in line:
#                 try:
#                     # Extract the percentage value
#                     percent_str = line.split('%')[0].split()[-1]
#                     percent = int(percent_str)
#                     # Extract the status message
#                     status_message = line.strip()

#                     # Send an update to the client
#                     self.update_state(
#                         state='PROGRESS',
#                         meta={'current': percent, 'total': 100, 'status': status_message}
#                     )
#                 except (ValueError, IndexError):
#                     # If parsing fails, just ignore this line for progress purposes
#                     pass

#         process.stdout.close()
#         # Wait for the process to finish and get the exit code
#         return_code = process.wait()

#         full_stderr = process.stderr.read()
#         process.stderr.close()

#         if return_code == 0:
#             return {"status": "SUCCESS", "output": full_stdout}
#         else:
#             error_output = f"--- STDOUT ---\n{full_stdout}\n--- STDERR ---\n{full_stderr}"
#             return {"status": "FAILURE", "output": error_output}

#     except Exception as e:
#         # Report failure back to the UI
#         return {"status": "FAILURE", "output": f"An unexpected error occurred: {str(e)}"}


@celery_app.task(bind=True)
def run_pipeline_task(self, script_name: str):
    """
    Executes a pipeline script as a subprocess and reports progress in real-time.
    This version uses a more robust method for streaming stdout.
    """
    project_root = Path(__file__).resolve().parents[1]
    python_executable = sys.executable

    try:
        cmd = [python_executable, "-u", "-m", script_name]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # IMPORTANT: We will handle decoding manually
            cwd=project_root,
        )

        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting process..."},
        )

        full_stdout = ""

        # --- ROBUST STREAMING AND DECODING ---
        # Use io.TextIOWrapper to handle real-time text decoding gracefully.
        # This is more reliable than process.stdout.readline() with text=True.
        with io.TextIOWrapper(process.stdout, encoding="utf-8", errors="ignore") as stdout_reader:
            for line in stdout_reader:
                full_stdout += line

                if "%" in line:
                    try:
                        percent_str = line.split("%")[0].split()[-1]
                        percent = int(percent_str)
                        status_message = line.strip().replace("\r", "")  # Clean the line

                        self.update_state(
                            state="PROGRESS",
                            meta={
                                "current": percent,
                                "total": 100,
                                "status": status_message,
                            },
                        )
                    except (ValueError, IndexError):
                        pass

        return_code = process.wait()
        full_stderr = process.stderr.read().decode("utf-8", errors="ignore")

        if return_code == 0:
            return {"status": "SUCCESS", "output": full_stdout}
        else:
            error_output = f"--- STDOUT ---\n{full_stdout}\n--- STDERR ---\n{full_stderr}"
            return {"status": "FAILURE", "output": error_output}

    except Exception as e:
        return {
            "status": "FAILURE",
            "output": f"An unexpected error occurred: {str(e)}",
        }


@celery_app.task(bind=True)
def reload_models_task(self):
    """
    Task to clear the worker's model cache. The 'self' argument is now correctly included.
    """
    # Note: Using 'self.app.control.broadcast' is a more robust way to signal all workers,
    # but for simplicity, clearing the local worker's MODELS dict is also effective
    # if you only have one worker or manage reloads carefully.
    if MODELS:
        MODELS.clear()
        return "SUCCESS: In-memory model cache has been cleared for this worker."
    return "INFO: Model cache was already empty for this worker."


# --- END CORRECTION ---
