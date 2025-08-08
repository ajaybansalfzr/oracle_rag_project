## AS PER RESPONE FROM GEMINI
import sqlite3
# import requests
# import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import nltk
from itertools import groupby
from operator import itemgetter
import sys
# At the top of the file
import concurrent.futures

# UPDATED: Import the logger and database utilities
from scripts.utils.database_utils import get_db_connection
from scripts.utils.utils import get_logger      
# Refactored Imports to use centralized config and utils
from scripts import config
# from scripts.utils.database_utils import get_db_connection
from scripts.utils.llm_utils import llama3_call
# Note: get_logger is already imported from utils, which is correct.


# Initialize logger for this script
logger = get_logger(__name__)

      
# Place this after your logger initialization
# --- Configuration for Parallel Processing ---
# You can adjust this number based on your Ollama server's capacity and your machine's resources.
# A good starting point is 4-8 concurrent requests.
MAX_CONCURRENT_LLM_CALLS = 4
tokenizer_path = config.MODEL_PATHS[config.PRIMARY_RETRIEVER_MODEL]

# Initialize tokenizer once and reuse
try:
    TOKENIZER = AutoTokenizer.from_pretrained(
    tokenizer_path, 
    local_files_only=True, 
    model_max_length=8192)  # set this to your Llama 3 context size (or less))
except Exception as e:
    logger.critical(f"Could not load the tokenizer model. This is a fatal error. Exception: {e}")
    raise

# # Download the sentence tokenizer model if you haven't already
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     # print("Downloading 'punkt' model for sentence tokenization...")
#     logger.info("Downloading 'punkt' model for sentence tokenization...")
#     nltk.download('punkt')

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Downloading 'punkt' model for sentence tokenization...")
    nltk.download('punkt', quiet=True)

# --- Core Logic ---

def generate_summary(text_to_summarize: str, section_header: str) -> str:
    """
    Generates a dense summary for a given block of text using an LLM.
    This function is now simplified as it always handles a "broad section".
    """
    if not text_to_summarize or not text_to_summarize.strip():
        logger.warning(f"Attempted to summarize empty text for section: '{section_header}'. Returning empty string.")
        return ""

    prompt_text = f"""
As a technical indexing expert, your task is to create a dense summary of a broad section from a technical document.
The goal is to capture all key entities, concepts, and technical terms to optimize for semantic search by future technical queries.
Do NOT explain what you are doing. Provide only the summary.

PARENT SECTION: "{section_header}"

TEXT TO SUMMARIZE:
"{text_to_summarize}"

DENSE SUMMARY:
"""
    system_prompt = "You are a technical indexing expert creating dense summaries for a semantic search system."
    return llama3_call(prompt_text, system_prompt)

def summarize_section_worker(job_data):
    """
    A worker function designed to be called by the ThreadPoolExecutor.
    It unpacks job data, calls the summary function, and returns the result.
    """
    full_section_text, section_header = job_data
    return generate_summary(full_section_text, section_header)


def chunk_text(text: str) -> list[str]:
    """
    Splits text into token-aware chunks with sentence overlap using config settings.
    """
    if not text:
        return []
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(TOKENIZER.encode(sentence))
        if current_chunk_tokens + sentence_tokens > config.CHUNK_SIZE_TOKENS and current_chunk_sentences:
            chunk_text_content = " ".join(current_chunk_sentences)
            if current_chunk_tokens >= config.CHUNK_MIN_TOKENS:
                chunks.append(chunk_text_content)
            # Overlap
            overlap_sentences = current_chunk_sentences[-config.CHUNK_OVERLAP_SENTENCES:]
            current_chunk_sentences = overlap_sentences
            current_chunk_tokens = len(TOKENIZER.encode(" ".join(overlap_sentences)))
        current_chunk_sentences.append(sentence)
        current_chunk_tokens += sentence_tokens
    if current_chunk_sentences:
        final_chunk_text = " ".join(current_chunk_sentences)
        if len(TOKENIZER.encode(final_chunk_text)) >= config.CHUNK_MIN_TOKENS:
            chunks.append(final_chunk_text)
        elif chunks:
            chunks[-1] += " " + final_chunk_text
    logger.debug(f"Chunked text into {len(chunks)} chunks.")
    return chunks

def main():
    """
    Main pipeline that processes documents one-by-one, performs only token-aware chunking,
    and saves these chunks directly (NO LLM SUMMARIZATION!).
    """
    logger.info("--- Starting content processing (Chunking-Only Strategy, No Summarization) ---")
    
    conn_outer = get_db_connection()
    docs_to_process = conn_outer.execute("""
        SELECT doc_id, doc_name FROM documents 
        WHERE processing_status = 'extracted' AND lifecycle_status = 'active'
    """).fetchall()
    conn_outer.close()

    if not docs_to_process:
        logger.info("No new documents with status 'extracted' to process.")
        return

    logger.info(f"Found {len(docs_to_process)} document(s) to process individually.")

    for doc in docs_to_process:
        doc_id, doc_name = doc['doc_id'], doc['doc_name']
        logger.info(f"--- Processing Document: '{doc_name}' (ID: {doc_id}) ---")
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT section_id, raw_text, section_header
                    FROM sections
                    WHERE doc_id = ? AND section_summary IS NULL
                    ORDER BY section_header, page_num;
                """, (doc_id,))
                sections_to_process = cursor.fetchall()

                if not sections_to_process:
                    logger.warning(f"Document '{doc_name}' has no unprocessed sections. Skipping to next.")
                    continue

                key_func = itemgetter('section_header')
                grouped_logical_sections = groupby(sections_to_process, key=key_func)
                all_chunks_for_doc = []

                for section_header, group in grouped_logical_sections:
                    section_parts = list(group)
                    full_section_text = "\n".join(part['raw_text'] for part in section_parts)
                    primary_section_id = section_parts[0]['section_id']

                    # --- CHUNKING ONLY ---
                    chunks = chunk_text(full_section_text)
                    if not chunks:
                        chunks = [full_section_text]

                    # Set summary as NULL/empty (since we are not summarizing)
                    for chunk_text_content in chunks:
                        all_chunks_for_doc.append((
                            primary_section_id, doc_id, chunk_text_content, None  # No summary!
                        ))

                # --- DB Writes ---
                if all_chunks_for_doc:
                    cursor.executemany(
                        "INSERT INTO chunks (section_id, doc_id, chunk_text, summary) VALUES (?, ?, ?, ?)",
                        all_chunks_for_doc
                    )
                    logger.info(f"Inserted {len(all_chunks_for_doc)} chunks for doc '{doc_name}'.")

                cursor.execute(
                    "UPDATE documents SET processing_status = 'chunked_and_summarized' WHERE doc_id = ?", (doc_id,)
                )
                logger.info(f"Status updated for '{doc_name}'.")

        except Exception as e:
            logger.error(
                f"Failed to process document '{doc_name}' (ID: {doc_id}). Transaction rolled back. Error: {e}", exc_info=True
            )
            continue
    logger.info("--- All documents have been attempted. Pipeline finished. ---")
    

if __name__ == "__main__":
    main()

# # Initialize tokenizer for accurately counting tokens
# TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# # ... (keep all Config, Prompts, LLM Call, Caching logic) ...
# ## AS PER RESPONE FROM GEMINI
# def chunk_text(text: str) -> list[str]:
#     """
#     FR-12: Sentence-Aware Overlapping Chunking.
#     Splits text into chunks based on token limits, with sentence-based overlap.
#     """
#     if not text:
#         return []
    
#     # 1. Split the text into sentences
#     sentences = nltk.sent_tokenize(text)
    
#     # 2. Group sentences into chunks
#     chunks = []
#     current_chunk_sentences = []
#     current_chunk_tokens = 0

#     for sentence in sentences:
#         sentence_tokens = len(TOKENIZER.encode(sentence))
        
#         # If adding the next sentence exceeds the chunk size, finalize the current chunk
#         if current_chunk_tokens + sentence_tokens > config.CHUNK_SIZE_TOKENS and current_chunk_sentences:
#             chunk_text_content = " ".join(current_chunk_sentences)
#             if current_chunk_tokens >= config.CHUNK_MIN_TOKENS:
#                 chunks.append(chunk_text_content)
            
#             # Start a new chunk with overlap
#             overlap_sentences = current_chunk_sentences[-config.CHUNK_OVERLAP_SENTENCES:]
#             current_chunk_sentences = overlap_sentences
#             current_chunk_tokens = len(TOKENIZER.encode(" ".join(overlap_sentences)))

#         # Add the sentence to the current chunk
#         current_chunk_sentences.append(sentence)
#         current_chunk_tokens += sentence_tokens

#     # 3. Add the last remaining chunk if it's valid
#     if current_chunk_sentences:
#         final_chunk_text = " ".join(current_chunk_sentences)
#         if len(TOKENIZER.encode(final_chunk_text)) >= config.CHUNK_MIN_TOKENS:
#             chunks.append(final_chunk_text)
#         # FR-12: Append trailing sentences to the last valid chunk
#         elif chunks:
#              chunks[-1] += " " + final_chunk_text

#     # print(f"    - Chunked text into {len(chunks)} chunks.")
#     logger.debug(f"Chunked text into {len(chunks)} chunks.")
#     return chunks

# def generate_summary(text_to_summarize: str, section_header: str, is_section_summary=False) -> str:
#     """
#     FR-3 & FR-9: Generates a dense summary using an LLM.
#     The prompt is tailored based on whether we are summarizing a chunk or a whole section.
#     """
#     if not text_to_summarize.strip():
#         return ""

#     # FR-18 (Refined Prompt Engineering) - Role-based instruction
#     summary_type = "broad section" if is_section_summary else "granular chunk"
#     prompt_text = f"""
# As a technical indexing expert, your task is to create a dense summary of a {summary_type} from a technical document.
# The goal is to capture all key entities, concepts, and technical terms to optimize for semantic search by future technical queries.
# Do NOT explain what you are doing. Provide only the summary.

# PARENT SECTION: "{section_header}"

# TEXT TO SUMMARIZE:
# "{text_to_summarize}"

# DENSE SUMMARY:
# """

#     system_prompt = "You are a technical indexing expert creating dense summaries for a semantic search system."
    
#     # CRITICAL FIX: Use the centralized llama3_call utility.
#     # It automatically uses the OLLAMA_URL and LLM_MODEL from config.py.
#     return llama3_call(prompt_text, system_prompt)

# ## AS PER RESPONE FROM GEMINI
# In scripts/process_and_summarize.py
# --- REPLACE the entire existing main() function ---

# def main():
#     """
#     Main function to chunk and summarize unprocessed sections in a single, atomic transaction.
#     This is the definitive, corrected version with proper connection and transaction scope.
#     """
#     logger.info("--- Starting content processing and summarization pipeline ---")
    
#     try:
#         # The 'with' block now correctly wraps the ENTIRE transaction.
#         # The connection will remain open until all documents are processed,
#         # and will then commit everything at once.
#         with get_db_connection() as conn:
#             cursor = conn.cursor()

#             # Identify all active documents ready for this stage.
#             cursor.execute("""
#                 SELECT doc_id, doc_name FROM documents 
#                 WHERE processing_status = 'extracted' AND lifecycle_status = 'active'
#             """)
#             docs_to_process = cursor.fetchall()

#             if not docs_to_process:
#                 logger.info("No new documents with status 'extracted' to process.")
#                 return

#             logger.info(f"Found {len(docs_to_process)} document(s) to process in a single transaction.")

#             for doc in docs_to_process:
#                 doc_id, doc_name = doc['doc_id'], doc['doc_name']
#                 logger.info(f"--> Processing document: '{doc_name}' (ID: {doc_id})")

#                 # Get ONLY the sections for this document that haven't been processed.
#                 cursor.execute("""
#                     SELECT s.section_id, s.raw_text, s.section_header 
#                     FROM sections s
#                     LEFT JOIN chunks c ON s.section_id = c.section_id
#                     WHERE s.doc_id = ? AND c.chunk_id IS NULL
#                 """, (doc_id,))
#                 sections_to_process = cursor.fetchall()
                
#                 if not sections_to_process:
#                     logger.warning(f"Document '{doc_name}' (ID: {doc_id}) has no unprocessed sections. Skipping.")
#                     continue

#                 all_chunks_for_doc = []
#                 for section in tqdm(sections_to_process, desc=f"  - Processing Sections for '{doc_name}'", leave=False):
#                     section_id, raw_text, section_header = section['section_id'], section['raw_text'], section['section_header']

#                     section_summary = generate_summary(raw_text, section_header, is_section_summary=True)
#                     cursor.execute("UPDATE sections SET section_summary = ? WHERE section_id = ?", (section_summary, section_id))

#                     chunks = chunk_text(raw_text) 
#                     for chunk_text_content in chunks:
#                         chunk_summary = generate_summary(chunk_text_content, section_header)
#                         all_chunks_for_doc.append((section_id, doc_id, chunk_text_content, chunk_summary))

#                 if all_chunks_for_doc:
#                     cursor.executemany(
#                         "INSERT INTO chunks (section_id, doc_id, chunk_text, summary) VALUES (?, ?, ?, ?)",
#                         all_chunks_for_doc
#                     )
#                     logger.debug(f"Staged {len(all_chunks_for_doc)} chunks for insertion for doc '{doc_name}'.")

#                 cursor.execute("UPDATE documents SET processing_status = 'chunked_and_summarized' WHERE doc_id = ?", (doc_id,))
#                 logger.info(f"--> Finished processing '{doc_name}'. Staged status update.")

#             logger.info(f"Successfully processed batch of {len(docs_to_process)} documents. Committing all changes.")
#             # The 'with' block handles the commit automatically upon successful exit of this block.

#     except sqlite3.Error as e:
#         logger.error(f"DATABASE ERROR during batch processing. The transaction has been rolled back. Error: {e}", exc_info=True)
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during batch processing. The transaction has been rolled back. Error: {e}", exc_info=True)

#     logger.info("--- Content processing and summarization pipeline finished ---")

# def main():
#     """
#     Main function to chunk and summarize unprocessed sections.
#     """
#     # print(" Starting content processing and summarization...")
#     logger.info("Starting content processing and summarization pipeline...")
#     # conn = get_db_connection()
#     # cursor = conn.cursor()


#     try:
#         with get_db_connection() as conn:
#             cursor = conn.cursor()
#         # FR-11: Idempotent Processing - Identify work to be done.
#         # Find documents that are 'extracted' but not yet 'chunked_and_summarized'.
#             cursor.execute("""
#                 SELECT doc_id, doc_name FROM documents 
#                 WHERE processing_status = 'extracted' AND lifecycle_status = 'active'
#             """)
#             docs_to_process = cursor.fetchall()

#             if not docs_to_process:
#                 # print(" No new documents with status 'extracted' to process.")
#                 logger.info("No new documents with status 'extracted' to process.")
#                 return

#             # print(f"Found {len(docs_to_process)} document(s) to process.")
#             logger.info(f"Found {len(docs_to_process)} document(s) to process.")

#             for doc in docs_to_process:
#                 doc_id, doc_name = doc['doc_id'], doc['doc_name']
#                 # print(f"\nProcessing document: '{doc_name}' (ID: {doc_id})")
#                 logger.info(f"Processing document: '{doc_name}' (ID: {doc_id})")

#                 # Get all sections for this document that haven't been processed
#                 cursor.execute("""
#                     SELECT s.section_id, s.raw_text, s.section_header 
#                     FROM sections s
#                     LEFT JOIN chunks c ON s.section_id = c.section_id
#                     WHERE s.doc_id = ? AND c.chunk_id IS NULL
#                 """, (doc_id,))
#                 sections_to_process = cursor.fetchall()
#                 if not sections_to_process:
#                     logger.warning(f"Document '{doc_name}' is marked 'extracted' but has no unprocessed sections. Moving to next stage.")
#                     continue
                
#                 all_chunks_for_doc = []

#                 for section in tqdm(sections_to_process, desc=f"  - Processing Sections for '{doc_name}'", leave=False):
#                     section_id, raw_text, section_header = section['section_id'], section['raw_text'], section['section_header']
#                     # section_id = section['section_id']
#                     # raw_text = section['raw_text']
#                     # section_header = section['section_header']

#                     # --- FR-9: Tier 2 (Section-level) Summarization ---
#                     # print(f"\n    - Generating Tier 2 summary for section: '{section_header[:50]}...'")
#                     logger.debug(f"Generating Tier 2 summary for section: '{section_header[:50]}...'")
                    
#                     section_summary = generate_summary(raw_text, section_header, is_section_summary=True)
#                     cursor.execute("UPDATE sections SET section_summary = ? WHERE section_id = ?", (section_summary, section_id))
#                     # print(f"    -  Tier 2 summary generated and saved.")
#                     logger.debug(f"Tier 2 summary generated and saved for section_id {section_id}.")

#                     # --- FR-12: Chunking ---
#                     chunks = chunk_text(raw_text)
                    
#                     for chunk_text_content in chunks:
#                         # --- FR-3 & FR-9: Tier 1 (Chunk-level) Summarization ---
#                         chunk_summary = generate_summary(chunk_text_content, section_header)
                        
#                         all_chunks_for_doc.append((
#                             section_id,
#                             doc_id,
#                             chunk_text_content,
#                             chunk_summary
#                         ))

#                 # Insert all generated chunks for the document into the database
#                 if all_chunks_for_doc:
#                     cursor.executemany(
#                         "INSERT INTO chunks (section_id, doc_id, chunk_text, summary) VALUES (?, ?, ?, ?)",
#                         all_chunks_for_doc
#                     )
#                     # print(f"\n  -  Successfully inserted {len(all_chunks_for_doc)} chunks and summaries for '{doc_name}'.")
#                     logger.debug(f"Staged {len(all_chunks_for_doc)} chunks for insertion for doc '{doc_name}'.")
#                     logger.info(f"Successfully inserted {len(all_chunks_for_doc)} chunks and their summaries for '{doc_name}'.")

#                 # Update the document status to 'chunked_and_summarized'
#                 cursor.execute("UPDATE documents SET processing_status = 'chunked_and_summarized' WHERE doc_id = ?", (doc_id,))
#                 # print(f"  -  Updated status for '{doc_name}' to 'chunked_and_summarized'.")
#                 logger.info(f"Updated status for document '{doc_name}' to 'chunked_and_summarized'.")
#                 # conn.commit()
#             logger.info(f"Successfully processed batch of {len(docs_to_process)} documents. Committing all changes.")
#                 # The 'with conn:' block handles the commit automatically here.

#     except sqlite3.Error as e:
#         logger.error(f"DATABASE ERROR during batch processing. The entire transaction has been rolled back. Error: {e}", exc_info=True)
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during batch processing. The entire transaction has been rolled back. Error: {e}", exc_info=True)

#         logger.info("--- Content processing and summarization pipeline finished ---")

#     except sqlite3.Error as e:
#         # print(f" DATABASE ERROR: {e}")
#         logger.error(f"DATABASE ERROR: {e}", exc_info=True)
#         if conn:
#             conn.rollback()
#     finally:
#         if conn:
#             conn.close()

#     # print("\n>> All content processing is complete.")
#     logger.info("Content processing and summarization pipeline complete.")

# def main():
#     """
#     Main pipeline that processes documents one-by-one. Each document is handled
#     in its own atomic transaction for improved resilience. Summarization of sections
#     within a document is still performed in parallel for performance.
#     """
#     logger.info("--- Starting content processing (Resilient Per-Document Strategy) ---")
    
#     # First, get a list of all documents that need processing.
#     # This connection is brief and outside the main loop.
#     conn_outer = get_db_connection()
#     docs_to_process = conn_outer.execute("""
#         SELECT doc_id, doc_name FROM documents 
#         WHERE processing_status = 'extracted' AND lifecycle_status = 'active'
#     """).fetchall()
#     conn_outer.close()

#     if not docs_to_process:
#         logger.info("No new documents with status 'extracted' to process.")
#         return

#     logger.info(f"Found {len(docs_to_process)} document(s) to process individually.")

#     # --- Main Loop: Process one document at a time ---
#     for doc in docs_to_process:
#         doc_id, doc_name = doc['doc_id'], doc['doc_name']
#         logger.info(f"--- Processing Document: '{doc_name}' (ID: {doc_id}) ---")
        
#         try:
#             # Start a new transaction for EACH document. This is the core of the change.
#             with get_db_connection() as conn:
#                 cursor = conn.cursor()

#                 # 1. Fetch sections for THIS document only.
#                 cursor.execute("""
#                     SELECT section_id, raw_text, section_header
#                     FROM sections
#                     WHERE doc_id = ? AND section_summary IS NULL
#                     ORDER BY section_header, page_num;
#                 """, (doc_id,))
#                 sections_to_process = cursor.fetchall()

#                 if not sections_to_process:
#                     logger.warning(f"Document '{doc_name}' has no unprocessed sections. Skipping to next.")
#                     continue

#                 # 2. Prepare jobs for parallel processing (same as before, but scoped to one doc).
#                 key_func = itemgetter('section_header')
#                 grouped_logical_sections = groupby(sections_to_process, key=key_func)
#                 all_chunks_for_doc = []
#                 sections_to_update = []

#                 for section_header, group in grouped_logical_sections:
#                     section_parts = list(group)
#                     full_section_text = "\n".join(part['raw_text'] for part in section_parts)
#                     primary_section_id = section_parts[0]['section_id']

#                     # --- Tier 2: Section Summary ---
#                     section_summary = generate_summary(full_section_text, section_header)
#                     for part in section_parts:
#                         sections_to_update.append((section_summary, part['section_id']))

#                     # --- Tier 1: Chunking ---  
#                     chunks = chunk_text(full_section_text)
#                     if not chunks:
#                         # Fallback: just use the section itself
#                         chunks = [full_section_text]

#                     # --- Parallel Summarization of Chunks ---
#                     chunk_summary_jobs = [(chunk, section_header) for chunk in chunks]
#                     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_LLM_CALLS) as executor:
#                         chunk_summaries = list(tqdm(
#                             executor.map(summarize_section_worker, chunk_summary_jobs),
#                             total=len(chunk_summary_jobs),
#                             desc=f"  - Summarizing chunks ({section_header[:40]}...)",
#                             ncols=100, file=sys.stdout
#                         ))

#                     # --- Prepare for DB insertion ---
#                     for chunk_text_content, chunk_summary in zip(chunks, chunk_summaries):
#                         all_chunks_for_doc.append((
#                             primary_section_id, doc_id, chunk_text_content, chunk_summary
#                         ))

#                 # --- DB Writes ---
#                 if all_chunks_for_doc:
#                     cursor.executemany(
#                         "INSERT INTO chunks (section_id, doc_id, chunk_text, summary) VALUES (?, ?, ?, ?)",
#                         all_chunks_for_doc
#                     )
#                     logger.info(f"Inserted {len(all_chunks_for_doc)} chunks for doc '{doc_name}'.")

#                 if sections_to_update:
#                     cursor.executemany(
#                         "UPDATE sections SET section_summary = ? WHERE section_id = ?",
#                         sections_to_update
#                     )
#                     logger.info(f"Updated {len(sections_to_update)} sections with summaries for doc '{doc_name}'.")

#                 cursor.execute(
#                     "UPDATE documents SET processing_status = 'chunked_and_summarized' WHERE doc_id = ?", (doc_id,)
#                 )
#                 logger.info(f"Status updated for '{doc_name}'.")

#         except Exception as e:
#             logger.error(
#                 f"Failed to process document '{doc_name}' (ID: {doc_id}). Transaction rolled back. Error: {e}", exc_info=True
#             )
#             continue

#         #         summary_jobs = []
#         #         job_metadata = []
#         #         for section_header, group in grouped_logical_sections:
#         #             section_parts = list(group)
#         #             full_section_text = "\n".join(part['raw_text'] for part in section_parts)
#         #             summary_jobs.append((full_section_text, section_header))
#         #             job_metadata.append({
#         #                 "primary_section_id": section_parts[0]['section_id'],
#         #                 "all_section_ids": [part['section_id'] for part in section_parts],
#         #                 "full_text": full_section_text
#         #             })

#         #         # 3. Execute summarization in parallel for this document's sections.
#         #         summaries = []
#         #         with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_LLM_CALLS) as executor:
#         #             results_iterator = executor.map(summarize_section_worker, summary_jobs)
#         #             summaries = list(tqdm(results_iterator, total=len(summary_jobs), desc=f"Summarizing '{doc_name}'", ncols=100, file=sys.stdout))
                
#         #         # 4. Aggregate results and stage database writes for this document.
#         #         chunks_to_insert = []
#         #         sections_to_update = []
#         #         for i, combined_summary in enumerate(summaries):
#         #             meta = job_metadata[i]
#         #             chunks_to_insert.append((meta['primary_section_id'], doc_id, meta['full_text'], combined_summary))
#         #             for sec_id in meta['all_section_ids']:
#         #                 sections_to_update.append((combined_summary, sec_id))

#         #         # 5. Perform batch writes and update the document status.
#         #         cursor.executemany("INSERT INTO chunks (section_id, doc_id, chunk_text, summary) VALUES (?, ?, ?, ?)", chunks_to_insert)
#         #         cursor.executemany("UPDATE sections SET section_summary = ? WHERE section_id = ?", sections_to_update)
#         #         cursor.execute("UPDATE documents SET processing_status = 'chunked_and_summarized' WHERE doc_id = ?", (doc_id,))

#         #         # The 'with conn:' block automatically commits the transaction for this document upon success.
#         #         logger.info(f"Successfully processed and committed document '{doc_name}'.")

#         # except Exception as e:
#         #     # If anything fails for this document, log the error and continue to the next one.
#         #     logger.error(f"Failed to process document '{doc_name}' (ID: {doc_id}). The transaction for this document has been rolled back. Error: {e}", exc_info=True)
#         #     # The loop will now proceed to the next document in docs_to_process.
#         #     continue
    
#     logger.info("--- All documents have been attempted. Pipeline finished. ---")

# def main():
#     """
#     Main function to process and summarize unprocessed sections.
#     This version intelligently groups multi-page sections before creating
#     a single, unified chunk for each logical section.
#     """
#     logger.info("--- Starting content processing (Logical Section Grouping Strategy) ---")
    
#     try:
#         with get_db_connection() as conn:
#             cursor = conn.cursor()

#             cursor.execute("""
#                 SELECT doc_id, doc_name FROM documents 
#                 WHERE processing_status = 'extracted' AND lifecycle_status = 'active'
#             """)
#             docs_to_process = cursor.fetchall()

#             if not docs_to_process:
#                 logger.info("No new documents with status 'extracted' to process.")
#                 return

#             logger.info(f"Found {len(docs_to_process)} document(s) to process.")

#             for doc in docs_to_process:
#                 doc_id, doc_name = doc['doc_id'], doc['doc_name']
#                 logger.info(f"--> Processing document: '{doc_name}' (ID: {doc_id})")

#                 # Step 1: Fetch ALL unprocessed sections, ordered for grouping.
#                 cursor.execute("""
#                     SELECT s.section_id, s.raw_text, s.section_header
#                     FROM sections s
#                     WHERE s.doc_id = ? AND s.section_summary IS NULL
#                     ORDER BY s.section_header, s.page_num
#                 """, (doc_id,))
#                 sections_to_process = cursor.fetchall()
                
#                 if not sections_to_process:
#                     logger.warning(f"Document '{doc_name}' has no unprocessed sections. Skipping.")
#                     continue

#                 # Step 2: Group the fetched rows by the 'section_header'.
#                 grouped_sections = groupby(sections_to_process, key=itemgetter('section_header'))
                
#                 chunks_to_insert = []
#                 for section_header, section_group in tqdm(grouped_sections, desc=f"  - Grouping Sections for '{doc_name}'"):
                    
#                     section_parts = list(section_group)
#                     full_section_text = "\n".join(part['raw_text'] for part in section_parts)
#                     primary_section_id = section_parts[0]['section_id']

#                     combined_summary = generate_summary(full_section_text, section_header, is_section_summary=True)
                    
#                     chunks_to_insert.append((
#                         primary_section_id, 
#                         doc_id, 
#                         full_section_text, 
#                         combined_summary
#                     ))
                    
#                     section_ids_in_group = [part['section_id'] for part in section_parts]
#                     update_data = [(combined_summary, sec_id) for sec_id in section_ids_in_group]
#                     cursor.executemany("UPDATE sections SET section_summary = ? WHERE section_id = ?", update_data)

#                 if chunks_to_insert:
#                     cursor.executemany(
#                         "INSERT INTO chunks (section_id, doc_id, chunk_text, summary) VALUES (?, ?, ?, ?)",
#                         chunks_to_insert
#                     )
                
#                 cursor.execute("UPDATE documents SET processing_status = 'chunked_and_summarized' WHERE doc_id = ?", (doc_id,))

#     except sqlite3.Error as e:
#         logger.error(f"DATABASE ERROR during processing. Transaction rolled back. Error: {e}", exc_info=True)
#     except Exception as e:
#         logger.error(f"An unexpected error occurred. Transaction rolled back. Error: {e}", exc_info=True)

#     logger.info("--- Content processing and summarization pipeline finished ---")

# Place this right before your main() function
# def main():
#     """
#     Main pipeline to process extracted document sections. It groups multi-page
#     sections, generates a unified summary, and creates a single corresponding
#     chunk in an atomic transaction.
#     """
#     logger.info("--- Starting content processing (Logical Section Grouping Strategy) ---")
    
#     try:
#         with get_db_connection() as conn:
#             cursor = conn.cursor()

#             # Refactored Query: Fetch all unprocessed sections from all active, extracted
#             # documents in one go. The sorting is key for the groupby operation.
#             cursor.execute("""
#                 SELECT
#                     s.doc_id,
#                     d.doc_name,
#                     s.section_id,
#                     s.raw_text,
#                     s.section_header
#                 FROM sections s
#                 JOIN documents d ON s.doc_id = d.doc_id
#                 WHERE
#                     d.processing_status = 'extracted' AND
#                     d.lifecycle_status = 'active' AND
#                     s.section_summary IS NULL
#                 ORDER BY d.doc_id, s.section_header, s.page_num;
#             """)
#             sections_to_process = cursor.fetchall()

#             if not sections_to_process:
#                 logger.info("No new document sections to process.")
#                 return

#             logger.info(f"Found {len(sections_to_process)} raw section parts to group and process.")

#             # Group by both doc_id and section_header to handle sections with the same name across different docs
#             key_func = itemgetter('doc_id', 'section_header')
#             grouped_logical_sections = groupby(sections_to_process, key=key_func)
            
#             chunks_to_insert = []
#             sections_to_update = []
#             processed_doc_ids = set()

#             for (doc_id, section_header), group in tqdm(grouped_logical_sections, desc="Processing logical sections", mininterval=1, ncols=100, file=sys.stdout):
                
#                 section_parts = list(group)
#                 full_section_text = "\n".join(part['raw_text'] for part in section_parts)
                
#                 # Use the section_id from the first part of the group as the primary reference
#                 primary_section_id = section_parts[0]['section_id']
                
#                 # Generate a single summary for the entire combined text
#                 combined_summary = generate_summary(full_section_text, section_header)
                
#                 # Prepare data for batch insertion/updation
#                 chunks_to_insert.append((
#                     primary_section_id,
#                     doc_id,
#                     full_section_text,
#                     combined_summary
#                 ))
                
#                 # Mark all constituent section parts as processed by adding the same summary
#                 for part in section_parts:
#                     sections_to_update.append((combined_summary, part['section_id']))

#                 processed_doc_ids.add(doc_id)

#             # --- Perform Batch Database Writes for Maximum Efficiency ---
#             if chunks_to_insert:
#                 cursor.executemany(
#                     "INSERT INTO chunks (section_id, doc_id, chunk_text, summary) VALUES (?, ?, ?, ?)",
#                     chunks_to_insert
#                 )
#                 logger.info(f"Staged {len(chunks_to_insert)} unified chunks for insertion.")
            
#             if sections_to_update:
#                 cursor.executemany(
#                     "UPDATE sections SET section_summary = ? WHERE section_id = ?",
#                     sections_to_update
#                 )
#                 logger.info(f"Staged {len(sections_to_update)} section parts for summary updates.")
            
#             if processed_doc_ids:
#                 # Convert set to list for executemany
#                 doc_ids_to_update_status = [(doc_id,) for doc_id in processed_doc_ids]
#                 cursor.executemany(
#                     "UPDATE documents SET processing_status = 'chunked_and_summarized' WHERE doc_id = ?",
#                     doc_ids_to_update_status
#                 )
#                 logger.info(f"Staged {len(processed_doc_ids)} documents for status update to 'chunked_and_summarized'.")

#             # The 'with' block will commit the transaction automatically upon success
#             logger.info("All database operations staged. Committing transaction.")

#     except sqlite3.Error as e:
#         logger.error(f"DATABASE ERROR during processing. Transaction will be rolled back. Error: {e}", exc_info=True)
#     except Exception as e:
#         logger.error(f"An unexpected error occurred. Transaction will be rolled back. Error: {e}", exc_info=True)

#     logger.info("--- Content processing and summarization pipeline finished ---")

## REPLACE YOUR ENTIRE main() FUNCTION WITH THIS ##

# def main():
#     """
#     Main pipeline that processes sections with high performance by first preparing all data,
#     then generating summaries in parallel, and finally committing all database
#     changes in a single atomic transaction.
#     """
#     logger.info("--- Starting content processing (High-Performance Parallel Strategy) ---")
    
#     try:
#         with get_db_connection() as conn:
#             cursor = conn.cursor()

#             # The initial query is already efficient. No changes needed here.
#             cursor.execute("""
#                 SELECT
#                     s.doc_id,
#                     d.doc_name,
#                     s.section_id,
#                     s.raw_text,
#                     s.section_header
#                 FROM sections s
#                 JOIN documents d ON s.doc_id = d.doc_id
#                 WHERE
#                     d.processing_status = 'extracted' AND
#                     d.lifecycle_status = 'active' AND
#                     s.section_summary IS NULL
#                 ORDER BY d.doc_id, s.section_header, s.page_num;
#             """)
#             sections_to_process = cursor.fetchall()

#             if not sections_to_process:
#                 logger.info("No new document sections to process.")
#                 return

#             logger.info(f"Found {len(sections_to_process)} raw section parts to group and process.")

#             # --- PHASE 1: PREPARE ALL JOBS. NO LLM CALLS HERE. ---
#             key_func = itemgetter('doc_id', 'section_header')
#             grouped_logical_sections = groupby(sections_to_process, key=key_func)
            
#             summary_jobs = []       # A list of tuples: (text_to_summarize, section_header)
#             job_metadata = []       # A parallel list to store IDs to link results back

#             logger.info("Phase 1: Preparing all summarization jobs...")
#             for (doc_id, section_header), group in grouped_logical_sections:
#                 section_parts = list(group)
#                 full_section_text = "\n".join(part['raw_text'] for part in section_parts)
                
#                 # Add the actual work to the jobs list
#                 summary_jobs.append((full_section_text, section_header))
                
#                 # Store the corresponding metadata to re-associate the result later
#                 job_metadata.append({
#                     "doc_id": doc_id,
#                     "primary_section_id": section_parts[0]['section_id'],
#                     "all_section_ids": [part['section_id'] for part in section_parts],
#                     "full_text": full_section_text
#                 })

#             logger.info(f"Phase 1 Complete: Prepared {len(summary_jobs)} unique sections for parallel summarization.")

#             # --- PHASE 2: EXECUTE SUMMARIZATION IN PARALLEL ---
#             summaries = []
#             logger.info(f"Phase 2: Starting parallel summarization with up to {MAX_CONCURRENT_LLM_CALLS} concurrent workers...")
#             with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_LLM_CALLS) as executor:
#                 # Use executor.map for a clean way to process jobs and get results in order
#                 results_iterator = executor.map(summarize_section_worker, summary_jobs)
                
#                 # Wrap with tqdm for a real-time progress bar of the parallel execution
#                 summaries = list(tqdm(results_iterator, total=len(summary_jobs), desc="Generating Summaries Concurrently", ncols=100, file=sys.stdout))
            
#             logger.info("Phase 2 Complete: All summaries have been generated.")

#             # --- PHASE 3: AGGREGATE RESULTS AND BATCH-WRITE TO DATABASE ---
#             logger.info("Phase 3: Aggregating results and preparing database transaction...")
#             chunks_to_insert = []
#             sections_to_update = []
#             processed_doc_ids = set()

#             for i, combined_summary in enumerate(summaries):
#                 meta = job_metadata[i]
#                 doc_id = meta['doc_id']
                
#                 # Prepare data for the 'chunks' table insertion
#                 chunks_to_insert.append((
#                     meta['primary_section_id'],
#                     doc_id,
#                     meta['full_text'],
#                     combined_summary
#                 ))
                
#                 # Prepare data to update all constituent 'sections'
#                 for sec_id in meta['all_section_ids']:
#                     sections_to_update.append((combined_summary, sec_id))

#                 processed_doc_ids.add(doc_id)

#             # Perform the highly efficient batch database writes
#             if chunks_to_insert:
#                 cursor.executemany(
#                     "INSERT INTO chunks (section_id, doc_id, chunk_text, summary) VALUES (?, ?, ?, ?)",
#                     chunks_to_insert
#                 )
#                 logger.info(f"Staged {len(chunks_to_insert)} unified chunks for insertion.")
            
#             if sections_to_update:
#                 cursor.executemany(
#                     "UPDATE sections SET section_summary = ? WHERE section_id = ?",
#                     sections_to_update
#                 )
#                 logger.info(f"Staged {len(sections_to_update)} section parts for summary updates.")
            
#             if processed_doc_ids:
#                 doc_ids_to_update_status = [(doc_id,) for doc_id in processed_doc_ids]
#                 cursor.executemany(
#                     "UPDATE documents SET processing_status = 'chunked_and_summarized' WHERE doc_id = ?",
#                     doc_ids_to_update_status
#                 )
#                 logger.info(f"Staged {len(processed_doc_ids)} documents for status update to 'chunked_and_summarized'.")

#             logger.info("Phase 3 Complete: All database operations staged. Committing transaction.")

#     except sqlite3.Error as e:
#         logger.error(f"DATABASE ERROR during processing. Transaction will be rolled back. Error: {e}", exc_info=True)
#     except Exception as e:
#         logger.error(f"An unexpected error occurred. Transaction will be rolled back. Error: {e}", exc_info=True)

#     logger.info("--- Content processing and summarization pipeline finished ---")