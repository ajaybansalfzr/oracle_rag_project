# scripts/generate_summaries.py

## # import pandas as pd
## import hashlib
## import requests
## from pathlib import Path
## from tqdm import tqdm
## from transformers import AutoTokenizer
## # import nltk
## from nltk.tokenize import sent_tokenize
## import sqlite3
## import re # Make sure re is imported

## from helpers.database_helper import get_db_connection


## # # --- Setup ---
## # try:
## #     nltk.data.find('tokenizers/punkt')
## # except:
## #     nltk.download('punkt', quiet=True)

## # --- Config ---
## MODEL_NAME = "llama3"
## OLLAMA_URL = "http://localhost:11434/api/generate"
## # CHUNK_SIZE_TOKENS = 200  # 🔧 Reduced from 256
## CHUNK_SIZE_TOKENS = 384
## CHUNK_MIN_TOKENS = 50
## CHUNK_OVERLAP_SENTENCES = 1
## # CSV_FOLDER = Path("./output")
## # DEFAULT_INPUT_PATTERN = "extracted_oracle_pdf_final_*.csv"

## # Tokenizer for token-aware chunking
## tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
## AS PER RESPONE FROM GEMINI
import sqlite3
import requests
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import nltk

# Download the sentence tokenizer model if you haven't already
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading 'punkt' model for sentence tokenization...")
    nltk.download('punkt')

# Import database helper
from helpers.database_helper import get_db_connection

# --- Configuration (from FSD) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "output" / "project_data.db"

# LLM and Chunking Parameters
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = 'llama3'
CHUNK_SIZE_TOKENS = 384
CHUNK_MIN_TOKENS = 50
CHUNK_OVERLAP_SENTENCES = 1

# Initialize tokenizer for accurately counting tokens
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# # --- Prompt Template (Summarizes a CHUNK with the context of its header) ---
# SUMMARY_PROMPT_TEMPLATE = """
# You are a text processing utility generating a dense summary for a semantic vector index.
# Your task is to create a concise summary of the text chunk below, using its section header as context.

# **Instructions:**
# 1.  **Use the Section Header for Context:** Understand what topic the chunk belongs to.
# 2.  **Summarize the Text Chunk:** Create a dense, 2-3 sentence summary of the main text.
# 3.  **Do Not Copy:** Do not copy exact sentences. Rephrase and condense the information.
# 4.  **Be Direct:** Return ONLY the final summary paragraph. Do not include boilerplate or conversational text.

# **Section Header (for context):**
# {section_header}

# **Text Chunk to Summarize:**
# {chunk_text}

# **Dense Summary:**
# """

# # --- Prompt Templates ---
# SECTION_PROMPT_TEMPLATE = """
# You are a summarization tool generating context chunks for SEMANTIC INDEXING.
# - Summarize the section title by focusing ONLY on the last two segments after "|" delimiter.
# - Ignore all generic headers like "Oracle Fusion Cloud Financials".
# - Return a 1-2 line compact description.
# - Do NOT copy section names directly.
# - DO NOT include boilerplate or commentary. RETURN ONLY RAW SUMMARY.

# Section Path:
# {section_id}
# """

# INDEXED_PROMPT_TEMPLATE = """
# You are a summarization tool generating context chunks for semantic indexing.
# - Create a dense, 2-3 sentence summary of the paragraph below.
# - DO NOT copy exact sentences.
# - DO NOT repeat section or heading names.
# - Focus on conveying core meaning with distinct compressed points.
# - RETURN SUMMARY TEXT ONLY. NO BOILERPLATE.

# Paragraph:
# {indexed_text}
# """

# # --- LLM Call ---
# def call_llama3(prompt: str) -> str:
#     try:
#         res = requests.post(OLLAMA_URL, json={
#             "model": MODEL_NAME, "prompt": prompt, "stream": False
#         }, timeout=60)
#         res.raise_for_status()
#         return res.json().get("response", "").strip()
#     except Exception as e:
#         # Return a clear error message that can be stored in the DB if needed
#         error_msg = f"[LLM Error: {e}]"
#         print(f"\n{error_msg}") # Print for visibility in tqdm
#         return error_msg

# # --- LLM Call ---
# def call_llama3(prompt: str) -> str:
#     try:
#         res = requests.post(OLLAMA_URL, json={
#             "model": MODEL_NAME,
#             "prompt": prompt,
#             "stream": False
#         }, timeout=60)
#         res.raise_for_status()
#         return res.json().get("response", "").strip()
#     except Exception as e:
#         return f"[LLM Error: {e}]"

# # --- Caching ---
# summary_cache = {}

# def get_cached_summary(text: str, prompt_template: str, field: str) -> str:
#     key = hashlib.md5((field + text).encode()).hexdigest()
#     if key in summary_cache:
#         return summary_cache[key]
#     prompt = prompt_template.format(**{field: text})
#     result = call_llama3(prompt)
#     summary_cache[key] = result
#     return result

# --- Chunking ---
# def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS) -> list[str]:
#     if not text or not text.strip():
#         return []

#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk_sents = []
#     current_chunk_tokens = 0

#     for i, sentence in enumerate(sentences):
#         sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
#         if len(sentence_tokens) > chunk_size:
#             if current_chunk_sents:
#                 chunks.append(" ".join(current_chunk_sents))
#             chunks.append(sentence)
#             current_chunk_sents = []
#             current_chunk_tokens = 0
#             continue

#         if current_chunk_tokens + len(sentence_tokens) > chunk_size and current_chunk_sents:
#             chunks.append(" ".join(current_chunk_sents))
#             current_chunk_sents = current_chunk_sents[-2:]  #  increase overlap to last 2 sentences
#             current_chunk_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk_sents)

#         current_chunk_sents.append(sentence)
#         current_chunk_tokens += len(sentence_tokens)

#     if current_chunk_sents:
#         chunks.append(" ".join(current_chunk_sents))

#     return chunks

# def chunk_text(text: str, max_tokens: int, min_tokens: int, overlap_sentences: int, tokenizer) -> list[str]:
#     if not isinstance(text, str) or not text.strip(): return []
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     sentences = [s.strip() for s in sentences if s.strip()]
#     if not sentences: return []

#     chunks, current_chunk_sentences, current_chunk_tokens = [], [], 0
#     for sentence in sentences:
#         sentence_token_count = len(tokenizer.encode(sentence))
#         if current_chunk_tokens > 0 and (current_chunk_tokens + sentence_token_count > max_tokens):
#             if current_chunk_tokens >= min_tokens:
#                 chunks.append(" ".join(current_chunk_sentences))
#             if overlap_sentences > 0:
#                 overlap_index = max(0, len(current_chunk_sentences) - overlap_sentences)
#                 current_chunk_sentences = current_chunk_sentences[overlap_index:]
#                 current_chunk_tokens = sum(len(tokenizer.encode(s)) for s in current_chunk_sentences)
#             else:
#                 current_chunk_sentences, current_chunk_tokens = [], 0
#         current_chunk_sentences.append(sentence)
#         current_chunk_tokens += sentence_token_count
#     if current_chunk_sentences and current_chunk_tokens >= min_tokens:
#         chunks.append(" ".join(current_chunk_sentences))
#     elif chunks and current_chunk_sentences:
#         chunks[-1] += " " + " ".join(current_chunk_sentences)
#     return chunks

# # ... (keep all Config, Prompts, LLM Call, Caching logic) ...
## AS PER RESPONE FROM GEMINI
def chunk_text(text: str) -> list[str]:
    """
    FR-12: Sentence-Aware Overlapping Chunking.
    Splits text into chunks based on token limits, with sentence-based overlap.
    """
    if not text:
        return []
    
    # 1. Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # 2. Group sentences into chunks
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(TOKENIZER.encode(sentence))
        
        # If adding the next sentence exceeds the chunk size, finalize the current chunk
        if current_chunk_tokens + sentence_tokens > CHUNK_SIZE_TOKENS and current_chunk_sentences:
            chunk_text_content = " ".join(current_chunk_sentences)
            if current_chunk_tokens >= CHUNK_MIN_TOKENS:
                chunks.append(chunk_text_content)
            
            # Start a new chunk with overlap
            overlap_sentences = current_chunk_sentences[-CHUNK_OVERLAP_SENTENCES:]
            current_chunk_sentences = overlap_sentences
            current_chunk_tokens = len(TOKENIZER.encode(" ".join(overlap_sentences)))

        # Add the sentence to the current chunk
        current_chunk_sentences.append(sentence)
        current_chunk_tokens += sentence_tokens

    # 3. Add the last remaining chunk if it's valid
    if current_chunk_sentences:
        final_chunk_text = " ".join(current_chunk_sentences)
        if len(TOKENIZER.encode(final_chunk_text)) >= CHUNK_MIN_TOKENS:
            chunks.append(final_chunk_text)
        # FR-12: Append trailing sentences to the last valid chunk
        elif chunks:
             chunks[-1] += " " + final_chunk_text

    print(f"    - Chunked text into {len(chunks)} chunks.")
    return chunks

def generate_summary(text_to_summarize: str, section_header: str, is_section_summary=False) -> str:
    """
    FR-3 & FR-9: Generates a dense summary using an LLM.
    The prompt is tailored based on whether we are summarizing a chunk or a whole section.
    """
    if not text_to_summarize.strip():
        return ""

    # FR-18 (Refined Prompt Engineering) - Role-based instruction
    summary_type = "broad section" if is_section_summary else "granular chunk"
    prompt_text = f"""
As a technical indexing expert, your task is to create a dense summary of a {summary_type} from a technical document.
The goal is to capture all key entities, concepts, and technical terms to optimize for semantic search by future technical queries.
Do NOT explain what you are doing. Provide only the summary.

PARENT SECTION: "{section_header}"

TEXT TO SUMMARIZE:
"{text_to_summarize}"

DENSE SUMMARY:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        summary = json.loads(response.text).get("response", "").strip()
        return summary
    except requests.exceptions.RequestException as e:
        print(f"    -  ERROR: Could not connect to Ollama at {OLLAMA_URL}. Is it running?")
        print(f"    - Details: {e}")
        return "" # Return empty string on failure


# def main():
#     """Finds un-summarized sections, generates summaries, and updates the DB."""
#     conn = get_db_connection()
#     cursor = conn.cursor()

#     # THE CORRECTED SQL QUERY: This query finds sections from 'extracted'
#     # documents that do not yet have any corresponding chunks in the 'chunks' table.
#     cursor.execute("""
#         SELECT DISTINCT s.section_id, s.raw_text, s.section_header, d.doc_id
#         FROM sections s
#         JOIN documents d ON s.doc_id = d.doc_id
#         LEFT JOIN chunks c ON s.section_id = c.section_id
#         WHERE d.status = 'extracted' AND c.chunk_id IS NULL
#     """)
#     rows_to_process = cursor.fetchall()

#     if not rows_to_process:
#         print(" No new sections to chunk and summarize.")
#         conn.close()
#         return

#     print(f"Found {len(rows_to_process)} new sections to process.")
#     docs_to_update = set()

#     for row in tqdm(rows_to_process, desc="Chunking & Summarizing"):
#         docs_to_update.add(row['doc_id'])
#         # 1. Chunk the raw text from the section
#         # chunks_of_text = chunk_text(row['raw_text'])
#         chunks_of_text = chunk_text(
#             row['raw_text'], CHUNK_SIZE_TOKENS, CHUNK_MIN_TOKENS, CHUNK_OVERLAP_SENTENCES, tokenizer
#         )
        
#         chunks_to_insert = []
#         for text_chunk in chunks_of_text:
#             # 2. Generate a summary for each chunk using the header for context
#             prompt = SUMMARY_PROMPT_TEMPLATE.format(
#                 section_header=row['section_header'],
#                 chunk_text=text_chunk
#             )
#             summary = call_llama3(prompt)
#             chunks_to_insert.append((row['section_id'], text_chunk, summary))

#         # 3. Insert all generated chunks for this section into the database
#         if chunks_to_insert:
#             cursor.executemany(
#                 "INSERT INTO chunks (section_id, chunk_text, summary) VALUES (?, ?, ?)",
#                 chunks_to_insert
#             )

#     # 4. After all sections are processed, update the status of the parent documents
#     if docs_to_update:
#         cursor.executemany(
#             "UPDATE documents SET status = 'chunked_and_summarized' WHERE doc_id = ?",
#             [(doc_id,) for doc_id in docs_to_update]
#         )

#     conn.commit()
#     conn.close()
#     print(f"\n Successfully chunked and summarized {len(rows_to_process)} sections.")

## AS PER RESPONE FROM GEMINI
def main():
    """
    Main function to chunk and summarize unprocessed sections.
    """
    print(" Starting content processing and summarization...")
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # FR-11: Idempotent Processing - Identify work to be done.
        # Find documents that are 'extracted' but not yet 'chunked_and_summarized'.
        cursor.execute("""
            SELECT doc_id, doc_name FROM documents 
            WHERE status = 'extracted'
        """)
        docs_to_process = cursor.fetchall()

        if not docs_to_process:
            print(" No new documents with status 'extracted' to process.")
            return

        print(f"Found {len(docs_to_process)} document(s) to process.")

        for doc in docs_to_process:
            doc_id, doc_name = doc['doc_id'], doc['doc_name']
            print(f"\nProcessing document: '{doc_name}' (ID: {doc_id})")

            # Get all sections for this document that haven't been processed
            cursor.execute("""
                SELECT s.section_id, s.raw_text, s.section_header 
                FROM sections s
                LEFT JOIN chunks c ON s.section_id = c.section_id
                WHERE s.doc_id = ? AND c.chunk_id IS NULL
            """, (doc_id,))
            sections_to_process = cursor.fetchall()
            
            all_chunks_for_doc = []

            for section in tqdm(sections_to_process, desc=f"  - Processing Sections for '{doc_name}'"):
                section_id = section['section_id']
                raw_text = section['raw_text']
                section_header = section['section_header']

                # --- FR-9: Tier 2 (Section-level) Summarization ---
                print(f"\n    - Generating Tier 2 summary for section: '{section_header[:50]}...'")
                section_summary = generate_summary(raw_text, section_header, is_section_summary=True)
                cursor.execute("UPDATE sections SET section_summary = ? WHERE section_id = ?", (section_summary, section_id))
                print(f"    -  Tier 2 summary generated and saved.")

                # --- FR-12: Chunking ---
                chunks = chunk_text(raw_text)
                
                for chunk_text_content in chunks:
                    # --- FR-3 & FR-9: Tier 1 (Chunk-level) Summarization ---
                    chunk_summary = generate_summary(chunk_text_content, section_header)
                    
                    all_chunks_for_doc.append((
                        section_id,
                        chunk_text_content,
                        chunk_summary
                    ))

            # Insert all generated chunks for the document into the database
            if all_chunks_for_doc:
                cursor.executemany(
                    "INSERT INTO chunks (section_id, chunk_text, summary) VALUES (?, ?, ?)",
                    all_chunks_for_doc
                )
                print(f"\n  -  Successfully inserted {len(all_chunks_for_doc)} chunks and summaries for '{doc_name}'.")

            # Update the document status to 'chunked_and_summarized'
            cursor.execute("UPDATE documents SET status = 'chunked_and_summarized' WHERE doc_id = ?", (doc_id,))
            print(f"  -  Updated status for '{doc_name}' to 'chunked_and_summarized'.")
            conn.commit()

    except sqlite3.Error as e:
        print(f" DATABASE ERROR: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

    print("\n>> All content processing is complete.")

if __name__ == "__main__":
    main()

#         # We now have two types of summaries, let's combine them into one
#         section_synopsis = get_cached_summary(row['section_header'], SECTION_PROMPT_TEMPLATE, "section_id")
#         indexed_synopsis = get_cached_summary(row['raw_text'], INDEXED_PROMPT_TEMPLATE, "indexed_text")
#         full_summary = f"Section Summary: {section_synopsis}\nContent Summary: {indexed_synopsis}"

#         # Update the section with its new summary
#         cursor.execute(
#             "UPDATE sections SET summary = ? WHERE section_id = ?",
#             (full_summary, row['section_id'])
#         )
    
#     # Update the status of documents that were processed
#     if docs_to_update:
#         cursor.executemany(
#             "UPDATE documents SET status = 'summarized' WHERE doc_id = ?",
#             [(doc_id,) for doc_id in docs_to_update]
#         )

#     conn.commit()
#     conn.close()
#     print(f" Successfully generated and saved summaries for {len(rows_to_process)} sections.")


# # --- Summarizer ---
# def generate_summaries(csv_path: Path):
#     df = pd.read_csv(csv_path).fillna('')
#     records = []
#     for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_path.name}"):
#         page_num = row['page_num']
#         section_id_raw = row['section_id']
#         section_parts = section_id_raw.split("|")
#         trimmed_sec_id = " | ".join(section_parts[-2:]) if len(section_parts) >= 2 else section_id_raw
#         if len(section_parts) < 2:
#             print(f" Warning: Section ID '{section_id_raw}' is short. Using as-is.")

#         section_synopsis = get_cached_summary(trimmed_sec_id, SECTION_PROMPT_TEMPLATE, "section_id")

#         chunks = chunk_text(row['indexed_text'])
#         if not chunks:
#             continue

#         for chunk_idx, chunk in enumerate(chunks):
#             if len(chunk) > 1200:
#                 print(f" Long chunk ({len(chunk)} chars) at page {page_num}, source_row_id={i}")

#             indexed_synopsis = get_cached_summary(chunk, INDEXED_PROMPT_TEMPLATE, "indexed_text")
#             summary_hash = hashlib.md5((section_synopsis + indexed_synopsis).encode()).hexdigest()

#             records.append({
#                 'source_row_id': i,
#                 'page_num': page_num,
#                 'section_id': section_id_raw,
#                 'chunk_id': f"{chunk_idx+1}/{len(chunks)}",
#                 'indexed_text': chunk,
#                 'section_synopsis': section_synopsis,
#                 'indexed_synopsis': indexed_synopsis,
#                 'context_type': 'chunk_summary',
#                 'summary_hash': summary_hash
#             })

#     if not records:
#         print(f" No data generated for {csv_path.name}. Check input file content.")
#         return

#     df_out = pd.DataFrame(records)
#     out_path = csv_path.parent / f"summarized_{csv_path.name}"
#     df_out.to_csv(out_path, index=False)
#     print(f" Saved summarized file to: {out_path}")

# # --- CLI Entry ---
# if __name__ == "__main__":
#     files = list(CSV_FOLDER.glob(DEFAULT_INPUT_PATTERN))
#     if not files:
#         print(f" No matching CSV files found in '{CSV_FOLDER}' with pattern '{DEFAULT_INPUT_PATTERN}'")
#     else:
#         for file_path in files:
#             print(f" Starting summarization for {file_path.name}...")
#             generate_summaries(file_path)
