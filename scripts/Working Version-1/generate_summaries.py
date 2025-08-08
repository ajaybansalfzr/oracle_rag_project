# scripts/generate_summaries.py

import pandas as pd
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

# --- Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

# --- Config ---
MODEL_NAME = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"
CHUNK_SIZE_TOKENS = 200  # 🔧 Reduced from 256
CSV_FOLDER = Path("./output")
DEFAULT_INPUT_PATTERN = "extracted_oracle_pdf_final_*.csv"

# Tokenizer for token-aware chunking
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# --- Prompt Templates ---
SECTION_PROMPT_TEMPLATE = """
You are a text processing utility CREATING TEXT SUMMARIZATION FOR EMBEDDING.
- Summarize the section title by focusing ONLY on the last two segments after "|" delimiter.
- Ignore all generic headers like "Oracle Fusion Cloud Financials".
- Return a 1-2 line compact description.
- Do NOT copy section names directly.
- DO NOT include boilerplate or commentary. RETURN ONLY RAW SUMMARY.

Section Path:
{section_id}
"""

INDEXED_PROMPT_TEMPLATE = """
You are a summarization tool generating context chunks for semantic indexing.
- Create a dense, 2-3 sentence summary of the paragraph below.
- DO NOT copy exact sentences.
- DO NOT repeat section or heading names.
- Focus on conveying core meaning with distinct compressed points.
- RETURN SUMMARY TEXT ONLY. NO BOILERPLATE.

Paragraph:
{indexed_text}
"""

# --- LLM Call ---
def call_llama3(prompt: str) -> str:
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        res.raise_for_status()
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"[Error: {e}]"

# --- Caching ---
summary_cache = {}

def get_cached_summary(text: str, prompt_template: str, field: str) -> str:
    key = hashlib.md5((field + text).encode()).hexdigest()
    if key in summary_cache:
        return summary_cache[key]
    prompt = prompt_template.format(**{field: text})
    result = call_llama3(prompt)
    summary_cache[key] = result
    return result

# --- Chunking ---
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS) -> list[str]:
    if not text or not text.strip():
        return []

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_sents = []
    current_chunk_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if len(sentence_tokens) > chunk_size:
            if current_chunk_sents:
                chunks.append(" ".join(current_chunk_sents))
            chunks.append(sentence)
            current_chunk_sents = []
            current_chunk_tokens = 0
            continue

        if current_chunk_tokens + len(sentence_tokens) > chunk_size and current_chunk_sents:
            chunks.append(" ".join(current_chunk_sents))
            current_chunk_sents = current_chunk_sents[-2:]  # 🔁 increase overlap to last 2 sentences
            current_chunk_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk_sents)

        current_chunk_sents.append(sentence)
        current_chunk_tokens += len(sentence_tokens)

    if current_chunk_sents:
        chunks.append(" ".join(current_chunk_sents))

    return chunks

# --- Summarizer ---
def generate_summaries(csv_path: Path):
    df = pd.read_csv(csv_path).fillna('')
    records = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_path.name}"):
        page_num = row['page_num']
        section_id_raw = row['section_id']
        section_parts = section_id_raw.split("|")
        trimmed_sec_id = " | ".join(section_parts[-2:]) if len(section_parts) >= 2 else section_id_raw
        if len(section_parts) < 2:
            print(f"⚠️ Warning: Section ID '{section_id_raw}' is short. Using as-is.")

        section_synopsis = get_cached_summary(trimmed_sec_id, SECTION_PROMPT_TEMPLATE, "section_id")

        chunks = chunk_text(row['indexed_text'])
        if not chunks:
            continue

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk) > 1200:
                print(f"⚠️ Long chunk ({len(chunk)} chars) at page {page_num}, source_row_id={i}")

            indexed_synopsis = get_cached_summary(chunk, INDEXED_PROMPT_TEMPLATE, "indexed_text")
            summary_hash = hashlib.md5((section_synopsis + indexed_synopsis).encode()).hexdigest()

            records.append({
                'source_row_id': i,
                'page_num': page_num,
                'section_id': section_id_raw,
                'chunk_id': f"{chunk_idx+1}/{len(chunks)}",
                'indexed_text': chunk,
                'section_synopsis': section_synopsis,
                'indexed_synopsis': indexed_synopsis,
                'context_type': 'chunk_summary',
                'summary_hash': summary_hash
            })

    if not records:
        print(f"⚠️ No data generated for {csv_path.name}. Check input file content.")
        return

    df_out = pd.DataFrame(records)
    out_path = csv_path.parent / f"summarized_{csv_path.name}"
    df_out.to_csv(out_path, index=False)
    print(f"✅ Saved summarized file to: {out_path}")

# --- CLI Entry ---
if __name__ == "__main__":
    files = list(CSV_FOLDER.glob(DEFAULT_INPUT_PATTERN))
    if not files:
        print(f"❌ No matching CSV files found in '{CSV_FOLDER}' with pattern '{DEFAULT_INPUT_PATTERN}'")
    else:
        for file_path in files:
            print(f"▶️ Starting summarization for {file_path.name}...")
            generate_summaries(file_path)
