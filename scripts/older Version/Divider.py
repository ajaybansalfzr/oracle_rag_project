import pandas as pd
import os
import re
from pathlib import Path
from transformers import AutoTokenizer

# Initialize tokenizer (offline-friendly fallback)
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    token_based = True
except:
    tokenizer = None
    token_based = False

# === 1. Locate latest CSV ===
csv_files = sorted(Path("./output").glob("extracted_oracle_pdf_final_*.csv"), key=os.path.getmtime)
if not csv_files:
    raise FileNotFoundError("❌ No extracted_oracle_pdf_final_*.csv file found.")
file_path = csv_files[-1]
print(f"📄 Loaded: {file_path.name}")
df = pd.read_csv(file_path)

# === 2. Robust sentence splitter ===
def split_sentences(text):
    if not isinstance(text, str):
        return []
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# === 3. Robust token-aware chunker ===
def robust_chunk(text, max_tokens=400, min_tokens=50, overlap_tokens=50):
    sentences = split_sentences(text)
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence) if token_based else list(sentence)
        sentence_len = len(sentence_tokens)

        if current_len + sentence_len > max_tokens:
            if current_len >= min_tokens:
                chunks.append(" ".join(current_chunk))
                overlap = current_chunk[-1:] if overlap_tokens else []
                current_chunk = overlap.copy()
                current_len = sum(len(tokenizer.tokenize(s)) if token_based else len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                chunks.append(" ".join(current_chunk))
                current_chunk, current_len = [], 0
        else:
            current_chunk.append(sentence)
            current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Post-process to merge very small chunks
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = tokenizer.tokenize(chunk) if token_based else list(chunk)
        if final_chunks and len(chunk_tokens) < min_tokens:
            final_chunks[-1] += " " + chunk
        else:
            final_chunks.append(chunk)

    return final_chunks

# === 4. Build Table 2 with robust metadata tagging ===
table2_rows = []
for _, row in df.iterrows():
    chunks = robust_chunk(row.get("indexed_text", ""))
    for idx, chunk in enumerate(chunks):
        metadata_tag = f"[Page:{row['page_num']}][Section:{row['section_id']}][Chunk:{idx+1}]"
        table2_rows.append({
            "page_num": row["page_num"],
            "section_id": row["section_id"],
            "chunk_id": f"{row['page_num']}_{row['section_id']}_{idx+1}",
            "indexed_text": chunk.strip(),
            "indexed_text_tagged": f"{metadata_tag} {chunk.strip()}"
        })

table2 = pd.DataFrame(table2_rows)

# === 5. Table 1: Reference metadata ===
table1 = df.drop_duplicates(subset=["page_num"])[[
    "page_num", "header_text", "footer_text", "hyperlink_text", "table_text"
]]

# === 6. Save outputs ===
output_dir = Path("./output")
table1.to_csv(output_dir / "table1_unique_by_page.csv", index=False)
table2.to_csv(output_dir / "table2_chunked_indexed_text.csv", index=False)

print("✅ Final Robust Divider Execution Completed.")