# scripts/diagnostic_part1_structure.py

import os
import fitz  # PyMuPDF
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from itertools import groupby

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADER_END_Y_RATIO = 0.08
FOOTER_START_Y_RATIO = 0.915
PARAGRAPH_FONT_COVERAGE_THRESHOLD = 0.974

def clean_text(text: str) -> str:
    text = re.sub(r"[^ -~]+", "", text)
    return re.sub(r"\s+", " ", text.replace("\u2026", "...").replace("\u2022", "-")).strip()

def classify_paragraph_fonts_by_size(doc: fitz.Document, doc_name: str):
    size_counter = Counter()
    print("Step 1: Scanning font sizes to identify paragraph sizes by character distribution...")

    for page in doc:
        page_height = page.rect.height
        blocks = [
            b for b in page.get_text("dict").get("blocks", [])
            if b.get("type") == 0 and HEADER_END_Y_RATIO * page_height < b['bbox'][1] < FOOTER_START_Y_RATIO * page_height
        ]
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span['size'])
                    size_counter[size] += len(span.get("text", ""))

    total_chars = sum(size_counter.values())
    sorted_sizes = sorted(size_counter.items(), key=lambda x: x[0])  # ✅ Sort by font size (small to large)
    paragraph_sizes = set()
    cumulative = 0

    for size, count in sorted_sizes:
        cumulative += count
        paragraph_sizes.add(size)
        if cumulative / total_chars >= PARAGRAPH_FONT_COVERAGE_THRESHOLD:
            break

    # Optional CSV debug
    debug_path = OUTPUT_DIR / f"font_size_stats_{doc_name}.csv"
    with open(debug_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Font_Size", "Char_Count", "Cumulative_Char_Share", "Is_Paragraph"])
        cum = 0
        for size, count in sorted(size_counter.items(), key=lambda x: x[0]):
            cum += count
            writer.writerow([size, count, round(cum / total_chars, 4), size in paragraph_sizes])

    return paragraph_sizes

def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in the '{DATA_DIR}' directory.")
        return

    pdf_path = pdf_files[0]
    doc_name = pdf_path.stem
    print(f"\nProcessing document: {pdf_path.name}")

    raw_rows = []
    try:
        with fitz.open(pdf_path) as doc:
            paragraph_sizes = classify_paragraph_fonts_by_size(doc, doc_name)
            heading_stack = defaultdict(str)
            all_blocks = []

            for page_num, page in enumerate(tqdm(doc, desc="Scanning Blocks"), 1):
                page_height = page.rect.height
                blocks = [
                    b for b in page.get_text("dict").get("blocks", [])
                    if b.get("type") == 0 and HEADER_END_Y_RATIO * page_height < b['bbox'][1] < FOOTER_START_Y_RATIO * page_height
                ]

                for b in blocks:
                    try:
                        spans = [s for l in b["lines"] for s in l["spans"] if s.get("text")]
                        if not spans:
                            continue
                        span = spans[0]
                        fsize = round(span['size'])
                        block_text = clean_text(" ".join(s['text'] for s in spans))
                        if not block_text:
                            continue
                        all_blocks.append({
                            "page_num": page_num,
                            "text": block_text,
                            "font_size": fsize,
                            "y_coord": b['bbox'][1]
                        })
                    except Exception:
                        continue

            print("Step 2: Assigning section hierarchy using font size only...")
            for block in tqdm(all_blocks, desc="Building Section IDs"):
                fsize = block['font_size']
                text = block['text']
                page_num = block['page_num']

                if fsize not in paragraph_sizes:
                    heading_stack[fsize] = text
                    for smaller in list(heading_stack):
                        if smaller < fsize:
                            heading_stack[smaller] = ""
                else:
                    ordered = [heading_stack[k] for k in sorted(heading_stack) if heading_stack[k]]
                    section_id = "|".join(ordered)
                    raw_rows.append({
                        "page_num": page_num,
                        "section_id": section_id,
                        "indexed_text": text
                    })

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    if not raw_rows:
        print("No data could be extracted.")
        return

    print("Step 3: Grouping rows by page and section...")
    grouped_rows = []
    keyfunc = lambda r: (r["page_num"], r["section_id"])
    raw_rows.sort(key=keyfunc)
    for (page_num, section_id), group in groupby(raw_rows, key=keyfunc):
        combined_text = "\n".join(r["indexed_text"] for r in group)
        grouped_rows.append({
            "page_num": page_num,
            "section_id": section_id,
            "indexed_text": combined_text
        })

    csv_path = OUTPUT_DIR / f"diagnostic_part1_section_id_{doc_name}.csv"
    print(f"\nStep 4: Saving {len(grouped_rows)} grouped rows to diagnostic file: {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page_num", "section_id", "indexed_text"])
        writer.writeheader()
        writer.writerows(grouped_rows)

    print("\n✅ Diagnostic Part 1 complete. Please review the CSV file in your 'output' folder.")


if __name__ == "__main__":
    main()
