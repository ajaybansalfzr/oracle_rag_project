# scripts/extract_pdf_v6_7.py

import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import groupby
from tqdm import tqdm

# Import helper functions from the adjacent helpers file
from helpers.helpers_v6_7 import (
    save_to_csv,
    clean_text,
    classify_paragraph_font_sizes,
    extract_page_enrichments
)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
PROCESSED_DIR = SRC_DIR / "processed"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Define body content area by excluding top 8% (header) and bottom 8.5% (footer)
HEADER_Y_RATIO = 0.08
FOOTER_Y_RATIO = 0.915
BODY_Y_BOUNDS = (HEADER_Y_RATIO, FOOTER_Y_RATIO)

def main():
    """
    Main function to process all PDF files in the source directory,
    extracting structured data and enrichments, saving the final
    output to a single CSV file per PDF, and then moving processed files.
    """
    pdf_files = list(SRC_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in '{SRC_DIR}'. Please add PDFs to process.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")
    
    # Keep track of files that were processed successfully to move them later
    successfully_processed_paths = []

    # --- Main Processing Loop ---
    for pdf_path in pdf_files:
        doc_name = pdf_path.stem
        print(f"\n▶️ Starting extraction for: {doc_name}")

        try:
            # The 'with' statement ensures the document is closed automatically after this block
            with fitz.open(pdf_path) as doc:
                # --- Pass 1: Analyze fonts and extract all text blocks ---
                print("Pass 1/3: Analyzing font structure and extracting raw blocks...")
                paragraph_sizes = classify_paragraph_font_sizes(doc, BODY_Y_BOUNDS)
                if not paragraph_sizes:
                    print(f"❗️ WARNING: Could not identify paragraph fonts for '{doc_name}'. Skipping.")
                    continue

                all_blocks = []
                for page in tqdm(doc, desc="  - Scanning pages"):
                    page_height = page.rect.height
                    blocks = page.get_text("dict").get("blocks", [])
                    body_blocks = [
                        b for b in blocks if b.get("type") == 0 and
                        (BODY_Y_BOUNDS[0] * page_height < b['bbox'][1] < BODY_Y_BOUNDS[1] * page_height)
                    ]
                    for b in body_blocks:
                        try:
                            block_text = clean_text(" ".join(s['text'] for l in b["lines"] for s in l["spans"]))
                            if not block_text: continue
                            first_span = b["lines"][0]["spans"][0]
                            all_blocks.append({
                                "page_num": page.number + 1,
                                "text": block_text,
                                "font_size": round(first_span['size']),
                            })
                        except (IndexError, KeyError):
                            continue

                # --- Pass 2: Build section hierarchy and extract enrichments ---
                print("Pass 2/3: Building section IDs and extracting page enrichments...")
                
                # Part A: Build Section IDs from text blocks
                structured_rows = []
                heading_stack = defaultdict(str)
                for block in all_blocks:
                    fsize = block['font_size']
                    if fsize not in paragraph_sizes:  # It's a heading
                        heading_stack[fsize] = block['text']
                        # Clear any smaller headings from the stack
                        for smaller_size in list(heading_stack):
                            if smaller_size < fsize: heading_stack[smaller_size] = ""
                    else:  # It's a paragraph
                        ordered_headings = [
                            heading_stack[k] for k in sorted(heading_stack.keys(), reverse=True) if heading_stack[k]
                        ]
                        section_id = "|".join(ordered_headings)
                        structured_rows.append({
                            "page_num": block['page_num'],
                            "section_id": section_id or "Introduction", # Default if no heading
                            "indexed_text": block['text']
                        })

                # Part B: Extract Page-level Enrichments (still inside the 'with' block)
                enrichment_rows = [
                    extract_page_enrichments(page, BODY_Y_BOUNDS)
                    for page in tqdm(doc, desc="  - Extracting enrichments")
                ]
            
            # --- Pass 3: Combine, Group, and Save ---
            # This part is now outside the 'with fitz.open(...)' block, ensuring the file is closed.
            print("Pass 3/3: Assembling final structured data...")
            if not structured_rows:
                print(f"❗️ WARNING: No structured text could be extracted from '{doc_name}'. Skipping.")
                continue

            # Group paragraphs belonging to the same section on the same page
            keyfunc = lambda r: (r["page_num"], r["section_id"])
            structured_rows.sort(key=keyfunc)
            grouped_text_rows = []
            for (page_num, section_id), group in groupby(structured_rows, key=keyfunc):
                grouped_text_rows.append({
                    "page_num": page_num,
                    "section_id": section_id,
                    "indexed_text": "\n".join(r["indexed_text"] for r in group),
                })
            
            # Convert to DataFrames and merge
            df_structure = pd.DataFrame(grouped_text_rows)
            df_enrich = pd.DataFrame(enrichment_rows)
            df_final = pd.merge(df_structure, df_enrich, on="page_num", how="left")
            
            # Reorder columns to match the required schema and fill NaNs
            final_schema = [
                "page_num", "section_id", "indexed_text", "header_text", 
                "footer_text", "hyperlink_text", "table_text"
            ]
            df_final = df_final[final_schema].fillna("")

            # Save the final merged CSV
            output_csv_path = OUTPUT_DIR / f"extracted_oracle_pdf_final_{doc_name}.csv"
            save_to_csv(df_final.to_dict('records'), str(output_csv_path))
            
            # Add the path to our list for later moving
            successfully_processed_paths.append(pdf_path)

        except Exception as e:
            print(f"❌ FATAL ERROR processing {doc_name}: {e}")
            continue

    # --- Final Step: Move all successfully processed files ---
    if successfully_processed_paths:
        print("\n▶️ Moving processed files...")
        for path in successfully_processed_paths:
            try:
                # Move the file from the source 'data' dir to the 'processed' subdir
                destination_path = PROCESSED_DIR / path.name
                path.replace(destination_path)
                print(f"  - Moved '{path.name}' to '{destination_path}'")
            except Exception as e:
                print(f"  - ❌ FAILED to move '{path.name}': {e}")

    print("\n>> All PDF files have been processed.")


if __name__ == "__main__":
    main()