# scripts/extract_pdf_v6_7.py

from collections import defaultdict
from itertools import groupby
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

try:
    import pytesseract
    from PIL import Image
except ImportError:
    print(" WARNING: 'Pillow' or 'pytesseract' not found. OCR functionality (FR-22) will be disabled.")
    print("Please run 'pip install Pillow pytesseract' and ensure Tesseract OCR is installed on your system.")
    Image = None
    pytesseract = None


# Import our new database helper
from helpers.database_helper import calculate_file_hash, get_db_connection, initialize_database

# Import helper functions from the adjacent helpers file
from helpers.helpers_v6_7 import (  # save_to_csv, is no longer needed
    classify_paragraph_font_sizes,
    clean_text,
    extract_page_enrichments,
)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = SRC_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"  # For logs, not CSVs

# --- FSD Requirement ---
# On some systems, you may need to specify the path to the Tesseract executable.
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Ensure Tesseract is installed and in your system's PATH. [1, 3]

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Define body content area by excluding top 8% (header) and bottom 8.5% (footer)
HEADER_Y_RATIO = 0.08
FOOTER_Y_RATIO = 0.915
BODY_Y_BOUNDS = (HEADER_Y_RATIO, FOOTER_Y_RATIO)

# def ocr_page(page: fitz.Page) -> str:
#     """
#     Performs OCR on a given page if it's determined to be image-based.
#     FR-22: OCR Fallback for Image-Based PDFs.
#     """
#     print(f"    - Page {page.number + 1} appears to be image-based or has no text, attempting OCR...")
#     try:
#         pix = page.get_pixmap(dpi=300)  # Render page to an image with higher DPI
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         ocr_text = pytesseract.image_to_string(img, lang='eng')
#         print(f"    - OCR successful for page {page.number + 1}.")
#         return clean_text(ocr_text)
#     except Exception as e:
#         print(f"    -  OCR failed for page {page.number + 1}: {e}")
#         return ""


def main():
    """Processes new PDFs, checks for duplicates, and saves to the database."""
    initialize_database()
    """
    Main function to process all PDF files in the source directory,
    extracting structured data and enrichments, saving the final
    output to a single CSV file per PDF, and then moving processed files.
    """
    pdf_files = list(SRC_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f" No PDF files found in '{SRC_DIR}'. Please add PDFs to process.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    # Keep track of files that were processed successfully to move them later
    successfully_processed_paths = []

    # --- Main Processing Loop ---
    for pdf_path in pdf_files:
        doc_name = pdf_path.stem
        print(f"\n Starting extraction for: {doc_name}")

        conn = None  # Ensure conn is defined
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # The section of code that checks for duplicates needs to be updated. The original logic only skips already-processed content but doesn't handle updated files.
            # This logic is insufficient as it only prevents re-ingestion of the exact same file content (based on the hash). According to FR-24, the system must detect when a file with the same name has been updated. The new logic will check the doc_name. If it exists, it compares file hashes. If the hashes differ, it will delete all data for the old version of the document before proceeding with the new version.
            # file_hash = calculate_file_hash(pdf_path)
            # cursor.execute("SELECT doc_id FROM documents WHERE file_hash = ?", (file_hash,))
            # if cursor.fetchone():
            #     print(f"  -  Document '{doc_name}' with this content is already in the database. Skipping.")
            #     continue
            # REPLACE WITH NEW LOGIC
            file_hash = calculate_file_hash(pdf_path)
            cursor.execute(
                "SELECT doc_id, file_hash FROM documents WHERE doc_name = ?",
                (doc_name,),
            )
            existing_doc = cursor.fetchone()

            if existing_doc:
                if existing_doc["file_hash"] == file_hash:
                    print(f"  -  Document '{doc_name}' with identical content is already processed. Skipping.")
                    continue
                else:
                    print(f"  -  Found updated version of '{doc_name}'. Deleting old data as per FR-24.")
                    old_doc_id = existing_doc["doc_id"]
                    # Granularly delete old data: chunks, then sections, then the document entry
                    cursor.execute(
                        "DELETE FROM chunks WHERE section_id IN (SELECT section_id FROM sections WHERE doc_id = ?)",
                        (old_doc_id,),
                    )
                    cursor.execute("DELETE FROM sections WHERE doc_id = ?", (old_doc_id,))
                    cursor.execute("DELETE FROM documents WHERE doc_id = ?", (old_doc_id,))
                    conn.commit()
                    print(f"  -  Old data for doc_id {old_doc_id} removed. Proceeding with new version.")

            # The 'with' statement ensures the document is closed automatically after this block
            with fitz.open(pdf_path) as doc:
                # --- Pass 1: Analyze fonts and extract all text blocks ---
                print("Pass 1/3: Analyzing font structure and extracting raw blocks...")
                paragraph_sizes = classify_paragraph_font_sizes(doc, BODY_Y_BOUNDS)
                if not paragraph_sizes:
                    print(f" WARNING: Could not identify paragraph fonts for '{doc_name}'. Skipping.")
                    continue

                all_blocks = []
                for page in tqdm(doc, desc="  - Scanning pages"):
                    page_height = page.rect.height
                    # Proceed with structured extraction for pages with text
                    blocks = page.get_text("dict").get("blocks", [])
                    body_blocks = [
                        b
                        for b in blocks
                        if b.get("type") == 0
                        and (BODY_Y_BOUNDS[0] * page_height < b["bbox"][1] < BODY_Y_BOUNDS[1] * page_height)
                    ]
                    # FR-22: OCR Fallback Logic
                    if not body_blocks and Image and pytesseract:
                        print(f"\n  - INFO: Page {page.number + 1} has no text blocks. Attempting OCR as per FR-22.")
                        try:
                            pix = page.get_pixmap(dpi=300)  # Render page to an image
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            ocr_text = pytesseract.image_to_string(img)

                            if ocr_text.strip():
                                # Add the entire OCR'd text as a single "block" for this page
                                # We use the most common paragraph font size as a reasonable default

                                all_blocks.append(
                                    {
                                        "page_num": page.number + 1,
                                        "text": clean_text(ocr_text),
                                        "font_size": -1,
                                    }
                                )
                                print(f"  -  OCR successful for page {page.number + 1}.")
                            else:
                                print(f"  -  OCR for page {page.number + 1} yielded no text.")
                            continue  # Skip the normal block processing for this page
                        except Exception as ocr_error:
                            print(f"  -  OCR failed for page {page.number + 1}: {ocr_error}")
                            continue
                    # If not an OCR page, process blocks as normal
                    for b in body_blocks:
                        try:
                            block_text = clean_text(" ".join(s["text"] for l in b["lines"] for s in l["spans"]))
                            if not block_text:
                                continue
                            first_span = b["lines"][0]["spans"][0]
                            all_blocks.append(
                                {
                                    "page_num": page.number + 1,
                                    "text": block_text,
                                    "font_size": round(first_span["size"]),
                                }
                            )
                        except (IndexError, KeyError):
                            continue

                # --- Pass 2: Build section hierarchy and extract enrichments ---
                print("Pass 2/3: Building section IDs and extracting page enrichments...")

                # Part A: Build Section IDs from text blocks
                structured_rows = []
                heading_stack = defaultdict(str)
                for block in all_blocks:
                    fsize = block["font_size"]

                    # Handle OCR content
                    if fsize == -1:
                        structured_rows.append(
                            {
                                "page_num": block["page_num"],
                                "section_id": "Scanned Content (OCR)",
                                "indexed_text": block["text"],
                            }
                        )
                        continue
                    # Handle structured content

                    if fsize not in paragraph_sizes:  # It's a heading
                        heading_stack[fsize] = block["text"]
                        # Clear any smaller headings from the stack
                        for smaller_size in list(heading_stack):
                            if smaller_size < fsize:
                                heading_stack[smaller_size] = ""
                    else:  # It's a paragraph
                        ordered_headings = [
                            heading_stack[k] for k in sorted(heading_stack.keys(), reverse=True) if heading_stack[k]
                        ]
                        section_id = "|".join(ordered_headings)
                        structured_rows.append(
                            {
                                "page_num": block["page_num"],
                                "section_id": section_id or "Introduction",  # Default if no heading
                                "indexed_text": block["text"],
                            }
                        )

                # Part B: Extract Page-level Enrichments (still inside the 'with' block)
                enrichment_rows = [
                    extract_page_enrichments(page, BODY_Y_BOUNDS)
                    for page in tqdm(doc, desc="  - Extracting enrichments")
                ]

            # --- Pass 3: Combine, Group, and Save ---
            # This part is now outside the 'with fitz.open(...)' block, ensuring the file is closed.
            print("Pass 3/3: Assembling final structured data...")
            if not structured_rows:
                print(f" WARNING: No structured text could be extracted from '{doc_name}'. Skipping.")
                continue

            # Group paragraphs belonging to the same section on the same page
            keyfunc = lambda r: (r["page_num"], r["section_id"])
            structured_rows.sort(key=keyfunc)
            grouped_text_rows = []
            for (page_num, section_id), group in groupby(structured_rows, key=keyfunc):
                grouped_text_rows.append(
                    {
                        "page_num": page_num,
                        "section_id": section_id,
                        "indexed_text": "\n".join(r["indexed_text"] for r in group),
                    }
                )

            # Convert to DataFrames and merge
            df_structure = pd.DataFrame(grouped_text_rows)
            df_enrich = pd.DataFrame(enrichment_rows)
            df_final = pd.merge(df_structure, df_enrich, on="page_num", how="left")

            # Reorder columns to match the required schema and fill NaNs
            final_schema = [
                "page_num",
                "section_id",
                "indexed_text",
                "header_text",
                "footer_text",
                "hyperlink_text",
                "table_text",
            ]
            df_final = df_final[final_schema].fillna("")

            if df_final.empty:
                print(f" WARNING: No text could be extracted from '{doc_name}'. Skipping.")
                continue

            # --- Now we can save the prepared `df_final` to the database ---
            cursor.execute(
                "INSERT INTO documents (doc_name, file_hash, status) VALUES (?, ?, ?)",
                (doc_name, file_hash, "extracted"),
            )
            doc_id = cursor.lastrowid

            # sections_to_insert = []
            # for _, row in df_final.iterrows():
            #     full_text = "\n".join(filter(None, [
            #         row['indexed_text'],
            #         f"Hyperlinks: {row['hyperlink_text']}" if row['hyperlink_text'] else "",
            #         f"Tables: {row['table_text']}" if row['table_text'] else "",
            #         f"Headers: {row['header_text']}" if row['header_text'] else "",
            #         f"Footers: {row['footer_text']}" if row['footer_text'] else ""
            #     ]))
            #     sections_to_insert.append(
            #         (doc_id, row['page_num'], row['section_id'], full_text)
            #     )
            sections_to_insert = []
            for _, row in df_final.iterrows():
                sections_to_insert.append(
                    (
                        doc_id,
                        row["page_num"],
                        row["section_id"],
                        row["indexed_text"],  # This is the raw_text
                        row["header_text"],
                        row["footer_text"],
                        row["hyperlink_text"],
                        row["table_text"],
                    )
                )

            cursor.executemany(
                # "INSERT INTO sections (doc_id, page_num, section_header, raw_text) VALUES (?, ?, ?, ?)",
                """
            INSERT INTO sections (
            doc_id, page_num, section_header, raw_text,
            header_text, footer_text, hyperlink_text, table_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                sections_to_insert,
            )

            conn.commit()
            print(f"  -  Successfully saved {len(df_final)} sections for '{doc_name}' to the database.")
            successfully_processed_paths.append(pdf_path)

        except Exception as e:
            print(f" FATAL ERROR processing {doc_name}: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

        #     # Save the final merged CSV
        #     output_csv_path = OUTPUT_DIR / f"extracted_oracle_pdf_final_{doc_name}.csv"
        #     save_to_csv(df_final.to_dict('records'), str(output_csv_path))

        #     # Add the path to our list for later moving
        #     successfully_processed_paths.append(pdf_path)

        # except Exception as e:
        #     print(f" FATAL ERROR processing {doc_name}: {e}")
        #     continue

    # --- Final Step: Move all successfully processed files ---
    if successfully_processed_paths:
        print("\n Moving processed files...")
        for path in successfully_processed_paths:
            try:
                # Move the file from the source 'data' dir to the 'processed' subdir
                destination_path = PROCESSED_DIR / path.name
                path.replace(destination_path)
                print(f"  - Moved '{path.name}' to '{destination_path}'")
            except Exception as e:
                print(f"  -  FAILED to move '{path.name}': {e}")

    print("\n>> All PDF files have been processed.")


if __name__ == "__main__":
    main()
