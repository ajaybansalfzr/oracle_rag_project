# scripts/extractor_for_pdf.py

import fitz  # PyMuPDF
from pathlib import Path
from collections import defaultdict, Counter
from itertools import groupby
from tqdm import tqdm
import sqlite3
import sys # Add this import at the top of the file

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

# Refactored Imports (v1.9) to use centralized config and utils
from scripts import config
from scripts.utils.database_utils import initialize_database, get_db_connection, calculate_file_hash
from scripts.utils.vector_store_utils import remove_document_vectors
from scripts.utils.utils import (
    get_logger,
    clean_text,
    classify_paragraph_font_sizes,
    extract_page_enrichments
)

logger = get_logger(__name__)

if Image is None or pytesseract is None:
    logger.warning("Pillow or Pytesseract not found. OCR functionality (FR-22) will be disabled.")
    # On some systems, you may need to specify the path to the Tesseract executable.
    # For example, on Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_logical_breadcrumb(heading_stack):
    """
    Builds a breadcrumb string like 'Chapter > Section > Subsection' from the current heading stack.
    """
    return " > ".join([v for k, v in sorted(heading_stack.items(), reverse=True) if v])

def detect_procedure(text):
    """
    Returns the procedure title if the text indicates the start of a procedure.
    Returns None if not a procedure step.
    """
    if text.lower().startswith(("procedure", "steps", "to ", "how to")):
        return text
    return None

def extract_image_summary(block):
    """
    Returns a basic summary or caption for an image block, if found.
    """
    # You may want to expand this logic for more robust captioning.
    # For now, just use the block's 'caption' field, or mark as 'Image found'.
    return block.get('caption', 'Image found') if isinstance(block, dict) else 'Image found'

def format_table_as_markdown(table_block):
    """
    Attempts to extract and format table data as Markdown/CSV.
    """
    try:
        rows = table_block.get('lines', [])
        table_rows = []
        for row in rows:
            cells = [span.get('text', '') for span in row.get('spans', [])]
            table_rows.append(" | ".join(cells))
        return "\n".join(table_rows)
    except Exception:
        return "Table found (could not extract in detail)."


def _get_ocr_text_from_page(page: fitz.Page) -> str:
    """
    Performs OCR on a page if it's image-based and returns cleaned text.
    Returns an empty string if OCR is not possible or fails.
    """
    if not (Image and pytesseract):
        return ""

    try:
        logger.info(f"Page {page.number + 1} appears to be image-based. Attempting OCR (FR-22).")
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_text = pytesseract.image_to_string(img)
        
        if ocr_text.strip():
            logger.info(f"OCR successful for page {page.number + 1}.")
            return clean_text(ocr_text)
        else:
            logger.warning(f"OCR for page {page.number + 1} produced no text.")
            return ""
    except Exception as ocr_error:
        logger.error(f"OCR process failed for page {page.number + 1}: {ocr_error}", exc_info=True)
        return ""

# def extract_semantic_chunks(pdf_document):
#     chunks = []
#     current_path = []
#     current_chunk = {
#         "text": "",
#         "section_title": None,
#         "procedure_title": None,
#         "logical_path": "",
#         "page_number": None,
#         "table_summary": None,
#         "image_summary": None
#     }
#     for page_num, page in enumerate(pdf_document):
#         blocks = page.get_text("blocks")
#         for block in blocks:
#             text = block[4].strip()
#             if is_heading(block):  # Define using font size, bold, etc.
#                 if current_chunk["text"]:
#                     chunks.append(current_chunk.copy())
#                 current_path = update_path(current_path, text)
#                 current_chunk = {
#                     "text": "",
#                     "section_title": text,
#                     "procedure_title": None,
#                     "logical_path": " > ".join(current_path),
#                     "page_number": page_num + 1,
#                     "table_summary": None,
#                     "image_summary": None
#                 }
#             elif is_procedure_start(text):
#                 if current_chunk["text"]:
#                     chunks.append(current_chunk.copy())
#                 current_chunk["procedure_title"] = text
#                 current_chunk["text"] = ""
#             elif is_table(block):
#                 current_chunk["table_summary"] = summarize_table(block)
#             elif is_image(block):
#                 current_chunk["image_summary"] = extract_image_caption(block)
#             else:
#                 current_chunk["text"] += text + "\n"
#         # At page end
#         if current_chunk["text"]:
#             chunks.append(current_chunk.copy())
#     return chunks

def evaluate_against_golden(golden_path, conn):
    """
    Compares extracted sections to golden answer file (CSV with columns: question, expected_answer).
    Logs hit/miss for each question and writes the results to golden_eval_results.csv.
    """
    import csv
    logger.info(f"Running golden question evaluation: {golden_path}")
    results = []
    with open(golden_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question, expected = row['question'], row['expected_answer']
            found = False
            for r in conn.execute(
                "SELECT page_num, section_header, raw_text FROM sections WHERE raw_text LIKE ?", 
                (f"%{expected}%",)
            ):
                logger.info(
                    f"GOLDEN HIT: Question: '{question}' found in Section '{r['section_header']}' on page {r['page_num']}"
                )
                results.append({
                    'question': question,
                    'expected_answer': expected,
                    'result': 'HIT',
                    'section': r['section_header'],
                    'page': r['page_num']
                })
                found = True
                break
            if not found:
                logger.warning(f"GOLDEN MISS: Question: '{question}' NOT found in any section.")
                results.append({
                    'question': question,
                    'expected_answer': expected,
                    'result': 'MISS',
                    'section': '',
                    'page': ''
                })
    # Write results to CSV file
    out_path = 'golden_eval_results.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Golden evaluation results written to {out_path}.")

# At the bottom of your script (or inside main), keep this:
if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
    evaluate_against_golden(sys.argv[1], get_db_connection())

# def evaluate_against_golden(golden_path, conn):
#     """
#     Compares extracted sections to golden answer file (CSV with columns: question, expected_answer).
#     Logs hit/miss for each question.
#     """
#     import csv
#     logger.info(f"Running golden question evaluation: {golden_path}")
#     with open(golden_path, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             question, expected = row['question'], row['expected_answer']
#             # For each golden question, try to find a section containing the expected answer.
#             found = False
#             for r in conn.execute("SELECT page_num, section_header, raw_text FROM sections WHERE raw_text LIKE ?", (f"%{expected}%",)):
#                 logger.info(f"GOLDEN HIT: Question: '{question}' found in Section '{r['section_header']}' on page {r['page_num']}")
#                 found = True
#                 break
#             if not found:
#                 logger.warning(f"GOLDEN MISS: Question: '{question}' NOT found in any section.")

# if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
#     evaluate_against_golden(sys.argv[1], get_db_connection())

def main():
    """Processes new PDFs, handles updates via FR-24, and saves structured data to the DB."""
    logger.info("--- Starting PDF Extraction Process ---")
    initialize_database()
    
    # Use paths from config file
    config.DATA_DIR.mkdir(exist_ok=True)
    config.PROCESSED_DIR.mkdir(exist_ok=True)

    pdf_files = list(config.DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in '{config.DATA_DIR}'. Please add PDFs to process.")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) to process.")
    successfully_processed_paths = []

    for pdf_path in pdf_files:
        doc_name = pdf_path.stem
        logger.info(f"--- Processing: {doc_name} ---")

        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            file_hash = calculate_file_hash(pdf_path)

            # FR-24: Granular document update logic
            cursor.execute("""
                SELECT doc_id, file_hash FROM documents
                WHERE doc_name = ? AND lifecycle_status = 'active'
            """, (doc_name,))
            active_doc = cursor.fetchone()

            if active_doc:
                if active_doc['file_hash'] == file_hash:
                    logger.info(f"Document '{doc_name}' with identical content is already active and processed. Skipping.")
                    successfully_processed_paths.append(pdf_path)
                    continue
                else:
                    # Implement the "soft-delete" archival strategy.
                    old_doc_id = active_doc['doc_id']
                    logger.warning(
                        f"Found updated version of '{doc_name}'. "
                        f"Archiving old version (doc_id: {old_doc_id}) and processing new one."
                    )
                    # Instead of deleting, we update its status, preserving all historical data and vectors.
                    cursor.execute("UPDATE documents SET lifecycle_status = 'archived' WHERE doc_id = ?", (old_doc_id,))
                    # conn.commit()
                    logger.info(f"Old document version (doc_id: {old_doc_id}) successfully archived.")
                    # NOTE: We DO NOT call remove_document_vectors. The query logic will be updated
                    # separately to only search against documents where lifecycle_status = 'active'.

            # cursor.execute("SELECT doc_id, file_hash FROM documents WHERE doc_name = ?", (doc_name,))
            # existing_doc = cursor.fetchone()

            # if existing_doc:
            #     if existing_doc['file_hash'] == file_hash:
            #         logger.info(f"Document '{doc_name}' with identical content is already processed. Skipping.")
            #         successfully_processed_paths.append(pdf_path)
            #         continue
            #     else:
            #         logger.warning(f"Found updated version of '{doc_name}'. Deleting old data as per FR-24.")
            #         old_doc_id = existing_doc['doc_id']

                    
                    # # Step 1: Delete vectors from FAISS/BM25 indexes first
                    # remove_document_vectors(old_doc_id)
                    
                    # # Step 2: Delete from the database (ON DELETE CASCADE handles children)
                    # cursor.execute("DELETE FROM documents WHERE doc_id = ?", (old_doc_id,))
                    # conn.commit()
                    # logger.info(f"Old database entries for doc_id {old_doc_id} removed. Proceeding with new version.")

            with fitz.open(pdf_path) as doc:
                logger.info("Pass 1/2: Analyzing font structure...")
                body_y_bounds = (config.HEADER_Y_RATIO, config.FOOTER_Y_RATIO)
                paragraph_sizes = classify_paragraph_font_sizes(doc, body_y_bounds)
                if not paragraph_sizes:
                    logger.warning(f"Could not identify paragraph fonts for '{doc_name}'. Skipping.")
                    continue

                structured_rows = []
                heading_stack = defaultdict(str)

                logger.info("Pass 2/2: Extracting text, enrichments, and building hierarchy...")
                for page in tqdm(doc, desc=f"  - Scanning Pages for {doc_name}", mininterval=1, ncols=100, file=sys.stdout):
                    # Part A: Extract Page-level Enrichments (FR-10)
                    enrichments = extract_page_enrichments(page, body_y_bounds)

                    # Part B: Extract Text Blocks and Build Hierarchy
                    page_height = page.rect.height
                    body_blocks = [
                        b for b in page.get_text("dict").get("blocks", []) if b.get("type") == 0 and
                        (body_y_bounds[0] * page_height < b['bbox'][1] < body_y_bounds[1] * page_height)
                    ]

                    # FR-22: OCR Fallback Logic
                    # if not body_blocks and Image and pytesseract:
                    #     logger.info(f"Page {page.number + 1} has no text blocks. Attempting OCR (FR-22).")
                    #     try:
                    #         pix = page.get_pixmap(dpi=300)
                    #         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    #         ocr_text = pytesseract.image_to_string(img)
                    #         if ocr_text.strip():
                    #             structured_rows.append({
                    #                 "page_num": page.number + 1, "section_id": "Scanned Content (OCR)",
                    #                 "indexed_text": clean_text(ocr_text), **enrichments
                    #             })
                    #             logger.info(f"OCR successful for page {page.number + 1}.")
                    #     except Exception as ocr_error:
                    #         logger.error(f"OCR failed for page {page.number + 1}: {ocr_error}", exc_info=True)
                    #     continue
                    if not body_blocks:
                        ocr_text = _get_ocr_text_from_page(page)
                        if ocr_text:
                            structured_rows.append({
                                "page_num": page.number + 1,
                                "section_id": "Scanned Content (OCR)",
                                "indexed_text": ocr_text,
                                "header_text": enrichments.get('header_text', ''),
                                "footer_text": enrichments.get('footer_text', ''),
                                "hyperlink_text": enrichments.get('hyperlink_text', ''),
                                "table_text": enrichments.get('table_text', ''),
                                "breadcrumb_path": "",
                                "procedure_title": "",
                                "table_summary": "",
                                "image_summary": "",    
                                # **enrichments
                            })
                        continue # Skip to the next page whether OCR succeeded or not

                    # Process regular text blocks
                    for b in body_blocks:
                        try:
                            block_text = clean_text(" ".join(s['text'] for l in b["lines"] for s in l["spans"]))
                            if not block_text: continue
                            fsize = round(b["lines"][0]["spans"][0]['size'])
                            
                            if fsize not in paragraph_sizes:
                                heading_stack[fsize] = block_text
                                for smaller_size in list(heading_stack):
                                    if smaller_size < fsize: heading_stack[smaller_size] = ""
                            else:
                                # ordered_headings = [heading_stack[k] for k in sorted(heading_stack.keys(), reverse=True) if heading_stack[k]]
                                # section_id = "|".join(ordered_headings) or "Introduction"
                                # structured_rows.append({
                                #     "page_num": page.number + 1, "section_id": section_id,
                                #     "indexed_text": block_text, **enrichments
                                # })
                                ordered_headings = [heading_stack[k] for k in sorted(heading_stack.keys(), reverse=True) if heading_stack[k]]
                                section_id = "|".join(ordered_headings) or "Introduction"

                                # --- Modern 2025: Extract advanced metadata ---
                                breadcrumb_path = get_logical_breadcrumb(heading_stack)
                                procedure_title = detect_procedure(block_text)
                                table_summary = enrichments.get('table_text')
                                if 'table' in str(b.get('type', '')).lower():
                                    table_summary = format_table_as_markdown(b)  # If you want, you can summarize tables more deeply
                                image_summary = None
                                if 'image' in str(b.get('type', '')).lower():  # crude image block check
                                    image_summary = extract_image_summary(b)  # Placeholder: add image extraction logic if needed

                                structured_rows.append({
                                    "page_num": page.number + 1, 
                                    "section_id": section_id,
                                    "indexed_text": block_text,
                                    "header_text": enrichments.get('header_text', ''),
                                    "footer_text": enrichments.get('footer_text', ''),
                                    "hyperlink_text": enrichments.get('hyperlink_text', ''),
                                    "table_text": enrichments.get('table_text', ''),
                                    "breadcrumb_path": breadcrumb_path,
                                    "procedure_title": procedure_title,
                                    "table_summary": table_summary,
                                    "image_summary": image_summary,
                                })
                        except (IndexError, KeyError):
                            continue
            
            # This section is now outside the 'with fitz.open(...)' block
            logger.info("Assembling and saving final structured data to database...")
            if not structured_rows:
                logger.warning(f"No structured text could be extracted from '{doc_name}'. Skipping.")
                continue

            # Group all paragraphs from the same section on the same page
            keyfunc = lambda r: (r["page_num"], r["section_id"])
            structured_rows.sort(key=keyfunc)
            
            sections_to_insert = []
            for (page_num, section_id), group in groupby(structured_rows, key=keyfunc):
                group_list = list(group)
                first_row = group_list[0]
                full_text = "\n".join(r["indexed_text"] for r in group_list)
                
                sections_to_insert.append((
                    page_num, section_id, full_text,
                    first_row['header_text'], first_row['footer_text'],
                    first_row['hyperlink_text'], first_row['table_text'],
                    first_row['breadcrumb_path'], first_row['procedure_title'],
                    first_row['table_summary'], first_row['image_summary']
                ))

            # Insert document record and get its ID
            # cursor.execute("INSERT INTO documents (doc_name, file_hash, status) VALUES (?, ?, ?)", (doc_name, file_hash, 'extracted'))
             # Corrected Version:
            cursor.execute("INSERT INTO documents (doc_name, file_hash, processing_status) VALUES (?, ?, ?)", (doc_name, file_hash, 'extracted'))
            doc_id = cursor.lastrowid

            # Prepare final section data with the correct doc_id
            final_sections_data = [(doc_id, *s) for s in sections_to_insert]

            cursor.executemany("""
                INSERT INTO sections (doc_id, page_num, section_header, raw_text,
                                      header_text, footer_text, hyperlink_text, table_text,
                                    breadcrumb_path, procedure_title, table_summary, image_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, final_sections_data)

            conn.commit()
            logger.info(f"Successfully saved {len(final_sections_data)} sections for '{doc_name}' to the database.")
            successfully_processed_paths.append(pdf_path)

        except Exception as e:
            logger.critical(f"FATAL ERROR processing {doc_name}: {e}", exc_info=True)
            if conn: conn.rollback()
        finally:
            if conn: conn.close()

    
    # if successfully_processed_paths:
    #     logger.info("--- Moving processed files to archive ---")
    #     for path in successfully_processed_paths:
    #         try:
    #             destination_path = config.PROCESSED_DIR / path.name
    #             path.replace(destination_path)
    #             logger.info(f"Moved '{path.name}' to '{destination_path}'")
    #         except Exception as e:
    #             logger.error(f"FAILED to move '{path.name}': {e}")
    if successfully_processed_paths:
        logger.info("--- Moving processed files to archive ---")
        for path in successfully_processed_paths:
            try:
                destination_path = config.PROCESSED_DIR / path.name
                path.replace(destination_path)
                logger.info(f"Moved '{path.name}' to '{destination_path}'.")
            except Exception as e:
                # This is a state that requires manual intervention. The log must be severe.
                logger.critical(
                    f"FATAL: FAILED to move processed file '{path.name}' to '{config.PROCESSED_DIR}'. "
                    f"The database transaction was committed, but the source file was not moved. "
                    f"Please move it manually to prevent processing issues. Error: {e}",
                    exc_info=True
                )

    logger.info("--- PDF Extraction Process Finished ---")

if __name__ == "__main__":
    main()