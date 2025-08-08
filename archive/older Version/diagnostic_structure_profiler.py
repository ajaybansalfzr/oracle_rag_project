# scripts/diagnostic_structure_profiler.py

import csv
import io
import os
import re
from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OCR_TIMEOUT_SECONDS = 10


def clean_text(text: str) -> str:
    """Cleans a string by removing non-ASCII chars and normalizing whitespace."""
    text = re.sub(r"[^ -~]+", "", text)
    return re.sub(r"\s+", " ", text.replace("…", "...").replace("•", "-")).strip()


def extract_text_from_images(doc: fitz.Document, page: fitz.Page) -> str:
    """Extracts text from images on a page."""
    image_texts = []
    for img in page.get_images(full=True):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            text = pytesseract.image_to_string(image, timeout=OCR_TIMEOUT_SECONDS)
            cleaned_text = clean_text(text)
            if len(cleaned_text) > 10:
                image_texts.append(cleaned_text)
        except Exception:
            continue
    return "\n---\n".join(image_texts)


def detect_tables(page: fitz.Page) -> str:
    """Detects and extracts tables from a page."""
    table_texts = []
    tabs = page.find_tables()
    if tabs.tables:
        for tab in tabs:
            table_data = tab.extract()
            table_str = "\n".join([" | ".join(map(lambda x: str(x or ""), r)) for r in table_data])
            table_texts.append(table_str)
    return "\n---\n".join(table_texts)


def main():
    """Main function to run the hierarchical structure profiler."""
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in '{DATA_DIR}'.")
        return

    pdf_path = pdf_files[0]
    doc_name = pdf_path.stem
    print(f"\nProcessing document: {pdf_path.name}")

    main_diagnostic_rows = []
    image_diagnostic_rows = []

    try:
        with fitz.open(pdf_path) as doc:
            print("Step 1: Extracting all text blocks and enrichments page by page...")

            # This list will hold all text blocks from the document with their properties
            all_text_blocks = []

            for page in tqdm(doc, desc="Analyzing Pages"):
                page_num = page.number + 1

                # --- Page-level enrichments ---
                page_enrichments = {
                    "image_text": extract_text_from_images(doc, page),
                    "hyperlink_text": "; ".join([link["uri"] for link in page.get_links() if "uri" in link]),
                    "table_text": detect_tables(page),
                }

                # Store image text for its separate CSV
                if page_enrichments["image_text"]:
                    image_diagnostic_rows.append(
                        {
                            "page_num": page_num,
                            "image_text": page_enrichments["image_text"],
                        }
                    )

                # --- Block-level extraction ---
                blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT).get("blocks", [])

                # Flag to ensure page-level enrichments are only added once per page
                is_first_block_on_page = True

                for b in blocks:
                    if b.get("type") == 0:  # It's a text block
                        try:
                            # Use the first span to represent the style of the whole block
                            first_span = b["lines"][0]["spans"][0]
                            block_text = clean_text(" ".join(s["text"] for l in b["lines"] for s in l["spans"]))
                            if not block_text:
                                continue

                            block_data = {
                                "page_num": page_num,
                                "text": block_text,
                                "font_size": round(first_span["size"]),
                                "y_coord": b["bbox"][1],  # Vertical position for sorting
                                # Assign enrichments only to the first block of the page
                                "hyperlink_text": (
                                    page_enrichments["hyperlink_text"] if is_first_block_on_page else ""
                                ),
                                "table_text": (page_enrichments["table_text"] if is_first_block_on_page else ""),
                            }
                            all_text_blocks.append(block_data)
                            is_first_block_on_page = False
                        except (IndexError, KeyError):
                            continue

            print("Step 2: Building hierarchical Section IDs...")
            # --- Hierarchical ID Generation Pass ---
            # Sort all blocks by page and then by vertical position
            all_text_blocks.sort(key=lambda x: (x["page_num"], x["y_coord"]))

            # Find the distinct font sizes that act as headings
            font_sizes = sorted(list(set(b["font_size"] for b in all_text_blocks)), reverse=True)

            for i, current_block in enumerate(all_text_blocks):
                parent_headings = []
                # Look backwards from the current block to find its parents
                for j in range(i - 1, -1, -1):
                    # A parent must be on the same page and have a larger font size
                    potential_parent = all_text_blocks[j]
                    if potential_parent["page_num"] != current_block["page_num"]:
                        break  # Stop searching once we leave the current page

                    if potential_parent["font_size"] > current_block["font_size"]:
                        # Check if this is the correct parent for its level
                        # This prevents adding multiple headings of the same size to the hierarchy
                        is_closest_parent = True
                        for ph in parent_headings:
                            if potential_parent["font_size"] == ph["font_size"]:
                                is_closest_parent = False
                                break
                        if is_closest_parent:
                            parent_headings.append(potential_parent)

                # Sort the found parents by font size (desc) to create the correct hierarchy
                parent_headings.sort(key=lambda x: x["font_size"], reverse=True)

                section_id = "|".join([ph["text"] for ph in parent_headings])

                # Final assembly for the main CSV row
                main_diagnostic_rows.append(
                    {
                        "page_num": current_block["page_num"],
                        "section_id": section_id,
                        "indexed_text": current_block["text"],
                        # Header/Footer logic would go here if needed, simplified for this diagnostic
                        "header_footer_text": f"Header for Page {current_block['page_num']} | Footer for Page {current_block['page_num']}",
                        "image_text": "",  # This is now in its own CSV
                        "hyperlink_text": current_block["hyperlink_text"],
                        "table_text": current_block["table_text"],
                    }
                )

    except Exception as e:
        print(f"An error occurred: {e}")
        return

    if not main_diagnostic_rows:
        print("No text data could be extracted for the main diagnostic file.")
    else:
        # --- Save Main Diagnostic CSV ---
        csv_path_main = OUTPUT_DIR / f"diagnostic_hierarchical_profile_{doc_name}.csv"
        print(f"\nStep 3: Saving {len(main_diagnostic_rows)} rows to main diagnostic file: {csv_path_main}")
        with open(csv_path_main, "w", newline="", encoding="utf-8") as f:
            headers = [
                "page_num",
                "section_id",
                "indexed_text",
                "header_footer_text",
                "image_text",
                "hyperlink_text",
                "table_text",
            ]
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(main_diagnostic_rows)

    if not image_diagnostic_rows:
        print("No image text was extracted.")
    else:
        # --- Save Image Text CSV ---
        csv_path_images = OUTPUT_DIR / f"diagnostic_image_text_{doc_name}.csv"
        print(f"Step 4: Saving {len(image_diagnostic_rows)} pages with image text to: {csv_path_images}")
        with open(csv_path_images, "w", newline="", encoding="utf-8") as f:
            headers = ["page_num", "image_text"]
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(image_diagnostic_rows)

    print("\n✅ Final diagnostic complete. Please review the CSV files in your 'output' folder.")


if __name__ == "__main__":
    main()
