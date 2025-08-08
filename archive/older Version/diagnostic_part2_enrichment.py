# scripts/diagnostic_part2_enrichment.py

import csv
import io
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADER_END_Y_RATIO = 0.08
FOOTER_START_Y_RATIO = 0.915
OCR_TIMEOUT_SECONDS = 10


def clean_text(text: str) -> str:
    text = re.sub(r"[^ -~]+", "", text)
    return re.sub(r"\s+", " ", text.replace("\u2026", "...").replace("\u2022", "-")).strip()


def extract_text_from_images_on_page(doc: fitz.Document, page: fitz.Page) -> str:
    """Extract OCR from raster images that are truly shown on the page."""
    image_texts = []
    seen_image_hashes = set()

    try:
        # Render only what is visually shown (no shared xref assumption)
        displaylist = page.get_displaylist()
        textpage = displaylist.get_textpage()
        images = displaylist.get_image_info()

        for img in images:
            xref = img["xref"]
            bbox = img["bbox"]

            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Dedup by image hash (same visual content across pages)
            image_hash = hash(image_bytes)
            if image_hash in seen_image_hashes:
                continue
            seen_image_hashes.add(image_hash)

            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, timeout=OCR_TIMEOUT_SECONDS)
            cleaned = clean_text(text)
            if len(cleaned) > 10:
                image_texts.append(cleaned)

    except Exception:
        pass

    return "\n---\n".join(image_texts)


def detect_tables(page: fitz.Page) -> str:
    table_texts = []
    tabs = page.find_tables()
    if tabs.tables:
        for tab in tabs:
            table_data = tab.extract()
            table_str = "\n".join([" | ".join(map(lambda x: str(x or ""), r)) for r in table_data if r])
            table_texts.append(table_str)
    return "\n---\n".join(table_texts)


def extract_page_enrichments(doc: fitz.Document, page: fitz.Page) -> dict:
    page_num = page.number + 1
    page_height = page.rect.height

    # Header and Footer Text
    header_rect = fitz.Rect(0, 0, page.rect.width, page_height * HEADER_END_Y_RATIO)
    footer_rect = fitz.Rect(0, page_height * FOOTER_START_Y_RATIO, page.rect.width, page_height)
    header_text = clean_text(page.get_text(clip=header_rect))
    footer_text = clean_text(page.get_text(clip=footer_rect))

    # Hyperlinks
    hyperlinks = "; ".join([link["uri"] for link in page.get_links() if "uri" in link])

    # Image Text
    image_text = extract_text_from_images_on_page(doc, page)

    # Table Text
    table_text = detect_tables(page)

    return {
        "page_num": page_num,
        "header_text": header_text,
        "footer_text": footer_text,
        "hyperlink_text": hyperlinks,
        "table_text": table_text,
        "image_text": image_text,
    }


def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in '{DATA_DIR}'.")
        return

    pdf_path = pdf_files[0]
    doc_name = pdf_path.stem
    print(f"\nProcessing document for page-level enrichments: {pdf_path.name}")

    all_page_rows = []
    try:
        with fitz.open(pdf_path) as doc:
            print("Extracting headers, footers, images, tables, and hyperlinks from each page...")
            for page in tqdm(doc, desc="Analyzing Pages"):
                enrichments = extract_page_enrichments(doc, page)
                all_page_rows.append(enrichments)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    if not all_page_rows:
        print("No enrichment data could be extracted.")
        return

    csv_path = OUTPUT_DIR / f"diagnostic_part2_enrichment_{doc_name}.csv"
    print(f"\nSaving {len(all_page_rows)} rows of page enrichments to: {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        headers = [
            "page_num",
            "header_text",
            "footer_text",
            "hyperlink_text",
            "table_text",
            "image_text",
        ]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_page_rows)

    print("\n✅ Diagnostic Part 2 complete. Please review the CSV file in your 'output' folder.")


if __name__ == "__main__":
    main()
