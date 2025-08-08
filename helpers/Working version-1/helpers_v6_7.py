# scripts/helpers/helpers_v6_7.py

import re
import csv
import fitz  # PyMuPDF
from collections import Counter, defaultdict

def log_info(doc_name: str, message: str):
    """Logs an informational message (stub function)."""
    print(f"INFO [{doc_name}]: {message}")

def save_to_csv(data: list, csv_path: str):
    """Saves a list of dictionaries to a CSV file."""
    if not data:
        print(f"WARNING: No data to save to {csv_path}.")
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Successfully saved {len(data)} rows to {csv_path}")

def clean_text(text: str) -> str:
    """Cleans a string by removing non-printable chars and normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^ -~]+", "", text)  # Remove non-ASCII/non-printable
    return re.sub(r"\s+", " ", text.replace("…", "...").replace("•", "-")).strip()

def classify_paragraph_font_sizes(doc: fitz.Document, body_y_bounds: tuple) -> set:
    """
    Identifies the most common font sizes that constitute paragraph text based on
    character count distribution within the main body of the document.
    """
    size_counter = Counter()
    header_y_ratio, footer_y_ratio = body_y_bounds

    for page in doc:
        page_height = page.rect.height
        blocks = page.get_text("dict").get("blocks", [])
        body_blocks = [
            b for b in blocks if b.get("type") == 0 and
            (header_y_ratio * page_height < b['bbox'][1] < footer_y_ratio * page_height)
        ]
        for b in body_blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span['size'])
                    size_counter[size] += len(span.get("text", ""))

    if not size_counter:
        return set()

    # Determine paragraph sizes by finding the font size with the most characters
    # and including any smaller sizes, which are likely footnotes or annotations.
    most_common_size = max(size_counter, key=size_counter.get)
    paragraph_sizes = {size for size in size_counter if size <= most_common_size}
    return paragraph_sizes

def extract_page_enrichments(page: fitz.Page, body_y_bounds: tuple) -> dict:
    """
    Extracts headers, footers, hyperlinks, and tables from a single page.
    Image extraction is explicitly disabled.
    """
    header_y_ratio, footer_y_ratio = body_y_bounds
    page_height = page.rect.height
    page_width = page.rect.width

    # 1. Header and Footer Extraction
    header_rect = fitz.Rect(0, 0, page_width, page_height * header_y_ratio)
    footer_rect = fitz.Rect(0, page_height * footer_y_ratio, page_width, page_height)
    header_text = clean_text(page.get_text(clip=header_rect))
    footer_text = clean_text(page.get_text(clip=footer_rect))

    # 2. Hyperlink Extraction
    links = page.get_links()
    hyperlink_texts = []
    for link in links:
        if link.get('kind') == 2 and 'uri' in link:  # Kind 2 is URI
            # Find text within the link's bounding box to get anchor text
            anchor = clean_text(page.get_text(clip=link['from']))
            hyperlink_texts.append(f"{anchor} ({link['uri']})")
    hyperlink_text = "; ".join(hyperlink_texts)

    # 3. Table Extraction
    table_texts = []
    try:
        tabs = page.find_tables()
        if tabs.tables:
            for i, tab in enumerate(tabs):
                table_data = tab.extract()
                if not table_data: continue
                # Format each row as a |-delimited string
                table_str = "\n".join([
                    " | ".join(clean_text(str(cell)) for cell in row)
                    for row in table_data if any(row)
                ])
                table_texts.append(f"[Table {i+1}]:\n{table_str}")
    except Exception as e:
        log_info(page.parent.name, f"Could not extract tables on page {page.number + 1}: {e}")

    # 4. Image Extraction is explicitly disabled per requirements.

    return {
        "page_num": page.number + 1,
        "header_text": header_text,
        "footer_text": footer_text,
        "hyperlink_text": hyperlink_text,
        "table_text": "\n---\n".join(table_texts)
    }