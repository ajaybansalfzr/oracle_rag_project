# Old Code in scripts/utils/utils.py

import re
import csv
import fitz  # PyMuPDF
from collections import Counter, defaultdict
import logging
import sys
import io



# # --- Standardized Logging Setup (MR-2) ---
# class Unbuffered(io.TextIOWrapper):
#     def __init__(self, stream):
#         super().__init__(stream, write_through=True)

# # Use this unbuffered stream for the logger
# sys.stdout = Unbuffered(sys.stdout)

# --- Force Unbuffered stdout for real-time progress streaming ---
# This robust implementation works whether the script is run directly
# or as a subprocess by wrapping the underlying binary buffer.
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
sys.stderr = io.TextIOWrapper(open(sys.stderr.fileno(), 'wb', 0), write_through=True)

# def log_info(doc_name: str, message: str):
#     """Logs an informational message (stub function)."""
#     print(f"INFO [{doc_name}]: {message}")

# def save_to_csv(data: list, csv_path: str):
#     """Saves a list of dictionaries to a CSV file."""
#     if not data:
#         print(f"WARNING: No data to save to {csv_path}.")
#         return
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=data[0].keys())
#         writer.writeheader()
#         writer.writerows(data)
#     print(f" Successfully saved {len(data)} rows to {csv_path}")

# --- Standardized Logging Setup (MR-2) ---
def get_logger(name: str, level=logging.DEBUG):
    """Creates and configures a standardized logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        # Console Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # TODO: Add FileHandler in a later step when config is centralized
        
    return logger

# Create a logger for this module
logger = get_logger(__name__)

def is_heading(block):
    return block['font_size'] >= 18 and block['is_bold']

def is_procedure_start(text):
    return text.lower().startswith(("steps", "procedure", "to "))

def is_table(block):
    return 'table' in block['type'].lower()

def is_image(block):
    return 'image' in block['type'].lower()

def summarize_table(block):
    # Custom logic: return a summary string or extracted CSV/TSV
    return table_to_csv(block)

def extract_image_caption(block):
    # Use block position or nearby text as caption
    return block['caption'] if 'caption' in block else None

def update_path(path, new_heading):
    # Maintain logical breadcrumbs for navigation
    return path[:-1] + [new_heading] if path else [new_heading]


# ... (keep the existing clean_text and classify_paragraph_font_sizes functions as they are) ...
def get_safe_model_name(model_name: str) -> str:
    """
    Sanitizes a model name by replacing special characters to make it
    safe for use in filenames and SQL column names.
    e.g., 'BAAI/bge-small-en-v1.5' -> 'baai_bge_small_en_v1_5'
    """
    return model_name.lower().replace('/', '_').replace('-', '_').replace('.', '_')

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
        # log_info(page.parent.name, f"Could not extract tables on page {page.number + 1}: {e}")
        logger.warning(f"Could not extract tables on page {page.number + 1} for doc '{page.parent.name}': {e}")

    # 4. Image Extraction is explicitly disabled per requirements.

    return {
        "page_num": page.number + 1,
        "header_text": header_text,
        "footer_text": footer_text,
        "hyperlink_text": hyperlink_text,
        "table_text": "\n---\n".join(table_texts)
    }

def split_text_into_token_chunks(text, tokenizer, max_tokens=256, overlap=0):
    sentences = [s for s in text.split('\n') if s.strip()]
    chunks = []
    chunk = []
    chunk_tokens = 0

    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if chunk_tokens + len(tokens) > max_tokens:
            if chunk:
                chunks.append('\n'.join(chunk))
            # Handle overlap if needed (optional, here is no overlap)
            chunk = [sentence]
            chunk_tokens = len(tokens)
        else:
            chunk.append(sentence)
            chunk_tokens += len(tokens)
    if chunk:
        chunks.append('\n'.join(chunk))
    return chunks
