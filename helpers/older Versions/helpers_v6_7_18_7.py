# scripts/helpers/helpers_v6_7.py

import os
import re
import csv
import fitz
from datetime import datetime
from pathlib import Path
from collections import Counter

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "output" / "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def log_info(doc_name: str, message: str):
    """Logs an informational message to a document-specific log file."""
    log_file = LOG_DIR / f"Oracle_Data_{doc_name}.log"
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} INFO: {message}\n")


def clean_text(text: str) -> str:
    """Cleans a string by removing non-ASCII chars and normalizing whitespace."""
    text = text.replace("…", "...").replace("•", "-").replace("\u00A0", " ")
    return re.sub(r"\s+", " ", text).strip()


def analyze_and_classify_styles(doc: fitz.Document) -> dict:
    """
    <<< THIS IS THE NEW "DOCUMENT PROFILER" & "STYLE CLASSIFIER" >>>
    Performs a statistical analysis of font styles in the document and
    dynamically classifies them into a hierarchy (h1, h2, h3, p).
    """
    styles = {}
    # 1. First Pass: Profile the document and count character length for each style
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if b.get("type") == 0:  # It's a text block
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        style_key = f"{round(s['size'])}_{s['font']}"
                        styles[style_key] = styles.get(style_key, 0) + len(s['text'])

    if not styles:
        log_info(Path(doc.name).stem, "No text styles found in document.")
        return {}

    # 2. Identify the most common style as the paragraph style
    paragraph_style_key = max(styles, key=styles.get)
    p_size = float(paragraph_style_key.split('_')[0])
    
    # 3. Classify all other styles relative to the paragraph style
    style_map = {}
    # Sort other styles by font size (descending) to find headings
    other_styles = sorted([item for item in styles.items() if item[0] != paragraph_style_key], key=lambda x: float(x[0].split('_')[0]), reverse=True)
    
    style_map[paragraph_style_key] = "p"
    heading_levels = ["h1", "h2", "h3"]

    for key, count in other_styles:
        size = float(key.split('_')[0])
        # Assign heading levels based on size relative to paragraph text
        if size > p_size and heading_levels:
            style_map[key] = heading_levels.pop(0) # Assign h1, then h2, etc.
        else:
            style_map[key] = "p" # Classify as paragraph if not significantly larger

    log_info(Path(doc.name).stem, f"Dynamically classified styles: {style_map}")
    return style_map


def convert_block_to_markdown(block, style_map):
    """Converts a text block to Markdown based on its dynamically classified style."""
    try:
        spans = block['lines'][0]['spans']
        if not spans: return "", None
        
        style_key = f"{round(spans[0]['size'])}_{spans[0]['font']}"
        tag = style_map.get(style_key, "p")
        
        text = " ".join([clean_text(s['text']) for s in spans])

        if tag == "h1": return f"# {text}", tag
        if tag == "h2": return f"## {text}", tag
        if tag == "h3": return f"### {text}", tag
        
        # For paragraphs, add bolding for emphasis
        markdown_text = ""
        for span in spans:
            if "bold" in span['font'].lower():
                markdown_text += f"**{clean_text(span['text'])}** "
            else:
                markdown_text += f"{clean_text(span['text'])} "
        return markdown_text.strip(), "p" # Ensure the tag is 'p'
    except:
        return "", None


def extract_hyperlinks(page: fitz.Page) -> str:
    """Extracts and formats all hyperlinks on a page."""
    links = [link["uri"] for link in page.get_links() if "uri" in link]
    return "; ".join(links)


def save_to_csv(rows: list, csv_path: str):
    """Saves the extracted data to a CSV file."""
    if not rows: return
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)