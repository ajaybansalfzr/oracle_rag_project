import re, sqlite3, csv, fitz, uuid, os

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pdf_extraction.log"), logging.StreamHandler()]
)

from datetime import datetime
from nltk.tokenize import sent_tokenize
from ollama import Client

OLLAMA = Client()

LOG_DIR = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\output\logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_error(doc_name: str, message: str):
    log_file = os.path.join(LOG_DIR, f"Oracle_Data_{doc_name}.log")
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} ERROR: {message}\n")

def log_info(doc_name: str, message: str):
    log_file = os.path.join(LOG_DIR, f"Oracle_Data_{doc_name}.log")
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} INFO: {message}\n")

def clean_text(text: str) -> str:
    text = text.replace("…", "...").replace("•", "-").replace("\u00A0", " ")
    text = re.sub(r"[^ -\x7E]+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_pdf_blocks(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    pages = {}
    for i in range(len(doc)):
        try:
            page = doc[i]
            blocks = page.get_text("dict")["blocks"]
            images = page.get_images(full=True)
            try:
                annots = []
                for annot in page.annots() or []:
                    # Extract only the needed info for hyperlinks
                    annots.append({
                        "uri": getattr(annot, "uri", None),
                        "info": dict(getattr(annot, "info", {}))
                    })
            except Exception as e:
                annots = []
                log_error(os.path.basename(pdf_path), f"Page {i+1} annotation error: {str(e)}")
            height = page.rect.height
            pages[i + 1] = {"blocks": blocks, "images": images, "annotations": annots, "height": height}
        except Exception as e:
            log_error(os.path.basename(pdf_path), f"Page {i+1} skipped: {str(e)}")
    doc.close()
    log_info(os.path.basename(pdf_path), f"Loaded {len(pages)} pages from PDF")
    return pages

def isolate_header_footer(blocks: list, page_height: float):
    header_lines, footer_lines, body_lines = [], [], []
    for block in blocks:
        for line in block.get("lines", []):
            y = line["bbox"][1]
            text = " ".join([span["text"] for span in line["spans"]])
            clean = clean_text(text)
            if not clean:
                continue
            if y < page_height * 0.08:
                header_lines.append((line["bbox"], clean))
            elif y > page_height * 0.915:
                footer_lines.append((line["bbox"], clean))
            else:
                body_lines.append((line["bbox"], line["spans"]))
    header = " | ".join(sorted(set([h[1] for h in header_lines]), key=lambda s: s.lower()))
    footer = " | ".join(sorted(set([f[1] for f in footer_lines]), key=lambda s: s.lower()))
    return body_lines, header.strip(), footer.strip()

def extract_page_number(footer: str) -> str:
    roman_pattern = r"\b[iIvVxXlLcCdDmM]{1,6}\b"
    arabic_pattern = r"\b\d{1,4}\b"
    roman_match = re.findall(roman_pattern, footer)
    arabic_match = re.findall(arabic_pattern, footer)
    return roman_match[-1] if roman_match else (arabic_match[-1] if arabic_match else "")

def extract_hyperlinks(blocks: list, annotations: list):
    links = set()
    for block in blocks:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = clean_text(span.get("text", ""))
                uri = span.get("uri", "")
                if uri and text:
                    links.add(f"{text} | {uri}")
    for annot in annotations:
        if annot.get("uri") and annot.get("info", {}).get("title"):
            links.add(f"{annot['info'].get('title')} | {annot['uri']}")
    return list(links)

def chunk_text_by_blank_lines(spans_list: list, min_words=30, max_words=500) -> list:
    chunks, buffer = [], []
    for bbox, spans in spans_list:
        text = " ".join(clean_text(span.get("text", "")) for span in spans).strip()
        if not text:
            if buffer:
                joined = " ".join(buffer).strip()
                if len(joined.split()) >= min_words:
                    chunks.append(joined)
                buffer = []
        else:
            buffer.append(text)
    if buffer:
        joined = " ".join(buffer).strip()
        if len(joined.split()) >= min_words:
            chunks.append(joined)
    return chunks

def detect_chapter_from_header(header_text: str) -> str:
    match = re.search(r"(Chapter\s+\d+)", header_text, re.IGNORECASE)
    return match.group(1) if match else ""

def detect_section_from_font_size(spans_list: list, current_section: str) -> str:
    section_candidates = []
    for _, spans in spans_list:
        for span in spans:
            text = clean_text(span["text"])
            if len(text.split()) <= 10 and re.search(r"^[A-Z][A-Za-z0-9\s:\-]{3,}$", text):
                section_candidates.append((span["size"], text))
    if not section_candidates:
        return current_section
    section = sorted(section_candidates, key=lambda x: -x[0])[0][1]
    return section.strip()

def strip_boilerplate(text: str) -> str:
    patterns = [
        r"^Here is a summary.*?:\s*",
        r"^This section explains.*?:\s*",
        r"^In summary.*?:\s*",
        r"^The following is.*?:\s*"
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()

def generate_synopsis(text: str) -> str:
    prompt = "Summarize this in 1–2 concise sentences (max 50 words), no preamble:\n\n" + text
    resp = OLLAMA.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    s = resp["message"]["content"].strip()
    s = strip_boilerplate(s)
    sents = sent_tokenize(s)
    out = " ".join(sents[:2])
    return " ".join(out.split()[:50]).strip()

def identify_context_type(text: str) -> str:
    prompt = "Choose exactly one: Overview, Instruction, Definition, Note, Tips, None. Reply with only that word.\n\n" + text
    resp = OLLAMA.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    tag = resp["message"]["content"].strip().strip('"\n ')
    return tag if tag in {"Overview", "Instruction", "Definition", "Note", "Tips", "None"} else "None"

def save_to_csv_sqlite(rows: list, csv_path: str, sqlite_path: str):
    fields = [
        "Document_Name", "Chapter", "Section", "Paragraph_Number",
        "Paragraph_Synopsis", "Sentence_Count", "Additional_Context",
        "Header", "Footer", "Indexed_Text", "Table_Delimited",
        "Image_Refs", "Hyperlinks"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            Document_Name TEXT, Chapter TEXT, Section TEXT,
            Paragraph_Number INTEGER, Paragraph_Synopsis TEXT,
            Sentence_Count INTEGER, Additional_Context TEXT,
            Header TEXT, Footer TEXT, Indexed_Text TEXT,
            Table_Delimited TEXT, Image_Refs TEXT, Hyperlinks TEXT
        )
    """)
    ins = "INSERT INTO rag_chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
    for r in rows:
        c.execute(ins, [r.get(k, "") for k in fields])
    conn.commit(); conn.close()

def extract_image_refs(image_list: list, page_num: int) -> list:
    refs = []
    for img in image_list:
        xref = img[0]
        name = f"page{page_num}_img_{xref}"
        refs.append(name)
    return refs

def detect_tables(lines: list) -> list:
    rows = []
    for bbox, spans in lines:
        cells = [clean_text(span.get("text", "")) for span in spans]
        if len(cells) > 1 and any(cells):
            rows.append(" | ".join(cells))
    return rows