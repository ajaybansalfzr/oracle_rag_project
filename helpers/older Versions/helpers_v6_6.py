import os, re, csv, sqlite3, fitz, hashlib
from datetime import datetime
from nltk.tokenize import sent_tokenize
from ollama import Client
from spire.pdf import PdfDocument

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

def extract_hyperlinks_fitz(pdf_path: str, page_num: int):
    doc = fitz.open(pdf_path)
    links = []
    page = doc[page_num - 1]
    for link in page.get_links():
        if link.get("uri"):
            rect = fitz.Rect(link["from"])
            anchor = page.get_textbox(rect).strip()
            if anchor:
                links.append(f"{anchor} | {link['uri']}")
    doc.close()
    return links

def detect_tables_font8(lines: list) -> list:
    rows = []
    for bbox, spans in lines:
        if all(round(span.get("size", 0)) == 8 for span in spans):
            row = [clean_text(span.get("text", "")) for span in spans]
            if any(row):
                rows.append(" | ".join(row))
    return rows

def extract_image_refs_aligned(images, page_num, body_lines):
    refs = []
    for img in images:
        img_y = img[1] if len(img) > 1 else 0
        try:
            img_y = float(img_y)
        except (ValueError, TypeError):
            img_y = 0
        for bbox, spans in body_lines:
            text_y = bbox[1] if bbox and len(bbox) > 1 else 0
            try:
                text_y = float(text_y)
            except (ValueError, TypeError):
                text_y = 0
            if abs(text_y - img_y) < 30:
                xref = img[0]
                refs.append(f"page{page_num}_img_{xref}")
    return refs

def load_bookmarks(pdf_path: str):
    bookmarks = []
    spire_pdf = PdfDocument()
    spire_pdf.LoadFromFile(pdf_path)

    def recurse(bookmark_collection, level=0):
        for i in range(bookmark_collection.Count):
            bm = bookmark_collection[i]
            title = bm.Title
            page = bm.Destination.PageNumber + 1
            bookmarks.append({"title": title, "page": page, "level": level})
            children = bm.ConvertToBookmarkCollection()
            if children and children.Count > 0:
                recurse(children, level + 1)

    recurse(spire_pdf.Bookmarks)
    spire_pdf.Close()
    return bookmarks

def detect_chapter_from_bookmarks(bookmarks, page):
    candidates = [b for b in bookmarks if b["page"] == page and b["level"] == 0]
    return candidates[0]["title"] if candidates else ""

def detect_section_from_bookmarks(bookmarks, page):
    candidates = [b for b in bookmarks if b["page"] == page and b["level"] == 1]
    return candidates[0]["title"] if candidates else ""

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

def generate_context_id(page, chapter, section, text, spans):
    # Use font details from first valid span
    for bbox, span_list in spans:
        for s in span_list:
            if clean_text(s.get("text", "")):
                font_size = int(s.get("size", 0))
                font_color = s.get("color", 0)
                is_bold = "bold" in s.get("font", "").lower()
                short = clean_text(text)[:15].replace(" ", "_")
                return f"{page}_{chapter}_{section}_{font_size}_{font_color}_{'B' if is_bold else 'N'}_{short}"
    return f"{page}_{chapter}_{section}_UNKNOWN"

def save_to_csv_sqlite(rows: list, csv_path: str, sqlite_path: str):
    fields = [
        "Document_Name", "Chapter", "Section", "Paragraph_Number",
        "Paragraph_Synopsis", "Sentence_Count", "Additional_Context",
        "Header", "Footer", "Indexed_Text", "Table_Delimited",
        "Image_Refs", "Hyperlinks", "Context_ID"
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
            Table_Delimited TEXT, Image_Refs TEXT, Hyperlinks TEXT, Context_ID TEXT
        )
    """)
    ins = "INSERT INTO rag_chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    for r in rows:
        c.execute(ins, [r.get(k, "") for k in fields])
    conn.commit(); conn.close()
