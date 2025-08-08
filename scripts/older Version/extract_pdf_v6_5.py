import os, glob

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pdf_extraction.log"), logging.StreamHandler()]
)

from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from helpers.helpers_v6_5 import (
    extract_pdf_blocks, isolate_header_footer, extract_page_number,
    detect_chapter_from_header, detect_section_from_font_size,
    chunk_text_by_blank_lines, generate_synopsis, identify_context_type,
    extract_image_refs, extract_hyperlinks, detect_tables,
    save_to_csv_sqlite, log_info
)

SRC = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\data"
DST = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\data\processed"
CSV = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\output"
SQL = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\output"
for d in (DST, CSV, SQL):
    os.makedirs(d, exist_ok=True)

for path in glob.glob(os.path.join(SRC, "*.pdf")):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"🔹 Processing {name}")
    log_info(name, "Started processing document")

    pages = extract_pdf_blocks(path)
    rows = []
    last_chapter, last_section = "", ""
    para_counter = {}

    for pg, pdata in tqdm(pages.items(), desc=name):
        blocks = pdata["blocks"]
        images = pdata["images"]
        annotations = pdata["annotations"]
        height = pdata["height"]

        body_lines, header, footer = isolate_header_footer(blocks, height)
        page_num_text = extract_page_number(footer)

        chapter = detect_chapter_from_header(header) or last_chapter
        section = detect_section_from_font_size(body_lines, last_section)

        if chapter != last_chapter or section != last_section:
            para_counter[(chapter, section)] = 0
            last_chapter, last_section = chapter, section

        hyperlinks = extract_hyperlinks(blocks, annotations)
        image_refs = extract_image_refs(images, pg)
        table_lines = detect_tables(body_lines)
        table_text = "\n".join(table_lines) if table_lines else ""

        chunks = chunk_text_by_blank_lines(body_lines)

        for chunk in chunks:
            para_counter[(chapter, section)] += 1
            pnum = para_counter[(chapter, section)]
            syn = generate_synopsis(chunk)
            ctx = identify_context_type(chunk)
            sent_count = len(sent_tokenize(chunk))
            rows.append({
                "Document_Name": name,
                "Chapter": chapter,
                "Section": section,
                "Paragraph_Number": pnum,
                "Paragraph_Synopsis": syn,
                "Sentence_Count": sent_count,
                "Additional_Context": ctx,
                "Header": header,
                "Footer": footer,
                "Indexed_Text": chunk,
                "Table_Delimited": table_text,
                "Image_Refs": "; ".join(image_refs),
                "Hyperlinks": "; ".join(hyperlinks)
            })

        log_info(name, f"Processed page {pg}: {len(chunks)} paragraphs, {len(image_refs)} images, {len(hyperlinks)} hyperlinks")

    csvf = os.path.join(CSV, f"Oracle_Data_{name}.csv")
    dbf  = os.path.join(SQL, f"Oracle_Data_{name}.db")
    save_to_csv_sqlite(rows, csvf, dbf)

    os.replace(path, os.path.join(DST, os.path.basename(path)))
    print(f"✅ {name}: {len(rows)} chunks extracted")
    log_info(name, f"Finished. Total Chunks: {len(rows)}. Saved to CSV and DB.")

print("🎉 All PDF files processed. Data ready for downstream RAG tasks.")