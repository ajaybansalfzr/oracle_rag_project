
import os, glob, logging
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from helpers.helpers_v6_7_old import (
    extract_pdf_blocks, isolate_header_footer, extract_page_number,
    detect_chapter_from_font, detect_section_from_bookmarks,
    detect_subsection_from_font, chunk_text_by_blank_lines,
    generate_synopsis, identify_context_type, extract_hyperlinks_fitz,
    detect_tables_font8, save_to_csv_sqlite, log_info
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
    last_chapter, last_section, last_subsection = "", "", ""

    for pg, pdata in tqdm(pages.items(), desc=name):
        blocks = pdata["blocks"]
        annotations = pdata["annotations"]
        height = pdata["height"]

        body_lines, header, footer = isolate_header_footer(blocks, height)
        page_num_text = extract_page_number(footer)

        chapter = detect_chapter_from_font(body_lines) or last_chapter
        section = detect_section_from_bookmarks(blocks, pg) or last_section
        subsection = detect_subsection_from_font(body_lines) or last_subsection

        if chapter: last_chapter = chapter
        if section: last_section = section
        if subsection: last_subsection = subsection

        hyperlinks = extract_hyperlinks_fitz(path, pg)
        table_lines = detect_tables_font8(body_lines)
        table_text = "\n".join(table_lines) if table_lines else ""

        chunks = chunk_text_by_blank_lines(body_lines)
        chunk_count = len(chunks)

        for i, chunk in enumerate(chunks):
            syn = generate_synopsis(chunk)
            ctx = identify_context_type(chunk)
            sent_count = len(sent_tokenize(chunk))  

            # Only assign table on first chunk, and hyperlinks on last chunk
            table_data = table_text if i == 0 else ""
            link_data = "; ".join(hyperlinks) if i == chunk_count - 1 else ""

            rows.append({
                "Document_Name": name,
                "Chapter": chapter,
                "Section": section,
                "SubSection": subsection,
                "Paragraph_Synopsis": syn,
                "Sentence_Count": sent_count,
                "Additional_Context": ctx,
                "Header": header,
                "Footer": footer,
                "Indexed_Text": chunk,
                "Table_Delimited": table_data,
                "Hyperlinks": link_data
            })

        log_info(name, f"Processed page {pg}: {len(chunks)} paragraphs, {len(hyperlinks)} hyperlinks")

    csvf = os.path.join(CSV, f"Oracle_Data_{name}.csv")
    dbf  = os.path.join(SQL, f"Oracle_Data_{name}.db")
    save_to_csv_sqlite(rows, csvf, dbf)

    os.replace(path, os.path.join(DST, os.path.basename(path)))
    print(f"✅ {name}: {len(rows)} chunks extracted")
    log_info(name, f"Finished. Total Chunks: {len(rows)}. Saved to CSV and DB.")

print("🎉 All PDF files processed. Data ready for downstream RAG tasks.")
