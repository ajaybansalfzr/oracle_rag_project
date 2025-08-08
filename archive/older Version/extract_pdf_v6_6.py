import glob
import os

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from helpers.helpers_v6_6 import (
    chunk_text_by_blank_lines,
    detect_chapter_from_bookmarks,
    detect_section_from_bookmarks,
    detect_tables_font8,
    extract_hyperlinks_fitz,
    extract_image_refs_aligned,
    extract_page_number,
    extract_pdf_blocks,
    generate_context_id,
    generate_synopsis,
    identify_context_type,
    isolate_header_footer,
    load_bookmarks,
    log_info,
    save_to_csv_sqlite,
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

    # Load pages, images, annotations
    pages = extract_pdf_blocks(path)

    # Load bookmarks (used for chapter/section detection)
    bookmarks = load_bookmarks(path)

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

        # New: Bookmark-driven chapter/section detection
        chapter = detect_chapter_from_bookmarks(bookmarks, pg) or last_chapter
        section = detect_section_from_bookmarks(bookmarks, pg) or last_section

        if chapter != last_chapter or section != last_section:
            para_counter[(chapter, section)] = 0
            last_chapter, last_section = chapter, section

        hyperlinks = extract_hyperlinks_fitz(path, pg)
        image_refs = extract_image_refs_aligned(images, pg, body_lines)
        table_lines = detect_tables_font8(body_lines)
        table_text = "\n".join(table_lines) if table_lines else ""

        chunks = chunk_text_by_blank_lines(body_lines)

        for chunk in chunks:
            para_counter[(chapter, section)] += 1
            pnum = para_counter[(chapter, section)]
            syn = generate_synopsis(chunk)
            ctx = identify_context_type(chunk)
            sent_count = len(sent_tokenize(chunk))
            context_id = generate_context_id(pg, chapter, section, chunk, body_lines)

            rows.append(
                {
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
                    "Hyperlinks": "; ".join(hyperlinks),
                    "Context_ID": context_id,
                }
            )

        log_info(
            name,
            f"Processed page {pg}: {len(chunks)} paragraphs, {len(image_refs)} images, {len(hyperlinks)} hyperlinks",
        )

    csvf = os.path.join(CSV, f"Oracle_Data_{name}.csv")
    dbf = os.path.join(SQL, f"Oracle_Data_{name}.db")
    save_to_csv_sqlite(rows, csvf, dbf)

    os.replace(path, os.path.join(DST, os.path.basename(path)))
    print(f"✅ {name}: {len(rows)} chunks extracted")
    log_info(name, f"Finished. Total Chunks: {len(rows)}. Saved to CSV and DB.")

print("🎉 All PDF files processed. Data ready for downstream RAG tasks.")
