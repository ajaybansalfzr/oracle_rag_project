# scripts/extract_pdf_v6_7.py

from pathlib import Path

import fitz
from tqdm import tqdm

from helpers.helpers_v6_7 import (
    analyze_and_classify_styles,
    convert_block_to_markdown,
    extract_hyperlinks,
    log_info,
    save_to_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "data"
DST_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

for d in (DST_DIR, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

pdf_files = list(SRC_DIR.glob("*.pdf"))
if not pdf_files:
    print("No PDF files found in 'data' directory.")
else:
    for path in pdf_files:
        name = path.stem
        print(f">> Starting dynamic structural analysis for: {name}")
        log_info(name, "Started dynamic extraction process.")

        all_chunks = []
        try:
            with fitz.open(path) as doc:
                # --- Phase 1 & 2: Analyze and classify styles for the entire document ---
                style_map = analyze_and_classify_styles(doc)
                if not style_map:
                    log_info(
                        name,
                        "Could not generate a style map. Aborting extraction for this file.",
                    )
                    continue

                # --- Phase 3: Assemble semantic chunks ---
                current_h1, current_h2, current_h3 = "", "", ""
                current_section_content = []
                chunk_id_counter = 0

                for page_num_0_based in tqdm(range(len(doc)), desc=name):
                    page = doc[page_num_0_based]
                    hyperlinks = extract_hyperlinks(page)
                    blocks = page.get_text("dict").get("blocks", [])

                    for block in blocks:
                        if block.get("type") == 0:  # It's a text block
                            markdown_text, tag = convert_block_to_markdown(block, style_map)
                            if not markdown_text:
                                continue

                            is_new_section = tag in ["h1", "h2", "h3"]
                            if is_new_section and current_section_content:
                                # Save the completed section before starting a new one
                                all_chunks.append(
                                    {
                                        "chunk_id": f"{name}_{chunk_id_counter}",
                                        "Document_Name": name,
                                        "Chapter": current_h1,
                                        "Section": current_h2,
                                        "SubSection": current_h3,
                                        "Content_Markdown": "\n".join(current_section_content),
                                        "Hyperlinks": hyperlinks,
                                    }
                                )
                                current_section_content = []
                                chunk_id_counter += 1
                                if tag == "h1":
                                    current_h2, current_h3 = "", ""
                                if tag == "h2":
                                    current_h3 = ""

                            # Update current heading state and add heading to content
                            if tag == "h1":
                                current_h1 = markdown_text.replace("# ", "")
                            elif tag == "h2":
                                current_h2 = markdown_text.replace("## ", "")
                            elif tag == "h3":
                                current_h3 = markdown_text.replace("### ", "")

                            current_section_content.append(markdown_text)

                # Add the very last section in the document
                if current_section_content:
                    all_chunks.append(
                        {
                            "chunk_id": f"{name}_{chunk_id_counter}",
                            "Document_Name": name,
                            "Chapter": current_h1,
                            "Section": current_h2,
                            "SubSection": current_h3,
                            "Content_Markdown": "\n".join(current_section_content),
                            "Hyperlinks": hyperlinks,
                        }
                    )

        except Exception as e:
            log_info(name, f"FATAL error during dynamic extraction: {e}")
            continue

        if all_chunks:
            csvf = OUTPUT_DIR / f"Oracle_Data_{name}.csv"
            save_to_csv(all_chunks, str(csvf))
            processed_path = DST_DIR / path.name
            path.replace(processed_path)
            print(f"OK: {name}: {len(all_chunks)} semantic sections extracted.")
        else:
            print(f"WARNING: No data extracted for {name}.")

    print("\n>> All PDF files processed with dynamic structural analysis.")
