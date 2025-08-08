import csv

import fitz  # PyMuPDF
from spire.pdf import PdfDocument


# ====== Hyperlink Extraction ====== #
def extract_hyperlinks(pdf_path):
    doc = fitz.open(pdf_path)
    links_data = []

    for page_num, page in enumerate(doc, start=1):
        links = page.get_links()
        for link in links:
            if link.get("uri"):
                rect = fitz.Rect(link["from"])
                anchor_text = page.get_textbox(rect).strip()
                links_data.append({"page": page_num, "anchor_text": anchor_text, "uri": link["uri"]})
    doc.close()
    return links_data


# ====== Bookmark Extraction with Font Info ====== #
def extract_bookmarks_with_fonts(pdf_path):
    # Step 1: Extract bookmark metadata via Spire.PDF
    spire_pdf = PdfDocument()
    spire_pdf.LoadFromFile(pdf_path)

    bookmarks = []

    def recurse(bookmark_collection, level=0):
        for i in range(bookmark_collection.Count):
            bookmark = bookmark_collection[i]
            title = bookmark.Title
            page_num = bookmark.Destination.PageNumber + 1

            # Default font details (in case not found)
            font_size = None
            font_color = None
            is_bold = None

            # Step 2: Try matching this title on the actual page using PyMuPDF
            with fitz.open(pdf_path) as doc:
                page = doc[page_num - 1]
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            if title.strip() in span["text"]:
                                font_size = span["size"]
                                font_color = span["color"]
                                is_bold = "bold" in span.get("font", "").lower()
                                break

            # Add all collected info
            bookmarks.append(
                {
                    "title": title,
                    "page": page_num,
                    "level": level,
                    "font_size": font_size,
                    "font_color": font_color,
                    "is_bold": is_bold,
                }
            )

            children = bookmark.ConvertToBookmarkCollection()
            if children and children.Count > 0:
                recurse(children, level + 1)

    recurse(spire_pdf.Bookmarks)
    spire_pdf.Close()
    return bookmarks


# ===== CSV Save ===== #
def save_bookmarks_to_csv(bookmarks, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["title", "page", "level", "font_size", "font_color", "is_bold"],
        )
        writer.writeheader()
        writer.writerows(bookmarks)


# ====== Main ====== #
if __name__ == "__main__":
    PDF_PATH = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\data\implementing-accounting-hub-reporting.pdf"  # ← Use your real PDF file path
    CSV_OUT = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\data\bookmarks_with_fonts.csv"

    print("\n📚 Extracting Bookmarks with Font Info...")
    bookmarks = extract_bookmarks_with_fonts(PDF_PATH)

    print(f"💾 Saving {len(bookmarks)} bookmarks to CSV: {CSV_OUT}")
    save_bookmarks_to_csv(bookmarks, CSV_OUT)

    print("✅ Done!")
