import fitz  # PyMuPDF
import csv

def extract_all_text_with_fonts(pdf_path):
    doc = fitz.open(pdf_path)
    extracted = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_name = span.get("font", "")
                    is_bold = "bold" in font_name.lower()
                    font_color = span.get("color", 0)
                    font_size = span.get("size", 0)
                    text = span.get("text", "").strip()
                    if text:
                        extracted.append({
                            "Page": page_num,
                            "Text": text,
                            "Font_Name": font_name,
                            "Font_Size": font_size,
                            "Font_Color": font_color,
                            "Is_Bold": is_bold,
                            "X0": span.get("bbox", [None, None, None, None])[0],
                            "Y0": span.get("bbox", [None, None, None, None])[1],
                            "X1": span.get("bbox", [None, None, None, None])[2],
                            "Y1": span.get("bbox", [None, None, None, None])[3],
                        })

    doc.close()
    return extracted

def save_text_data_to_csv(data, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

# ===== MAIN ===== #
if __name__ == "__main__":
    PDF_PATH = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\data\implementing-accounting-hub-reporting.pdf"  # Replace this with your file
    CSV_OUT = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\data\bookmarks_with_fonts.csv"

    print("🔍 Extracting text + font info from PDF...")
    all_text_data = extract_all_text_with_fonts(PDF_PATH)

    print(f"💾 Saving {len(all_text_data)} records to {CSV_OUT}")
    save_text_data_to_csv(all_text_data, CSV_OUT)

    print("✅ Done!")
