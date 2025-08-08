import pickle

file_path = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\embeddings\chunk_bm25_all_minilm_l6_v2.pkl"  # Replace with actual path

with open(file_path, "rb") as f:
    bm25_data = pickle.load(f)
    doc_map = bm25_data.get("doc_map", {})
    doc_ids = {entry["doc_id"] for entry in doc_map.values()}
    print("Total Entries:", len(doc_map))
    print("Unique doc_ids:", sorted(doc_ids))
