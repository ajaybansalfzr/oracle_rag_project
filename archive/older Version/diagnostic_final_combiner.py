# scripts/diagnostic_final_combiner.py

from pathlib import Path

import pandas as pd

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
DOC_NAME = "implementing-accounting-hub-reporting"

part1_path = OUTPUT_DIR / f"diagnostic_part1_section_id_{DOC_NAME}.csv"
part2_path = OUTPUT_DIR / f"diagnostic_part2_enrichment_{DOC_NAME}.csv"
final_path = OUTPUT_DIR / f"diagnostic_final_combined_{DOC_NAME}.csv"

# --- Load both datasets ---
df_structure = pd.read_csv(part1_path)
df_enrich = pd.read_csv(part2_path)

# --- Merge on page_num ---
df_final = pd.merge(df_structure, df_enrich, on="page_num", how="left")

# --- Drop image_text if exists ---
if "image_text" in df_final.columns:
    df_final.drop(columns=["image_text"], inplace=True)

# --- Save final merged CSV ---
df_final.to_csv(final_path, index=False)
print(f"\n✅ Final merged file saved to: {final_path}")
