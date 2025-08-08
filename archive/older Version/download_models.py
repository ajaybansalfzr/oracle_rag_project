# Updated Code for scripts/download_models.py

import os

from sentence_transformers import CrossEncoder, SentenceTransformer

from scripts import config


def main():
    """
    Downloads all required embedding and reranking models directly into
    the local project cache directory specified in the config file.
    """
    print(f"Ensuring local model cache directory exists at: {config.CACHE_DIR}")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # --- Create a list of all unique models to download ---
    all_models_to_download = config.EMBEDDING_MODELS_LIST + [config.RERANKER_MODEL]
    unique_model_names = list(dict.fromkeys(all_models_to_download))

    for model_name in unique_model_names:
        local_path = config.MODEL_PATHS.get(model_name)
        if not local_path:
            print(f"WARNING: No path defined in config for model '{model_name}'. Skipping.")
            continue

        # Check if the model is a cross-encoder (reranker) or a standard sentence transformer
        is_cross_encoder = "cross-encoder" in model_name

        print(f"\nProcessing model: {model_name}")
        if os.path.exists(local_path):
            print(f"  - Model already exists locally at {local_path}. Skipping download.")
        else:
            try:
                print(f"  - Downloading model '{model_name}'...")
                if is_cross_encoder:
                    model = CrossEncoder(model_name)
                else:
                    model = SentenceTransformer(model_name)

                print(f"  - Saving model to: {local_path}")
                model.save(local_path)
                print("  - ...Model saved successfully.")
            except Exception as e:
                print(f"  - ERROR: Failed to download or save model '{model_name}'. Details: {e}")

    print("\nAll model downloads are complete.")


if __name__ == "__main__":
    main()
