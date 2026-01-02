"""Build image embeddings for local train images and store them in Milvus (cloud).

This script:
- Scans images under: data/train/train/<id>/image.png
- Extracts text from images using OCR (Tesseract/PaddleOCR)
- Computes embeddings using BGE-small-en (same as text pipeline)
- Stores vectors and basic metadata in a dedicated Milvus collection.

NOTE:
- Uses OCR + BGE instead of CLIP to avoid internet dependency
- Text pipeline (BGE for text) and existing collections are NOT touched.
- This script creates a separate IMAGE collection so it won't break anything.
"""

import sys
from pathlib import Path
from typing import List, Dict

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import numpy as np
from PIL import Image
from tqdm import tqdm
from pymilvus import MilvusClient, DataType

from config.settings import PROJECT_ROOT, MILVUS_URI, MILVUS_TOKEN, EMBEDDING_MODEL_NAME
from utils.embeddings import BGEEmbeddings
from tools.ocr_tool import OCRTool


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_train_images(base_dir: Path) -> List[Dict]:
    """Find images under data/train/train/<id>/image.png."""
    results: List[Dict] = []
    if not base_dir.exists():
        logger.error("Base directory does not exist: %s", base_dir)
        return results

    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        image_path = subdir / "image.png"
        if not image_path.exists():
            continue
        image_id = subdir.name
        results.append(
            {
                "image_id": image_id,
                "path": image_path,
            }
        )
    return results


def build_and_upload() -> None:
    """Embed train images and upload vectors to Milvus."""
    train_dir = PROJECT_ROOT / "data" / "train" / "train"
    logger.info("Scanning images in %s", train_dir)
    images = find_train_images(train_dir)
    if not images:
        logger.error("No images found under %s", train_dir)
        return

    logger.info("Found %d images to embed.", len(images))

    # Initialize OCR and BGE embedder (same as text pipeline)
    logger.info("Initializing OCR tool...")
    ocr_tool = OCRTool(ocr_engine="tesseract")  # Use tesseract by default
    logger.info("Initializing BGE embedding model (%s)...", EMBEDDING_MODEL_NAME)
    embedder = BGEEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectors: List[np.ndarray] = []
    metas: List[Dict] = []
    skipped_no_text = 0

    for item in tqdm(images, desc="Processing images (OCR + BGE)"):
        try:
            # Step 1: Extract text from image using OCR
            ocr_result = ocr_tool.extract_text(item["path"], preprocess=True)
            extracted_text = ocr_result.get("text", "").strip()
            
            if not extracted_text:
                skipped_no_text += 1
                logger.debug("No text extracted from %s, skipping", item["path"])
                continue
            
            # Step 2: Generate embedding using BGE (same as text pipeline)
            vec = embedder.embed_query(extracted_text)
            vectors.append(vec)
            metas.append(
                {
                    "image_id": item["image_id"],
                    "path": str(item["path"]),
                    "extracted_text": extracted_text[:500],  # Store first 500 chars for reference
                    "ocr_confidence": ocr_result.get("confidence"),
                }
            )
        except Exception as exc:
            logger.warning("Failed to process %s: %s", item["path"], exc)
            continue

    if not vectors:
        logger.error("No embeddings generated; aborting.")
        if skipped_no_text > 0:
            logger.warning("%d images were skipped because no text could be extracted via OCR", skipped_no_text)
        return

    emb_matrix = np.stack(vectors).astype("float32")
    dim = emb_matrix.shape[1]
    logger.info("Generated %d image embeddings with dim=%d (BGE-small-en)", emb_matrix.shape[0], dim)
    if skipped_no_text > 0:
        logger.info("Skipped %d images with no extractable text", skipped_no_text)

    # Connect to Milvus
    logger.info("Connecting to Milvus at %s", MILVUS_URI)
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN, timeout=60)

    collection_name = "train_image_bge"
    logger.info("Preparing collection '%s'...", collection_name)

    if client.has_collection(collection_name):
        logger.info("Collection '%s' exists; dropping it first.", collection_name)
        client.drop_collection(collection_name)

    # Create schema: image_id (string PK), path, extracted_text, ocr_confidence, vector
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("image_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("path", DataType.VARCHAR, max_length=512)
    schema.add_field("extracted_text", DataType.VARCHAR, max_length=512)  # OCR text preview
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Bounded",
    )
    logger.info("Collection '%s' created.", collection_name)

    # Insert data
    entities = []
    for meta, vec in zip(metas, emb_matrix):
        entities.append(
            {
                "image_id": str(meta["image_id"]),
                "path": meta["path"],
                "extracted_text": meta.get("extracted_text", "")[:512],  # Truncate to 512 chars
                "vector": vec.tolist(),
            }
        )

    logger.info("Inserting %d image vectors into Milvus...", len(entities))
    insert_result = client.insert(collection_name=collection_name, data=entities)
    logger.info("Insert result: %s", insert_result)

    client.flush(collection_name)
    logger.info("Flush completed.")

    stats = client.get_collection_stats(collection_name)
    logger.info("Collection stats: %s", stats)

    logger.info("DONE. Total embedded images: %d", len(entities))


def main():
    build_and_upload()


if __name__ == "__main__":
    main()


