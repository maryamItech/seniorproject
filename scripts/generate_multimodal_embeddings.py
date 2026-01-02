"""Generate multimodal embeddings (text + image) using Visualized_BGE.

This script:
- Reads ScienceQA CSV file
- For each record, generates text and image embeddings
- Links text and image embeddings by ID
- Saves results in JSON format for later Milvus upload

Output format:
{
    "id": "unique_id",
    "text": "combined text from question + choices + answer + explanation",
    "image_path": "path/to/image.png",
    "text_embedding": [list of 768 floats],
    "image_embedding": [list of 768 floats]
}
"""

import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add FlagEmbedding to path
FLAG_EMBEDDING_DIR = PROJECT_ROOT / "FlagEmbedding-master"
if str(FLAG_EMBEDDING_DIR / "research") not in sys.path:
    sys.path.insert(0, str(FLAG_EMBEDDING_DIR / "research"))

from visual_bge.modeling import Visualized_BGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def combine_text_fields(row: pd.Series) -> str:
    """Combine question, choices, answer, and explanation into a single text."""
    parts = []
    
    if pd.notna(row.get("question", "")):
        parts.append(f"Question: {row['question']}")
    
    if pd.notna(row.get("choices", "")):
        parts.append(f"Choices: {row['choices']}")
    
    if pd.notna(row.get("answer", "")):
        parts.append(f"Answer: {row['answer']}")
    
    if pd.notna(row.get("explanation", "")):
        parts.append(f"Explanation: {row['explanation']}")
    
    return "\n".join(parts)


def find_image_path(base_dir: Path, record_id: int) -> Optional[Path]:
    """Find image path for a given record ID.
    
    Tries multiple strategies:
    1. Direct match: {record_id}/image.png
    2. String match: {str(record_id)}/image.png (for non-sequential IDs)
    """
    # Try direct path first
    image_path = base_dir / str(record_id) / "image.png"
    if image_path.exists():
        return image_path
    return None


def get_all_image_folders(base_dir: Path) -> List[str]:
    """Get all folder names that contain images, sorted numerically."""
    folders = []
    for d in base_dir.iterdir():
        if d.is_dir():
            image_path = d / "image.png"
            if image_path.exists():
                try:
                    # Try to convert to int for sorting
                    folder_num = int(d.name)
                    folders.append((folder_num, d.name))
                except ValueError:
                    # If not numeric, append as-is
                    folders.append((float('inf'), d.name))
    
    # Sort by numeric value
    folders.sort(key=lambda x: x[0])
    return [name for _, name in folders]


def find_image_path_by_folder(base_dir: Path, folder_name: str) -> Optional[Path]:
    """Find image path using folder name."""
    image_path = base_dir / folder_name / "image.png"
    if image_path.exists():
        return image_path
    return None


def generate_embeddings(
    csv_path: Path,
    images_dir: Path,
    model_weight_path: Path,
    output_path: Path,
    max_records: Optional[int] = None,
    start_from: int = 0,
    model_name: str = "BAAI/bge-base-en-v1.5",
    resume: bool = False,
) -> None:
    """Generate multimodal embeddings for all records in CSV.
    
    Args:
        csv_path: Path to ScienceQA CSV file
        images_dir: Path to images directory (data/train/train)
        model_weight_path: Path to Visualized_BGE model weight
        output_path: Output JSON file path
        max_records: Maximum number of records to process (None = all)
        start_from: Start processing from this record index (0-based)
        model_name: BGE model name
        resume: If True, load existing results and continue from last processed ID
    """
    
    # Load existing results if resuming
    existing_ids = set()
    if resume and output_path.exists():
        logger.info(f"Loading existing results from {output_path} to resume...")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_ids = {item["id"] for item in existing_data}
                logger.info(f"Found {len(existing_ids)} existing records. Will skip them.")
        except Exception as exc:
            logger.warning(f"Could not load existing results: {exc}. Starting fresh.")
    
    # Load CSV
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from CSV")
    
    # Apply start_from and max_records
    if start_from > 0:
        df = df.iloc[start_from:]
        logger.info(f"Starting from record index {start_from}")
    
    if max_records:
        df = df.head(max_records)
        logger.info(f"Processing {max_records} records")
    
    # Get all available image folders (sorted numerically)
    logger.info("Scanning for image folders...")
    available_folders = get_all_image_folders(images_dir)
    logger.info(f"Found {len(available_folders)} folders with images")
    if len(available_folders) > 0:
        logger.info(f"Sample folder names: {available_folders[:10]}...{available_folders[-5:]}")
    
    # Create a set for fast lookup
    available_folders_set = set(available_folders)
    
    # Initialize Visualized_BGE model
    logger.info("Initializing Visualized_BGE model...")
    logger.info(f"Model: {model_name}")
    logger.info(f"Model weight: {model_weight_path}")
    
    model = Visualized_BGE(
        model_name_bge=model_name,
        model_weight=str(model_weight_path)
    )
    model.eval()
    logger.info("Model loaded successfully")
    
    # Process records
    results: List[Dict] = []
    skipped_no_image = 0
    skipped_errors = 0
    
    logger.info("Processing records...")
    # Track which folders we've used to avoid duplicates
    used_folders = set()
    
    for df_idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        try:
            # Try to find matching folder for this CSV row
            # Strategy: Use CSV index + 1 as folder number (since folders are numbered 1, 2, 3...)
            # But also check if folder exists in available folders
            potential_folder = str(df_idx + 1)
            
            # Combine text fields first
            combined_text = combine_text_fields(row)
            if not combined_text.strip():
                logger.warning(f"Record {df_idx + 1}: Empty text, skipping")
                skipped_errors += 1
                continue
            
            # Find image path - try folder name matching CSV index + 1
            folder_name = potential_folder if potential_folder in available_folders_set else None
            
            # If not found, try to find any unused folder (fallback - not ideal)
            if folder_name is None:
                # Try to match by checking if any folder number matches
                for folder in available_folders:
                    if folder not in used_folders:
                        try:
                            folder_num = int(folder)
                            if folder_num == df_idx + 1:
                                folder_name = folder
                                break
                        except ValueError:
                            continue
            
            if folder_name is None or folder_name in used_folders:
                logger.debug(f"Record {df_idx + 1}: No available folder found, skipping")
                skipped_no_image += 1
                continue
            
            image_path = find_image_path_by_folder(images_dir, folder_name)
            if image_path is None:
                logger.debug(f"Record {df_idx + 1}: Image not found in folder {folder_name}, skipping")
                skipped_no_image += 1
                continue
            
            # Mark folder as used
            used_folders.add(folder_name)
            
            # Use folder name as record ID (this is the actual image ID)
            record_id = folder_name
            
            # Skip if already processed (when resuming) - check after finding folder
            if record_id in existing_ids:
                continue
            
            # Generate embeddings
            with torch.no_grad():
                # Text embedding
                text_emb = model.encode(text=combined_text)
                text_emb_list = text_emb[0].cpu().detach().numpy().tolist()
                
                # Image embedding
                img_emb = model.encode(image=str(image_path))
                img_emb_list = img_emb[0].cpu().detach().numpy().tolist()
            
            # Store result
            result = {
                "id": record_id,
                "text": combined_text,
                "image_path": str(image_path),
                "text_embedding": text_emb_list,
                "image_embedding": img_emb_list,
                # Additional metadata
                "question": str(row.get("question", "")),
                "choices": str(row.get("choices", "")),
                "answer": str(row.get("answer", "")),
                "explanation": str(row.get("explanation", "")),
                "split": str(row.get("split", "")),
            }
            results.append(result)
            
        except Exception as exc:
            logger.error(f"Record {df_idx + 1}: Error processing - {exc}")
            skipped_errors += 1
            continue
    
    # Load existing results if resuming and append new results
    if resume and output_path.exists() and existing_ids:
        logger.info(f"Loading existing {len(existing_ids)} records to merge with new {len(results)} records...")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            # Filter out any duplicates and merge
            existing_data_dict = {item["id"]: item for item in existing_data}
            for result in results:
                existing_data_dict[result["id"]] = result
            results = list(existing_data_dict.values())
            logger.info(f"Merged results: {len(results)} total records")
        except Exception as exc:
            logger.warning(f"Could not merge with existing results: {exc}. Saving new results only.")
    
    # Save results
    logger.info(f"\nProcessing complete!")
    logger.info(f"  Total processed in this run: {len(results) if not resume else 'merged'}")
    logger.info(f"  Skipped (no image): {skipped_no_image}")
    logger.info(f"  Skipped (errors): {skipped_errors}")
    
    if not results:
        logger.error("No embeddings generated. Aborting.")
        return
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total records with embeddings: {len(results)}")
    
    # Print sample statistics
    if results:
        sample = results[0]
        logger.info(f"\nSample record:")
        logger.info(f"  ID: {sample['id']}")
        logger.info(f"  Text length: {len(sample['text'])} chars")
        logger.info(f"  Text embedding dim: {len(sample['text_embedding'])}")
        logger.info(f"  Image embedding dim: {len(sample['image_embedding'])}")
        logger.info(f"  Image path: {sample['image_path']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate multimodal embeddings using Visualized_BGE")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "science_qa.csv"),
        help="Path to ScienceQA CSV file"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "train" / "train"),
        help="Path to images directory (train/train)"
    )
    parser.add_argument(
        "--model-weight",
        type=str,
        default=str(PROJECT_ROOT / "Visualized_base_en_v1.5.pth"),
        help="Path to Visualized_BGE model weight file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "embeddings" / "multimodal_embeddings.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records to process (for testing)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start processing from this record index (0-based)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already processed records)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="BGE model name"
    )
    
    args = parser.parse_args()
    
    generate_embeddings(
        csv_path=Path(args.csv),
        images_dir=Path(args.images_dir),
        model_weight_path=Path(args.model_weight),
        output_path=Path(args.output),
        max_records=args.max_records,
        start_from=args.start_from,
        resume=args.resume,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()

