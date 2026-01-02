"""Batch evaluation script for RAG system on ScienceQA dataset.

This script:
- Loads ScienceQA dataset
- Runs RAG system on each question
- Evaluates each answer using LLM as Judge
- Calculates overall accuracy metrics
- Saves results to JSON
"""

import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.rag_graph import RAGGraph
from retrieval.multimodal_milvus_store import MultimodalMilvusStore
from pymilvus import MilvusClient
from config.settings import (
    MILVUS_URI,
    MILVUS_TOKEN,
    MILVUS_MULTIMODAL_COLLECTION_NAME,
    PROJECT_ROOT,
    SCIENCEQA_METADATA_PATH,
    SCIENCEQA_IMAGES_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_image_path_for_record(base_dir: Path, record_id: int, split: str = "train") -> Optional[Path]:
    """Find image path for a given record ID."""
    # Try different path patterns
    patterns = [
        base_dir / split / f"{record_id}.png",
        base_dir / "train" / "train" / str(record_id) / "image.png",
        base_dir / split / str(record_id) / "image.png",
    ]
    
    for pattern in patterns:
        if pattern.exists():
            return pattern
    return None


def evaluate_dataset(
    csv_path: Path,
    images_dir: Path,
    model_weight_path: Path,
    output_path: Path,
    max_records: Optional[int] = None,
    start_from: int = 0,
    split: Optional[str] = None,  # "train", "validation", "test", or None for all
    model_name: str = "BAAI/bge-base-en-v1.5",
) -> None:
    """
    Evaluate RAG system on ScienceQA dataset.
    
    Args:
        csv_path: Path to ScienceQA CSV file
        images_dir: Path to images directory
        model_weight_path: Path to Visualized_BGE model weight
        output_path: Output JSON file path for evaluation results
        max_records: Maximum number of records to process (None = all)
        start_from: Start processing from this record index (0-based)
        split: Dataset split to evaluate ("train", "validation", "test", or None for all)
        model_name: BGE model name
    """
    
    # Load CSV
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from CSV")
    
    # Filter by split if specified
    if split:
        df = df[df["split"] == split]
        logger.info(f"Filtered to {len(df)} records in '{split}' split")
    
    # Apply start_from and max_records
    if start_from > 0:
        df = df.iloc[start_from:]
        logger.info(f"Starting from record index {start_from}")
    
    if max_records:
        df = df.head(max_records)
        logger.info(f"Processing {max_records} records")
    
    # Initialize Milvus client and vector store
    logger.info("Initializing Milvus client...")
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    
    logger.info("Initializing MultimodalMilvusStore...")
    vector_store = MultimodalMilvusStore(
        client=client,
        collection_name=MILVUS_MULTIMODAL_COLLECTION_NAME,
        model_weight_path=model_weight_path,
        model_name=model_name,
    )
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag_system = RAGGraph(vector_store=vector_store)
    
    # Evaluation results
    results: List[Dict] = []
    total_correct = 0
    total_partially_correct = 0
    total_incorrect = 0
    total_errors = 0
    total_relevant = 0
    total_recall_at_k = 0
    
    logger.info("Starting evaluation...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            # Extract question data
            question = str(row.get("question", "")).strip()
            choices = str(row.get("choices", "")).strip() if pd.notna(row.get("choices")) else ""
            ground_truth = str(row.get("answer", "")).strip()
            split_name = str(row.get("split", "train"))
            record_id = int(row.get("id", idx))
            
            if not question or not ground_truth:
                logger.warning(f"Record {idx}: Missing question or answer, skipping")
                total_errors += 1
                continue
            
            # Find image path if available
            image_path = None
            if pd.notna(row.get("image")):
                image_path = find_image_path_for_record(images_dir, record_id, split_name)
            
            # Determine input type
            input_type = "image" if image_path else "text"
            
            # Run RAG system
            logger.debug(f"Record {idx}: Processing question '{question[:50]}...'")
            rag_result = rag_system.run(
                question=question,
                input_type=input_type,
                image_path=str(image_path) if image_path else None,
            )
            
            system_answer = rag_result.get("answer", "").strip()
            retrieved_docs = rag_result.get("retrieved_docs", [])
            
            if not system_answer:
                logger.warning(f"Record {idx}: No answer generated")
                total_errors += 1
                continue
            
            # Evaluate answer using comprehensive RAG evaluation
            logger.debug(f"Record {idx}: Evaluating answer...")
            
            # Prepare input_data for relevance evaluation
            input_data = question
            if input_type == "image" and image_path:
                # For image queries, use question or image description if available
                visual_desc = rag_result.get("visual_description")
                ocr_text = rag_result.get("ocr_text")
                if visual_desc:
                    input_data = visual_desc
                elif ocr_text:
                    input_data = ocr_text
            
            evaluation = rag_system.rag_chain.evaluate_answer(
                question=question,
                ground_truth=ground_truth,
                system_answer=system_answer,
                choices=choices if choices else None,
                retrieved_docs=retrieved_docs,
                recall_at_k=5,
                input_data=input_data
            )
            
            # Update counters
            correctness = evaluation.get("correctness", "Incorrect")
            relevance = evaluation.get("relevance", "Irrelevant")
            recall_at_k = evaluation.get("recall_at_k", False)
            
            if correctness == "Correct":
                total_correct += 1
            elif correctness == "Partially Correct":
                total_partially_correct += 1
            else:
                total_incorrect += 1
            
            if relevance == "Relevant":
                total_relevant += 1
            
            if recall_at_k:
                total_recall_at_k += 1
            
            # Store result with new format
            result = {
                "record_id": record_id,
                "index": idx,
                "split": split_name,
                "question": question,
                "choices": choices,
                "ground_truth": ground_truth,
                "system_answer": system_answer,
                "relevance": evaluation.get("relevance", "Irrelevant"),
                "recall_at_k": evaluation.get("recall_at_k", False),
                "correctness": correctness,
                "reason": evaluation.get("reason", ""),
                "image_path": str(image_path) if image_path else None,
                "input_type": input_type,
                "retrieved_docs_count": len(retrieved_docs),
            }
            results.append(result)
            
            logger.debug(f"Record {idx}: {eval_result}")
            
        except Exception as e:
            logger.error(f"Record {idx}: Error - {e}")
            import traceback
            traceback.print_exc()
            total_errors += 1
            continue
    
    # Calculate metrics
    total_evaluated = len(results)
    accuracy = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
    partial_accuracy = ((total_correct + total_partially_correct) / total_evaluated * 100) if total_evaluated > 0 else 0
    relevance_percent = (total_relevant / total_evaluated * 100) if total_evaluated > 0 else 0
    recall_at_k_percent = (total_recall_at_k / total_evaluated * 100) if total_evaluated > 0 else 0
    
    summary = {
        "total_records": len(df),
        "total_evaluated": total_evaluated,
        "total_errors": total_errors,
        "correct": total_correct,
        "partially_correct": total_partially_correct,
        "incorrect": total_incorrect,
        "relevant_retrievals": total_relevant,
        "recall_at_k_success": total_recall_at_k,
        "accuracy_percent": accuracy,
        "partial_accuracy_percent": partial_accuracy,
        "relevance_percent": relevance_percent,
        "recall_at_k_percent": recall_at_k_percent,
        "split": split,
        "start_from": start_from,
        "max_records": max_records,
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Generate final evaluation report using LLM
    logger.info("Generating final evaluation report...")
    try:
        final_report = rag_system.rag_chain.generate_final_evaluation_report(results)
        summary["final_report"] = final_report
    except Exception as e:
        logger.warning(f"Failed to generate final report: {e}")
        summary["final_report"] = "Report generation failed"
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total records processed: {total_evaluated}")
    logger.info(f"Correct: {total_correct} ({accuracy:.2f}%)")
    logger.info(f"Partially Correct: {total_partially_correct}")
    logger.info(f"Incorrect: {total_incorrect}")
    logger.info(f"Relevant Retrievals: {total_relevant} ({relevance_percent:.2f}%)")
    logger.info(f"Recall@K Success: {total_recall_at_k} ({recall_at_k_percent:.2f}%)")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Overall Accuracy: {accuracy:.2f}%")
    logger.info(f"Partial Accuracy (Correct + Partially Correct): {partial_accuracy:.2f}%")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)
    
    if "final_report" in summary and summary["final_report"]:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL EVALUATION REPORT")
        logger.info("=" * 80)
        logger.info(summary["final_report"])
        logger.info("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system on ScienceQA dataset")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(SCIENCEQA_METADATA_PATH),
        help="Path to ScienceQA CSV file"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(SCIENCEQA_IMAGES_DIR),
        help="Path to images directory"
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
        default=str(PROJECT_ROOT / "evaluation_results" / "dataset_evaluation.json"),
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
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default=None,
        help="Dataset split to evaluate (train/validation/test)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="BGE model name"
    )
    
    args = parser.parse_args()
    
    evaluate_dataset(
        csv_path=Path(args.csv),
        images_dir=Path(args.images_dir),
        model_weight_path=Path(args.model_weight),
        output_path=Path(args.output),
        max_records=args.max_records,
        start_from=args.start_from,
        split=args.split,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()



