#!/usr/bin/env python3
"""
Evaluation Script for Sparse Encoders with Sentence Transformers v5

This script provides evaluation capabilities for sparse encoder models
trained with Sentence Transformers v5.

Usage:
    python evaluate_sparse_encoder_v5.py --model_path ./models/my-model --eval_type all
    python evaluate_sparse_encoder_v5.py --model_path naver/splade-v3 --eval_type beir
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_sparsity(model, sample_texts=None):
    """Evaluate sparsity characteristics of the model."""
    logger.info("Evaluating sparsity characteristics...")
    
    if sample_texts is None:
        sample_texts = [
            "How to implement a binary search algorithm?",
            "What are the best practices for error handling in Python?",
            "Explain the concept of object-oriented programming.",
            "How to optimize database queries for better performance?",
            "What is the difference between supervised and unsupervised learning?"
        ]
    
    # Test both query and document encoding
    query_embeddings = model.encode_query(sample_texts)
    doc_embeddings = model.encode_document(sample_texts)
    
    # Calculate sparsity statistics
    query_stats = model.sparsity(query_embeddings)
    doc_stats = model.sparsity(doc_embeddings)
    
    results = {
        "query_sparsity": {
            "sparsity_ratio": query_stats["sparsity_ratio"],
            "active_dims": query_stats["active_dims"],
            "total_dims": query_embeddings.shape[1]
        },
        "document_sparsity": {
            "sparsity_ratio": doc_stats["sparsity_ratio"],
            "active_dims": doc_stats["active_dims"],
            "total_dims": doc_embeddings.shape[1]
        }
    }
    
    logger.info(f"Query sparsity: {query_stats['sparsity_ratio']:.2%}, Active dims: {query_stats['active_dims']:.1f}")
    logger.info(f"Document sparsity: {doc_stats['sparsity_ratio']:.2%}, Active dims: {doc_stats['active_dims']:.1f}")
    
    return results


def evaluate_beir(model, datasets=None):
    """Evaluate on BEIR datasets using SparseNanoBEIREvaluator."""
    logger.info("Evaluating on BEIR datasets...")
    
    if datasets is None:
        datasets = ["msmarco", "nfcorpus", "nq"]
    
    try:
        evaluator = SparseNanoBEIREvaluator(dataset_names=datasets, batch_size=16)
        results = evaluator(model)
        
        logger.info("BEIR evaluation results:")
        for dataset in datasets:
            if dataset in results:
                dataset_results = results[dataset]
                logger.info(f"  {dataset}:")
                logger.info(f"    NDCG@10: {dataset_results.get('ndcg_at_10', 0):.4f}")
                logger.info(f"    Recall@100: {dataset_results.get('recall_at_100', 0):.4f}")
                logger.info(f"    MAP: {dataset_results.get('map', 0):.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"BEIR evaluation failed: {e}")
        return {}


def evaluate_custom_dataset(model, dataset_name, num_samples=1000):
    """Evaluate on a custom dataset."""
    logger.info(f"Evaluating on custom dataset: {dataset_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        # Prepare queries and documents
        queries = []
        documents = []
        
        for item in dataset:
            if "query" in item and "document" in item:
                queries.append(item["query"])
                documents.append(item["document"])
            elif "question" in item and "answer" in item:
                queries.append(item["question"])
                documents.append(item["answer"])
        
        if not queries or not documents:
            logger.warning("Could not find suitable query-document pairs in dataset")
            return {}
        
        # Encode and evaluate
        start_time = time.time()
        query_embeddings = model.encode_query(queries[:100])  # Limit for speed
        doc_embeddings = model.encode_document(documents[:100])
        encoding_time = time.time() - start_time
        
        # Calculate similarities
        similarities = model.similarity(query_embeddings, doc_embeddings)
        
        # Calculate metrics
        avg_similarity = float(torch.mean(torch.diag(similarities)))
        max_similarity = float(torch.max(similarities))
        min_similarity = float(torch.min(similarities))
        
        results = {
            "dataset_name": dataset_name,
            "num_samples": len(queries),
            "encoding_time": encoding_time,
            "avg_diagonal_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "embedding_shape": list(query_embeddings.shape)
        }
        
        logger.info(f"Custom dataset results:")
        logger.info(f"  Samples processed: {len(queries):,}")
        logger.info(f"  Encoding time: {encoding_time:.2f}s")
        logger.info(f"  Avg diagonal similarity: {avg_similarity:.4f}")
        logger.info(f"  Embedding shape: {query_embeddings.shape}")
        
        return results
        
    except Exception as e:
        logger.error(f"Custom dataset evaluation failed: {e}")
        return {}


def benchmark_performance(model, batch_sizes=None):
    """Benchmark encoding performance across different batch sizes."""
    logger.info("Benchmarking performance...")
    
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32]
    
    test_texts = [f"This is test sentence number {i} for performance benchmarking." for i in range(100)]
    
    results = {}
    
    for batch_size in batch_sizes:
        # Query encoding benchmark
        start_time = time.time()
        query_embeddings = model.encode_query(test_texts[:batch_size], batch_size=batch_size)
        query_time = time.time() - start_time
        
        # Document encoding benchmark
        start_time = time.time()
        doc_embeddings = model.encode_document(test_texts[:batch_size], batch_size=batch_size)
        doc_time = time.time() - start_time
        
        # Calculate throughput
        total_time = query_time + doc_time
        throughput = (2 * batch_size) / total_time if total_time > 0 else 0
        
        results[batch_size] = {
            "query_time": query_time,
            "doc_time": doc_time,
            "throughput": throughput
        }
        
        logger.info(f"Batch size {batch_size}: Query {query_time:.3f}s, Doc {doc_time:.3f}s, Throughput {throughput:.1f} texts/s")
    
    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate sparse encoder models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model or Hugging Face model ID")
    parser.add_argument("--eval_type", choices=["sparsity", "beir", "custom", "performance", "all"], 
                       nargs="+", default=["all"], help="Types of evaluation to perform")
    parser.add_argument("--custom_dataset", type=str,
                       help="Custom dataset name for evaluation")
    parser.add_argument("--output_path", type=str,
                       help="Path to save evaluation report")
    parser.add_argument("--device", type=str,
                       help="Device to run evaluation on (cuda/cpu)")
    parser.add_argument("--beir_datasets", nargs="+", default=["msmarco", "nfcorpus", "nq"],
                       help="BEIR datasets to evaluate on")
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("SPARSE ENCODER EVALUATION WITH SENTENCE TRANSFORMERS V5")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Evaluation types: {args.eval_type}")
    logger.info("=" * 80)
    
    # Load model
    logger.info("Loading model...")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseEncoder(args.model_path, device=device)
    logger.info(f"Model loaded on device: {device}")
    
    # Run evaluations
    results = {
        "model_path": args.model_path,
        "device": device,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {}
    }
    
    eval_types = args.eval_type if "all" not in args.eval_type else ["sparsity", "beir", "performance"]
    
    if "sparsity" in eval_types:
        logger.info("\n" + "="*50)
        logger.info("SPARSITY EVALUATION")
        logger.info("="*50)
        results["results"]["sparsity"] = evaluate_sparsity(model)
    
    if "beir" in eval_types:
        logger.info("\n" + "="*50)
        logger.info("BEIR EVALUATION")
        logger.info("="*50)
        results["results"]["beir"] = evaluate_beir(model, args.beir_datasets)
    
    if "custom" in eval_types and args.custom_dataset:
        logger.info("\n" + "="*50)
        logger.info("CUSTOM DATASET EVALUATION")
        logger.info("="*50)
        results["results"]["custom"] = evaluate_custom_dataset(model, args.custom_dataset)
    
    if "performance" in eval_types:
        logger.info("\n" + "="*50)
        logger.info("PERFORMANCE EVALUATION")
        logger.info("="*50)
        results["results"]["performance"] = benchmark_performance(model)
    
    # Save results if output path provided
    if args.output_path:
        output_file = Path(args.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {output_file}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    main()
