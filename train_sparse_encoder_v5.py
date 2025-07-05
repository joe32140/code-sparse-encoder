#!/usr/bin/env python3
"""
Sparse Encoder Training with Sentence Transformers v5 and Cornstack Dataset

This script demonstrates modern sparse encoder training using the latest Sentence Transformers v5
features with the high-quality Cornstack dataset from Nomic AI.

Usage:
    python train_sparse_encoder_v5.py --dataset python --model_type splade --config development
    python train_sparse_encoder_v5.py --dataset java --model_type inference_free_splade --config production
"""

import argparse
import logging
import os
import wandb

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.models import Router
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.sparse_encoder.models import MLMTransformer, SparseStaticEmbedding, SpladePooling
from sentence_transformers.training_args import BatchSamplers

from config import CORNSTACK_DATASETS, BASE_MODELS, TRAINING_CONFIGS, MODEL_HYPERPARAMS, EVALUATION_DATASETS

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cornstack_dataset(dataset_name, max_samples=None):
    """Load and prepare the Cornstack dataset for training."""
    logger.info(f"Loading Cornstack dataset: {dataset_name}")
    
    # Load the dataset
    if max_samples:
        full_dataset = load_dataset(dataset_name, split="train")
        full_dataset = full_dataset.select(range(min(max_samples, len(full_dataset))))
        logger.info(f"Limited dataset to {max_samples} samples")
    else:
        full_dataset = load_dataset(dataset_name, split="train")
        logger.info("Loading full dataset")
    
    # Split into train and eval
    dataset_dict = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    logger.info(f"Dataset features: {list(train_dataset.features.keys())}")
    
    # Log sample data structure
    sample = train_dataset[0]
    logger.info("Sample data structure:")
    for key, value in sample.items():
        if isinstance(value, str):
            preview = value[:100] + "..." if len(value) > 100 else value
            logger.info(f"  {key}: {preview}")
        else:
            logger.info(f"  {key}: {type(value)} - {value}")
    
    return train_dataset, eval_dataset


def create_sparse_encoder_model(base_model, model_type, dataset_lang):
    """Create a sparse encoder model."""
    logger.info(f"Creating {model_type} sparse encoder with base model: {base_model}")
    
    if model_type == "splade":
        # Standard SPLADE model
        mlm_transformer = MLMTransformer(base_model, tokenizer_args={"model_max_length": 512})
        splade_pooling = SpladePooling(
            pooling_strategy="max", 
            word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension()
        )
        
        model = SparseEncoder(
            modules=[mlm_transformer, splade_pooling],
            model_card_data=SparseEncoderModelCardData(
                language="en",
                license="apache-2.0",
                model_name=f"SPLADE {base_model} trained on Cornstack {dataset_lang}",
            ),
        )
        
    elif model_type == "inference_free_splade":
        # Inference-free SPLADE model with Router (v5)
        mlm_transformer = MLMTransformer(base_model, tokenizer_args={"model_max_length": 512})
        splade_pooling = SpladePooling(
            pooling_strategy="max", 
            word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension()
        )
        
        # Use v5 Router.for_query_document method
        router = Router.for_query_document(
            query_modules=[SparseStaticEmbedding(tokenizer=mlm_transformer.tokenizer, frozen=False)],
            document_modules=[mlm_transformer, splade_pooling],
        )
        
        model = SparseEncoder(
            modules=[router],
            model_card_data=SparseEncoderModelCardData(
                language="en",
                license="apache-2.0",
                model_name=f"Inference-free SPLADE {base_model} trained on Cornstack {dataset_lang}",
            ),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model


def create_loss_function(model, model_type):
    """Create the loss function for training."""
    hyperparams = MODEL_HYPERPARAMS[model_type]
    
    loss = SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=hyperparams["query_regularizer_weight"],
        document_regularizer_weight=hyperparams["document_regularizer_weight"],
    )
    
    return loss


def create_training_arguments(run_name, model_type, config_name, output_dir):
    """Create training arguments with v5 features."""
    config = TRAINING_CONFIGS[config_name]
    hyperparams = MODEL_HYPERPARAMS[model_type]
    
    # Prepare v5-specific arguments
    learning_rate_mapping = None
    router_mapping = None
    
    if model_type == "inference_free_splade":
        learning_rate_mapping = {r"SparseStaticEmbedding\.weight": hyperparams["static_embedding_lr"]}
        router_mapping = {"query": "query", "document": "document"}
    
    args = SparseEncoderTrainingArguments(
        # Output and run configuration
        output_dir=f"{output_dir}/{run_name}",
        run_name=run_name,
        
        # Training parameters
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        
        # Learning rate configuration
        learning_rate=hyperparams["learning_rate"],
        learning_rate_mapping=learning_rate_mapping,
        warmup_ratio=0.1,
        
        # Hardware optimization
        fp16=False,
        bf16=True,  # Better for modern GPUs
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Important for contrastive learning
        
        # v5 Router configuration
        router_mapping=router_mapping,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=3,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # v5 Data handling
        dataloader_drop_last=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # Tracking
        report_to=["wandb"],
    )
    
    return args


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train sparse encoder with Sentence Transformers v5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--dataset", choices=list(CORNSTACK_DATASETS.keys()), default="python",
                       help="Programming language dataset to use")
    parser.add_argument("--model_type", choices=["splade", "inference_free_splade"], default="splade",
                       help="Type of sparse encoder model")
    parser.add_argument("--base_model", choices=list(BASE_MODELS.keys()), default="general",
                       help="Base transformer model")
    parser.add_argument("--config", choices=list(TRAINING_CONFIGS.keys()), default="development",
                       help="Training configuration preset")
    parser.add_argument("--eval_config", choices=list(EVALUATION_DATASETS.keys()), default="general",
                       help="Evaluation dataset configuration")
    parser.add_argument("--output_dir", type=str, default="./models",
                       help="Output directory for trained models")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push trained model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str,
                       help="Model ID for Hugging Face Hub")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Build configuration
    dataset_name = CORNSTACK_DATASETS[args.dataset]
    base_model_name = BASE_MODELS[args.base_model]
    training_config = TRAINING_CONFIGS[args.config]
    eval_datasets = EVALUATION_DATASETS[args.eval_config]
    
    # Create run name
    run_name = f"sparse-encoder-v5-{args.dataset}-{args.model_type}-{args.base_model}-{args.config}"
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="sparse-encoder-v5",
            name=run_name,
            config={
                "dataset": args.dataset,
                "dataset_name": dataset_name,
                "model_type": args.model_type,
                "base_model": args.base_model,
                "base_model_name": base_model_name,
                "training_config": args.config,
                **training_config,
            }
        )
    
    logger.info("=" * 80)
    logger.info("SPARSE ENCODER TRAINING WITH SENTENCE TRANSFORMERS V5")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset} ({dataset_name})")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Base model: {args.base_model} ({base_model_name})")
    logger.info(f"Training config: {args.config}")
    logger.info(f"Run name: {run_name}")
    logger.info("=" * 80)
    
    try:
        # 1. Load dataset
        logger.info("Step 1: Loading Dataset")
        train_dataset, eval_dataset = load_cornstack_dataset(
            dataset_name, 
            training_config["max_samples"]
        )
        
        # 2. Create model
        logger.info("Step 2: Creating Model")
        model = create_sparse_encoder_model(base_model_name, args.model_type, args.dataset)
        
        # 3. Create loss
        logger.info("Step 3: Creating Loss Function")
        loss = create_loss_function(model, args.model_type)
        
        # 4. Create training arguments
        logger.info("Step 4: Creating Training Arguments")
        training_args = create_training_arguments(run_name, args.model_type, args.config, args.output_dir)
        
        # 5. Create evaluator
        logger.info("Step 5: Creating Evaluator")
        evaluator = SparseNanoBEIREvaluator(dataset_names=eval_datasets, batch_size=16)
        
        # 6. Create trainer
        logger.info("Step 6: Creating Trainer")
        trainer = SparseEncoderTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
        )
        
        # 7. Train
        logger.info("Step 7: Starting Training")
        trainer.train()
        
        # 8. Final evaluation
        logger.info("Step 8: Final Evaluation")
        final_results = evaluator(model)
        logger.info(f"Final results: {final_results}")
        
        if not args.no_wandb:
            wandb.log({"final_evaluation": final_results})
        
        # 9. Save model
        logger.info("Step 9: Saving Model")
        final_model_path = f"{args.output_dir}/{run_name}/final"
        model.save_pretrained(final_model_path)
        logger.info(f"Model saved to {final_model_path}")
        
        # 10. Test model with v5 features
        logger.info("Step 10: Testing Model")
        test_queries = [
            f"How to implement a function in {args.dataset}?",
            f"Best practices for {args.dataset} programming",
            f"Error handling in {args.dataset} code"
        ]
        
        test_documents = [
            f"def example_function(): # Example {args.dataset} function",
            f"# Best practices for {args.dataset} development", 
            f"try: # Error handling in {args.dataset}"
        ]
        
        # Use v5 encode_query() and encode_document() methods
        query_embeddings = model.encode_query(test_queries)
        doc_embeddings = model.encode_document(test_documents)
        similarities = model.similarity(query_embeddings, doc_embeddings)
        
        logger.info("Test results:")
        test_results = {}
        for i, query in enumerate(test_queries):
            logger.info(f"Query: {query}")
            query_results = {}
            for j, sim in enumerate(similarities[i]):
                logger.info(f"  Doc {j}: {sim:.4f}")
                query_results[f"doc_{j}_similarity"] = float(sim)
            test_results[f"query_{i}"] = query_results
        
        # Check sparsity statistics
        stats = model.sparsity(query_embeddings)
        logger.info(f"Sparsity: {stats['sparsity_ratio']:.2%}, Active dims: {stats['active_dims']:.1f}")
        
        if not args.no_wandb:
            wandb.log({
                "test_results": test_results,
                "sparsity_ratio": stats['sparsity_ratio'],
                "average_active_dims": stats['active_dims']
            })
        
        # 11. Push to hub if requested
        if args.push_to_hub:
            logger.info("Step 11: Pushing to Hub")
            hub_model_id = args.hub_model_id or run_name
            model.push_to_hub(hub_model_id)
            logger.info(f"Model pushed to Hub as {hub_model_id}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
