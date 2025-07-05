#!/usr/bin/env python3
"""
Train a Sparse Encoder with Sentence Transformers v5 using the Cornstack Dataset

This script demonstrates how to train a sparse encoder model using the latest
Sentence Transformers v5 library with the high-quality Cornstack dataset from Nomic AI.

The Cornstack dataset contains high-quality contrastive (text, code) pairs for training
embedding models and re-rankers for code retrieval via contrastive learning.

Updated for Sentence Transformers v5 migration guide compatibility:
- Uses encode_query() and encode_document() methods instead of encode() with task parameters
- Router module usage updated to v5 syntax with Router.for_query_document()
- Added support for new v5 features like truncate_dim
- Demonstrates migration from v4 encode_multi_process to v5 encode with device parameter
"""

import logging
import os
import wandb

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset, DatasetDict
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

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cornstack_dataset(dataset_name="nomic-ai/cornstack-python-v1", max_samples=1000):
    """
    Load and prepare the Cornstack dataset for training.
    
    Args:
        dataset_name: Name of the Cornstack dataset to load
        max_samples: Maximum number of samples to use for training
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading Cornstack dataset: {dataset_name}")
    
    # Load the dataset with limited samples directly
    if max_samples:
        full_dataset = load_dataset(dataset_name, split=f"train", data_files=["shard-00000.jsonl.gz"])
        full_dataset = full_dataset.select(range(min(max_samples, len(full_dataset))))
        logger.info(f"Loading first {max_samples} samples from dataset")
    else:
        full_dataset = load_dataset(dataset_name, split=f"train", data_files=["shard-00000.jsonl.gz"])
        logger.info("Loading full dataset")
    
    # Split into train and eval
    dataset_dict = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    logger.info(f"Dataset features: {train_dataset.features}")
    
    # Print a sample to understand the data structure
    sample = train_dataset[0]
    logger.info("Sample data structure:")
    for key, value in sample.items():
        if isinstance(value, str):
            logger.info(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {type(value)} - {value}")
    
    return train_dataset, eval_dataset

def create_sparse_encoder_model(base_model="answerdotai/ModernBERT-base", model_type="splade"):
    """
    Create a sparse encoder model.
    
    Args:
        base_model: Base transformer model to use
        model_type: Type of sparse encoder ("splade" or "inference_free_splade")
    
    Returns:
        SparseEncoder model
    """
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
                model_name=f"SPLADE {base_model} trained on Cornstack dataset",
            ),
        )
        
    elif model_type == "inference_free_splade":
        # Inference-free SPLADE model with Router (updated for v5)
        mlm_transformer = MLMTransformer(base_model, tokenizer_args={"model_max_length": 512})
        splade_pooling = SpladePooling(
            pooling_strategy="max", 
            word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension()
        )
        
        # Use the new Router.for_query_document method from v5
        router = Router.for_query_document(
            query_modules=[SparseStaticEmbedding(tokenizer=mlm_transformer.tokenizer, frozen=False)],
            document_modules=[mlm_transformer, splade_pooling],
        )
        
        model = SparseEncoder(
            modules=[router],
            model_card_data=SparseEncoderModelCardData(
                language="en",
                license="apache-2.0",
                model_name=f"Inference-free SPLADE {base_model} trained on Cornstack dataset",
            ),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model

def create_loss_function(model, model_type="splade"):
    """
    Create the loss function for training.
    
    Args:
        model: The sparse encoder model
        model_type: Type of model being trained
    
    Returns:
        Loss function
    """
    logger.info(f"Creating loss function for {model_type}")
    
    if model_type == "splade":
        # Standard SPLADE loss
        loss = SpladeLoss(
            model=model,
            loss=SparseMultipleNegativesRankingLoss(model=model),
            query_regularizer_weight=5e-5,  # Weight for query sparsity regularization
            document_regularizer_weight=3e-5,  # Weight for document sparsity regularization
        )
    elif model_type == "inference_free_splade":
        # Inference-free SPLADE loss (no query regularization since queries use static embeddings)
        loss = SpladeLoss(
            model=model,
            loss=SparseMultipleNegativesRankingLoss(model=model),
            query_regularizer_weight=0,  # No regularization for static embeddings
            document_regularizer_weight=3e-4,  # Higher regularization for documents
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return loss

def create_training_arguments(run_name, model_type="splade", num_epochs=3, batch_size=16):
    """
    Create training arguments.
    
    Args:
        run_name: Name for the training run
        model_type: Type of model being trained
        num_epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        SparseEncoderTrainingArguments
    """
    logger.info("Creating training arguments")
    
    args = SparseEncoderTrainingArguments(
        # Required parameter
        output_dir=f"models/{run_name}",
        
        # Training parameters
        num_train_epochs=num_epochs,  # Configurable
        per_device_train_batch_size=batch_size,  # Configurable
        per_device_eval_batch_size=batch_size,  # Configurable
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Use BF16 instead for RTX 4090
        bf16=True,  # RTX 4090 supports BF16 for better performance
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Important for contrastive learning
        
        # Router mapping for Cornstack dataset (v5 compatible)
        router_mapping={"query": "query", "document": "document"} if model_type == "inference_free_splade" else {},
        
        # Learning rate mapping for inference-free SPLADE (v5 feature)
        learning_rate_mapping={r"SparseStaticEmbedding\.weight": 1e-3} if model_type == "inference_free_splade" else {},
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=300,  # More frequent for test runs
        save_steps=300,
        save_total_limit=3,
        logging_steps=10,  # More frequent logging
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Tracking
        run_name=run_name,  # Will be used in W&B
        report_to=["wandb"],  # Enable wandb logging
        
        # Additional v5 features
        dataloader_drop_last=False,  # Keep all samples
        remove_unused_columns=False,  # Keep all dataset columns
        
        # Performance optimizations for your 32-core system
        dataloader_num_workers=8,  # Reduced to avoid tokenizer forking issues
        dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
        gradient_checkpointing=False,  # Set to True if you run out of GPU memory
    )
    
    return args

def main():
    """Main training function."""
    
    # Configuration (can be overridden by environment variables)
    BASE_MODEL = os.getenv("BASE_MODEL", "answerdotai/ModernBERT-base")
    MODEL_TYPE = os.getenv("MODEL_TYPE", "splade")  # or "inference_free_splade"
    DATASET_NAME = "nomic-ai/cornstack-python-v1"
    MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "50000"))  # Set to None for full dataset
    RUN_NAME = os.getenv("RUN_NAME", f"sparse-encoder-cornstack-{MODEL_TYPE}-{BASE_MODEL.split('/')[-1]}")
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "2"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    
    # Initialize wandb
    wandb.init(
        project="sparse-encoder-cornstack",
        name=RUN_NAME,
        config={
            "base_model": BASE_MODEL,
            "model_type": MODEL_TYPE,
            "dataset": DATASET_NAME,
            "max_samples": MAX_SAMPLES,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
        }
    )
    
    logger.info("Starting sparse encoder training with Cornstack dataset")
    logger.info(f"Configuration:")
    logger.info(f"  Base model: {BASE_MODEL}")
    logger.info(f"  Model type: {MODEL_TYPE}")
    logger.info(f"  Dataset: {DATASET_NAME}")
    logger.info(f"  Max samples: {MAX_SAMPLES}")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Run name: {RUN_NAME}")
    
    # 1. Load and prepare the dataset
    train_dataset, eval_dataset = load_cornstack_dataset(DATASET_NAME, MAX_SAMPLES)
    
    # 2. Create the sparse encoder model
    model = create_sparse_encoder_model(BASE_MODEL, MODEL_TYPE)
    
    # 3. Create the loss function
    loss = create_loss_function(model, MODEL_TYPE)
    
    # 4. Create training arguments
    args = create_training_arguments(RUN_NAME, MODEL_TYPE, NUM_EPOCHS, BATCH_SIZE)
    
    # 5. Create evaluator (optional but recommended)
    logger.info("Creating evaluator")
    dev_evaluator = SparseNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"], 
        batch_size=16
    )
    
    # 6. Create trainer and start training
    logger.info("Creating trainer")
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    
    # 7. Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # 8. Evaluate the final model
    logger.info("Evaluating final model...")
    final_results = dev_evaluator(model)
    logger.info(f"Final evaluation results: {final_results}")
    
    # Log final results to wandb
    wandb.log({"final_evaluation": final_results})
    
    # 9. Save the trained model
    final_model_path = f"models/{RUN_NAME}/final"
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(final_model_path)
    
    # 10. Optional: Push to Hugging Face Hub
    # Uncomment the following lines if you want to push to the Hub
    # logger.info("Pushing model to Hugging Face Hub...")
    # model.push_to_hub(RUN_NAME)
    
    logger.info("Training completed successfully!")
    
    # 11. Test the trained model
    logger.info("Testing the trained model...")
    test_queries = [
        "How to read a file in Python?",
        "Function to load stop words from file",
        "Create a hash table for word storage"
    ]
    
    test_documents = [
        "def read_file(filename): with open(filename, 'r') as f: return f.read()",
        "def load_stop_words(file_path): with open(file_path) as f: return f.read().split()",
        "class HashTable: def __init__(self, size): self.table = [[] for _ in range(size)]"
    ]
    
    # Use the new v5 encode_query() and encode_document() methods
    # These methods automatically handle prompt selection and task routing
    # They are the recommended way to encode in v5 instead of encode() with task parameters
    query_embeddings = model.encode_query(test_queries)
    doc_embeddings = model.encode_document(test_documents)
    
    # Optional: You can also use additional v5 features like truncate_dim
    # query_embeddings = model.encode_query(test_queries, truncate_dim=256)
    # doc_embeddings = model.encode_document(test_documents, truncate_dim=256)
    
    similarities = model.similarity(query_embeddings, doc_embeddings)
    
    logger.info("Test results:")
    test_results = {}
    for i, query in enumerate(test_queries):
        logger.info(f"Query: {query}")
        query_results = {}
        for j, doc in enumerate(test_documents):
            similarity_score = similarities[i][j]
            logger.info(f"  Doc {j}: {similarity_score:.4f} - {doc[:50]}...")
            query_results[f"doc_{j}_similarity"] = similarity_score
        test_results[f"query_{i}"] = query_results
        logger.info("")
    
    # Log test results to wandb
    wandb.log({"test_results": test_results})
    
    # Check sparsity statistics
    stats = model.sparsity(query_embeddings)
    logger.info(f"Query embeddings sparsity: {stats['sparsity_ratio']:.2%}")
    logger.info(f"Average active dimensions: {stats['active_dims']:.2f}")
    
    # Log sparsity stats to wandb
    wandb.log({
        "sparsity_ratio": stats['sparsity_ratio'],
        "average_active_dims": stats['active_dims']
    })
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
