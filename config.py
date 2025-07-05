"""
Configuration file for sparse encoder training with different Cornstack datasets and model types.
Updated for Sentence Transformers v5 compatibility.
"""

# Available Cornstack datasets
CORNSTACK_DATASETS = {
    "python": "nomic-ai/cornstack-python-v1",
    "java": "nomic-ai/cornstack-java-v1", 
    "javascript": "nomic-ai/cornstack-javascript-v1",
    "typescript": "nomic-ai/cornstack-typescript-v1",
    "go": "nomic-ai/cornstack-go-v1",
    "rust": "nomic-ai/cornstack-rust-v1",
    "cpp": "nomic-ai/cornstack-cpp-v1",
}

# Base models optimized for different use cases
BASE_MODELS = {
    "general": "answerdotai/ModernBERT-base",  # Best overall performance
    "code": "microsoft/codebert-base",  # Optimized for code understanding
    "small": "distilbert/distilbert-base-uncased",  # Faster training/inference
    "large": "google-bert/bert-large-uncased",  # Maximum performance
}

# Training configurations for different scenarios
TRAINING_CONFIGS = {
    "quick_test": {
        "max_samples": 10000,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "eval_steps": 500,
        "save_steps": 500,
        "description": "Fast testing configuration"
    },
    "development": {
        "max_samples": 50000,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 16,
        "eval_steps": 1000,
        "save_steps": 1000,
        "description": "Development and experimentation"
    },
    "production": {
        "max_samples": None,  # Use full dataset
        "num_train_epochs": 5,
        "per_device_train_batch_size": 32,
        "eval_steps": 2000,
        "save_steps": 2000,
        "description": "Full production training"
    }
}

# Model-specific hyperparameters optimized for v5
MODEL_HYPERPARAMS = {
    "splade": {
        "query_regularizer_weight": 5e-5,
        "document_regularizer_weight": 3e-5,
        "learning_rate": 2e-5,
        "description": "Standard SPLADE model with MLM + SpladePooling"
    },
    "inference_free_splade": {
        "query_regularizer_weight": 0,  # No regularization for static embeddings
        "document_regularizer_weight": 3e-4,
        "learning_rate": 2e-5,
        "static_embedding_lr": 1e-3,  # Higher LR for SparseStaticEmbedding
        "description": "Inference-free SPLADE with Router architecture"
    }
}

# Evaluation datasets for different domains
EVALUATION_DATASETS = {
    "general": ["msmarco", "nfcorpus", "nq"],
    "code": ["msmarco", "nfcorpus"],  # Add code-specific datasets when available
    "minimal": ["nfcorpus"],  # For quick evaluation
    "comprehensive": ["msmarco", "nfcorpus", "nq", "fiqa", "trec-covid"],
}

# v5 specific configurations
V5_FEATURES = {
    "use_truncate_dim": False,  # Enable dimension truncation
    "truncate_dim": 256,  # Dimension to truncate to
    "use_device_map": True,  # Enable device mapping for multi-GPU
    "enable_gradient_checkpointing": False,  # Memory optimization
    "use_bf16": True,  # Use BF16 for better performance on modern GPUs
}
