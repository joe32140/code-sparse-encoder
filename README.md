# Sparse Encoder Training with Sentence Transformers v5 and Cornstack Dataset

This repository contains clean, production-ready scripts for training sparse encoder models using **Sentence Transformers v5** with the high-quality Cornstack dataset from Nomic AI.

## 🚀 What's New in v5

- **Explicit Task Routing**: Use `encode_query()` and `encode_document()` instead of `encode()` with task parameters
- **Enhanced Router Architecture**: Improved `Router.for_query_document()` for inference-free models
- **Advanced Training Arguments**: New `SparseEncoderTrainingArguments` with v5-specific features
- **Better Batch Sampling**: Optimized `BatchSamplers.NO_DUPLICATES` for contrastive learning
- **Improved Evaluation**: Enhanced `SparseNanoBEIREvaluator` with comprehensive metrics
- **Performance Optimizations**: BF16 support, gradient checkpointing, and multi-device training

## 📊 Dataset Overview

The Cornstack dataset is a large-scale, high-quality contrastive dataset for code retrieval spanning multiple programming languages:

- **Consistency Filtering**: Eliminates noisy positives for cleaner training data
- **Hard Negatives**: Enriched with mined hard negatives for better contrastive learning
- **Multi-Language Support**: Python, Java, JavaScript, TypeScript, Go, Rust, and C++
- **High Quality**: Curated specifically for code retrieval and embedding tasks

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/joe32140/code-sparse-encoder.git
cd code-sparse-encoder
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import sentence_transformers; print(f'Sentence Transformers version: {sentence_transformers.__version__}')"
```

## 🚀 Quick Start

### Working Training Script (Recommended)

Use the proven working training script:

```bash
# Basic training with default settings
python train_sparse_encoder_cornstack.py

# Configure via environment variables
BASE_MODEL=answerdotai/ModernBERT-base MODEL_TYPE=splade MAX_SAMPLES=50000 python train_sparse_encoder_cornstack.py

# Inference-free SPLADE training
MODEL_TYPE=inference_free_splade python train_sparse_encoder_cornstack.py
```

### Modern v5 Training Script

Use the configurable v5 training script:

```bash
# Quick test on Python dataset
python train_sparse_encoder_v5.py --dataset python --model_type splade --config quick_test

# Development training with inference-free SPLADE
python train_sparse_encoder_v5.py --dataset java --model_type inference_free_splade --config development

# Production training
python train_sparse_encoder_v5.py --dataset python --model_type splade --config production --push_to_hub
```

## 📋 Available Options

### **Datasets**
| Language | Dataset ID | Description |
|----------|------------|-------------|
| `python` | `nomic-ai/cornstack-python-v1` | Python code dataset |
| `java` | `nomic-ai/cornstack-java-v1` | Java code dataset |
| `javascript` | `nomic-ai/cornstack-javascript-v1` | JavaScript code dataset |
| `typescript` | `nomic-ai/cornstack-typescript-v1` | TypeScript code dataset |
| `go` | `nomic-ai/cornstack-go-v1` | Go code dataset |
| `rust` | `nomic-ai/cornstack-rust-v1` | Rust code dataset |
| `cpp` | `nomic-ai/cornstack-cpp-v1` | C++ code dataset |

### **Model Types**
| Type | Description | Use Case |
|------|-------------|----------|
| `splade` | Standard SPLADE with MLM + SpladePooling | General purpose, balanced performance |
| `inference_free_splade` | Router-based with SparseStaticEmbedding | Fast query encoding, production systems |

### **Base Models**
| Key | Model | Best For |
|-----|-------|----------|
| `general` | `answerdotai/ModernBERT-base` | Best overall performance |
| `code` | `microsoft/codebert-base` | Code understanding tasks |
| `small` | `distilbert/distilbert-base-uncased` | Fast training/inference |
| `large` | `google-bert/bert-large-uncased` | Maximum performance |

### **Training Configurations**
| Config | Samples | Epochs | Batch Size | Use Case |
|--------|---------|--------|------------|----------|
| `quick_test` | 10K | 1 | 8 | Fast testing and debugging |
| `development` | 50K | 2 | 16 | Development and experimentation |
| `production` | Full dataset | 5 | 32 | Production deployment |

## 🏗️ Model Architectures

### Standard SPLADE Model
```
Input Text → MLMTransformer → SpladePooling → Sparse Embedding
```
- **Components**: MLMTransformer + SpladePooling
- **Output**: Sparse embeddings (vocabulary-sized)
- **Use Case**: General purpose sparse retrieval

### Inference-Free SPLADE Model
```
Query: Input → SparseStaticEmbedding → Sparse Embedding
Document: Input → MLMTransformer → SpladePooling → Sparse Embedding
```
- **Components**: Router with separate query/document paths
- **Query Path**: SparseStaticEmbedding (fast, lightweight)
- **Document Path**: MLMTransformer + SpladePooling
- **Use Case**: Production systems requiring fast query encoding

## 📊 Comprehensive Evaluation

Use the evaluation script for detailed analysis:

```bash
# Complete evaluation suite
python evaluate_sparse_encoder_v5.py --model_path ./models/my-model --eval_type all

# Specific evaluations
python evaluate_sparse_encoder_v5.py --model_path naver/splade-v3 --eval_type sparsity beir

# Custom dataset evaluation
python evaluate_sparse_encoder_v5.py --model_path ./models/my-model --eval_type custom --custom_dataset nomic-ai/cornstack-python-v1

# Generate detailed report
python evaluate_sparse_encoder_v5.py --model_path ./models/my-model --eval_type all --output_path ./evaluation_report.json
```

### Evaluation Types
- **🔍 Sparsity Analysis**: Detailed sparsity statistics and characteristics
- **📊 BEIR Evaluation**: Performance on standard retrieval benchmarks
- **⚡ Performance Benchmarking**: Speed and throughput analysis
- **📋 Custom Dataset**: Evaluation on specific datasets

## 💻 Example Usage (v5 API)

```python
from sentence_transformers import SparseEncoder

# Load trained model
model = SparseEncoder("./models/your-trained-model")

# v5 API: Use explicit encoding methods
queries = ["How to read a file in Python?"]
documents = ["def read_file(filename): with open(filename, 'r') as f: return f.read()"]

# Encode with v5 methods
query_embeddings = model.encode_query(queries)
doc_embeddings = model.encode_document(documents)

# Compute similarity (dot product by default for sparse models)
similarities = model.similarity(query_embeddings, doc_embeddings)
print(f"Similarity: {similarities[0][0]:.4f}")

# Analyze sparsity
stats = model.sparsity(query_embeddings)
print(f"Sparsity: {stats['sparsity_ratio']:.2%}")
print(f"Active dimensions: {stats['active_dims']:.1f}")
```

## ⚙️ Configuration System

The `config.py` file provides comprehensive configuration options:

```python
# Example configuration usage
from config import CORNSTACK_DATASETS, BASE_MODELS, TRAINING_CONFIGS

# Access configurations
dataset_name = CORNSTACK_DATASETS["python"]
base_model = BASE_MODELS["general"]
training_config = TRAINING_CONFIGS["development"]
```

## 🎯 Training Tips & Best Practices

### **Environment Variables (for train_sparse_encoder_cornstack.py)**
```bash
export BASE_MODEL="answerdotai/ModernBERT-base"
export MODEL_TYPE="splade"  # or "inference_free_splade"
export MAX_SAMPLES="50000"  # or leave unset for full dataset
export NUM_EPOCHS="2"
export BATCH_SIZE="32"
export RUN_NAME="my-custom-run"
```

### **Sparsity Optimization**
- Monitor both performance metrics AND sparsity ratios
- Tune `query_regularizer_weight` and `document_regularizer_weight`
- Target >99% sparsity for optimal efficiency

### **Performance Optimization**
- Use `BatchSamplers.NO_DUPLICATES` for contrastive learning
- Enable BF16 on modern GPUs: `bf16=True`
- Use larger batch sizes when possible (32+ recommended)
- Consider gradient checkpointing for memory efficiency

### **v5 Specific Tips**
- Use `encode_query()` and `encode_document()` instead of `encode()` with task parameters
- Use `learning_rate_mapping` for fine-grained learning rate control
- Enable `router_mapping` for inference-free models

## 🔧 Troubleshooting

### **Common Issues & Solutions**

| Issue | Solution |
|-------|----------|
| **GPU Memory Issues** | Reduce batch size, enable gradient checkpointing, use FP16/BF16 |
| **Low Sparsity** | Increase regularization weights, check loss configuration |
| **Poor Performance** | Try different base models, increase training data, tune hyperparameters |
| **Slow Training** | Use smaller models for testing, optimize batch size, enable mixed precision |
| **Tokenizer Warnings** | Set `TOKENIZERS_PARALLELISM=false` environment variable |

## 📁 Repository Structure

```
code-sparse-encoder/
├── train_sparse_encoder_cornstack.py    # Working training script (recommended)
├── train_sparse_encoder_v5.py           # Configurable v5 training script
├── evaluate_sparse_encoder_v5.py        # Comprehensive evaluation script
├── config.py                            # Configuration for datasets and models
├── requirements.txt                      # Dependencies for v5
└── README.md                            # This file
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Add tests if applicable
5. Submit a pull request

## 📄 Citation

If you use this code or the Cornstack dataset, please cite:

```bibtex
@article{suresh2024cornstack,
  title={CoRNStack: High-Quality Contrastive Data for Better Code Retrieval and Reranking},
  author={Suresh, Tarun and Reddy, Revanth Gangi and Xu, Yifei and Nussbaum, Zach and Mulyar, Andriy and Duderstadt, Brandon and Ji, Heng},
  journal={arXiv preprint arXiv:2412.01007},
  year={2024}
}

@software{sentence_transformers_v5,
  title={Sentence Transformers v5: Modern Sparse and Dense Embedding Training},
  author={Reimers, Nils and Gurevych, Iryna},
  year={2024},
  url={https://github.com/UKPLab/sentence-transformers}
}
```

## 📜 License

This project is licensed under the Apache 2.0 License.

## 🔗 Links

- **Sentence Transformers v5**: [Documentation](https://sbert.net/)
- **Cornstack Dataset**: [Hugging Face](https://huggingface.co/datasets/nomic-ai/cornstack-python-v1)
- **SPLADE Paper**: [arXiv](https://arxiv.org/abs/2109.10086)
- **Issues & Support**: [GitHub Issues](https://github.com/joe32140/code-sparse-encoder/issues)
