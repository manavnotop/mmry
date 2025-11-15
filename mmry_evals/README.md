# Mmry Evaluation

This directory contains evaluation tools and scripts for the mmry library.

## 📋 Overview

This evaluation pipeline tests the mmry memory system against various conversational datasets to measure:
- Memory recall accuracy
- Response quality
- Performance metrics

## 📁 Project Structure

```
.
├── src/                    # Source code for mmry evaluation
│   └── mmry_eval_manager.py # Mmry evaluation manager
├── metrics/                # Code for evaluation metrics
│   ├── llm_judge.py        # LLM-based answer evaluation
│   └── utils.py            # Utility functions for metrics
├── run_evals.py            # Script to run evaluations
├── evals.py                # Main evaluation script
├── Makefile                # Commands for running evaluations
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

Create a `.env` file with your API keys and configurations:

```
# OpenRouter API key for LLMs and embeddings
OPENROUTER_API_KEY="your-openrouter-api-key"

# Model configuration
MODEL="openai/gpt-3.5-turbo"  # or your preferred model
```

### Running Evaluations

You can run evaluations using the provided Makefile commands:

#### Basic Evaluation

```bash
# Run mmry evaluation
make run-eval

# Calculate metrics for results
make eval-metrics

# Run full pipeline
make full-eval
```

Alternatively, you can run the scripts directly:

```bash
# Run evaluation
python -m mmry_evals.run_evals --data_path dataset/locomo10_rag.json --output_path results/mmry_results.json

# Evaluate results
python -m mmry_evals.evals --input_file results/mmry_results.json --output_file results/mmry_evaluation_metrics.json
```

#### Command-line Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_path` | Path to the dataset | dataset/locomo10_rag.json |
| `--output_path` | Path to save results | results/mmry_results.json |
| `--top_k` | Number of memories to retrieve | 3 |
| `--similarity_threshold` | Similarity threshold for memory matching | 0.7 |
| `--user_id` | User ID for evaluation | mmry_eval |

### 📊 Evaluation Metrics

We use several metrics to evaluate the performance of mmry:

1. **BLEU Score**: Measures the similarity between the model's response and the ground truth
2. **F1 Score**: Measures the harmonic mean of precision and recall
3. **LLM Score**: A binary score (0 or 1) determined by an LLM judge evaluating the correctness of responses
4. **Token Consumption**: Number of tokens required to generate final answer
5. **Latency**: Time required during search and to generate response

## 📈 Interpreting Results

The evaluation pipeline generates two types of output files:

1. **Results file** (`mmry_results.json`): Contains the raw responses from the evaluation
2. **Metrics file** (`mmry_evaluation_metrics.json`): Contains aggregated metrics for analysis

## 🛠️ Troubleshooting

- Make sure you have Qdrant running locally on `http://localhost:6333`
- Verify that your API keys are properly configured
- Check that the dataset files exist in the specified locations

## 📚 Citation

If you use this evaluation pipeline in your research, please acknowledge the mmry library.