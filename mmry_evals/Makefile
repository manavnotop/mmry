# mmry Evaluation Makefile

# Ensure results directory exists
RESULTS_DIR = results
DATASET_DIR = dataset

.PHONY: setup clean run-eval eval-metrics help

setup:
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p $(DATASET_DIR)

# Run mmry evaluation on dataset
run-eval: setup
	python -m mmry_evals.run_evals --data_path $(DATASET_DIR)/locomo10_rag.json --output_path $(RESULTS_DIR)/mmry_results.json --top_k 3 --embed_model all-MiniLM-L6-v2 --embed_model_type local

# Run mmry evaluation on dataset with OpenRouter embeddings
run-eval-openrouter: setup
	python -m mmry_evals.run_evals --data_path $(DATASET_DIR)/locomo10_rag.json --output_path $(RESULTS_DIR)/mmry_results_openrouter.json --top_k 3 --embed_model openai/text-embedding-3-small --embed_model_type openrouter

# Evaluate the results using metrics
eval-metrics: setup
	python -m mmry_evals.evals --input_file $(RESULTS_DIR)/mmry_results.json --output_file $(RESULTS_DIR)/mmry_evaluation_metrics.json

# Run full evaluation pipeline
full-eval: setup run-eval eval-metrics
	@echo "Full mmry evaluation pipeline completed!"

# Generate final scores
generate-scores:
	python -m mmry_evals.generate_scores

# Clean results
clean:
	rm -rf $(RESULTS_DIR)/*.json
	@echo "Results cleaned"

# Help
help:
	@echo "Available commands:"
	@echo "  make setup        - Create required directories"
	@echo "  make run-eval     - Run mmry evaluation on dataset"
	@echo "  make eval-metrics - Calculate evaluation metrics for results"
	@echo "  make full-eval    - Run complete evaluation pipeline"
	@echo "  make clean        - Clean results directory"
	@echo "  make help         - Show this help message"