fix:
	uv run ruff check --select I --fix .
	uv run ruff format

test:
	-docker rm -f qdrant-test >/dev/null 2>&1 || true
	docker run -d --name qdrant-test -p 6333:6333 qdrant/qdrant >/dev/null
	sleep 1

	uv run test.py

	docker rm -f qdrant-test >/dev/null

# Mmry Evaluation Commands
eval-setup:
	mkdir -p results dataset

eval-run:
	uv run -m mmry_evals.run_evals --top_k 3 --embed_model all-MiniLM-L6-v2 --embed_model_type local

eval-run-openrouter:
	uv run -m mmry_evals.run_evals --top_k 3 --embed_model openai/text-embedding-3-small --embed_model_type openrouter --output_path results/mmry_results_openrouter.json

eval-metrics:
	uv run -m mmry_evals.evals

eval-metrics-openrouter:
	uv run -m mmry_evals.evals --input_file results/mmry_results_openrouter.json --output_file results/mmry_evaluation_metrics_openrouter.json

eval-scores:
	uv run -m mmry_evals.generate_scores --input_file results/mmry_evaluation_metrics.json --output_file results/mmry_final_scores.json

eval-scores-openrouter:
	uv run -m mmry_evals.generate_scores --input_file results/mmry_evaluation_metrics_openrouter.json --output_file results/mmry_final_scores_openrouter.json

eval-full: eval-setup eval-run eval-metrics eval-scores
	@echo "Complete mmry evaluation pipeline finished!"

eval-full-openrouter: eval-setup eval-run-openrouter eval-metrics-openrouter eval-scores
	@echo "Complete mmry evaluation pipeline with OpenRouter embeddings finished!"