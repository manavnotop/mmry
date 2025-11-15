import argparse
import os

from mmry_evals.src.mmry_eval_manager import MmryEvalManager


def main():
    parser = argparse.ArgumentParser(description="Run mmry memory experiments")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/locomo10_rag.json",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/mmry_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of memories to retrieve"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for mmry",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--embed_model_type",
        type=str,
        default="local",
        choices=["local", "openrouter"],
        help="Type of embedding model to use",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="mmry_evaluation",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--user_id", type=str, default="mmry_eval", help="User ID for evaluation"
    )

    args = parser.parse_args()

    print(
        f"Running mmry evaluation with top_k: {args.top_k}, threshold: {args.similarity_threshold}, embed_model: {args.embed_model}, embed_model_type: {args.embed_model_type}"
    )

    # Create mmry evaluation manager
    mmry_manager = MmryEvalManager(
        data_path=args.data_path,
        k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        embed_model=args.embed_model,
        embed_model_type=args.embed_model_type,
        collection_name=args.collection_name,
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Process all conversations
    mmry_manager.process_all_conversations(args.output_path, user_id=args.user_id)

    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
