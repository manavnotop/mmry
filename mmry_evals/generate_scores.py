import argparse
import json
from collections import defaultdict

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Generate final scores from evaluation metrics"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/mmry_evaluation_metrics.json",
        help="Path to the evaluation metrics file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/mmry_final_scores.json",
        help="Path to save the final scores",
    )

    args = parser.parse_args()

    # Load the evaluation metrics
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # Prepare data for analysis
    categories = defaultdict(
        lambda: {"bleu_score": [], "f1_score": [], "llm_score": []}
    )
    all_scores = {"bleu_score": [], "f1_score": [], "llm_score": []}

    for key, items in data.items():
        for item in items:
            category = item["category"]
            for metric in ["bleu_score", "f1_score", "llm_score"]:
                value = item[metric]
                categories[category][metric].append(value)
                all_scores[metric].append(value)

    # Calculate mean scores per category
    category_data = []
    for category, metrics in categories.items():
        row = {"category": category}
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = sum(values) / len(values) if values else 0
            row[f"{metric}_count"] = len(values)
        category_data.append(row)

    # Create DataFrame for categories
    df_categories = pd.DataFrame(category_data)
    df_categories.set_index("category", inplace=True)

    # Calculate overall scores
    overall_scores = {}
    for metric in ["bleu_score", "f1_score", "llm_score"]:
        overall_scores[metric] = (
            sum(all_scores[metric]) / len(all_scores[metric])
            if all_scores[metric]
            else 0
        )

    # Print results
    print("Mean Scores Per Category:")
    print(df_categories.round(4))
    print("\nOverall Mean Scores:")
    for metric, score in overall_scores.items():
        print(f"{metric}: {score:.4f}")

    # Save to file
    with open(args.output_file, "w") as f:
        json.dump({"categories": category_data, "overall": overall_scores}, f, indent=4)

    print(f"\nFinal scores saved to {args.output_file}")


if __name__ == "__main__":
    main()
