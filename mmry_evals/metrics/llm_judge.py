import argparse
import json
import os
from collections import defaultdict

import numpy as np
import requests

from mmry_evals.metrics.utils import extract_json
from mmry_evals.prompts import LLM_JUDGE_PROMPT


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4o-mini",  # Use GPT-4o-mini via OpenRouter
            "messages": [
                {
                    "role": "user",
                    "content": LLM_JUDGE_PROMPT.format(
                        question=question,
                        gold_answer=gold_answer,
                        generated_answer=generated_answer,
                    ),
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        },
    )

    response.raise_for_status()
    result = response.json()

    content = result["choices"][0]["message"]["content"]
    extracted_json = extract_json(content)
    try:
        label = json.loads(extracted_json)["label"]
        return 1 if label == "CORRECT" else 0
    except:
        # If JSON parsing fails, return 0 (incorrect)
        return 0


def main():
    """Main function to evaluate mmry results using LLM judge."""
    parser = argparse.ArgumentParser(
        description="Evaluate mmry results using LLM judge"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/mmry_results.json",
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            # Store the results
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            # Print current accuracy for all categories
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # Only print if there are results for this category
                    print(
                        f"  Category {cat}: {np.mean(results):.4f} ({sum(results)}/{len(results)})"
                    )
            print("------------------------------------------")
        index += 1

    # Save final results
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Print final summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()
