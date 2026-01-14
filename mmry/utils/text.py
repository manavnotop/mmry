import re


def clean_summary(summary: str) -> str:
    """
    Clean summary text by removing markdown formatting and converting
    numbered lists to simple sentences for better vector search similarity.
    """
    # Remove markdown bold/italic
    summary = re.sub(r"\*\*([^*]+)\*\*", r"\1", summary)
    summary = re.sub(r"\*([^*]+)\*", r"\1", summary)
    summary = re.sub(r"__([^_]+)__", r"\1", summary)
    summary = re.sub(r"_([^_]+)_", r"\1", summary)

    # Convert numbered/bulleted lists to sentences
    # Match patterns like "1. text" or "- text" or "• text"
    lines = summary.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove list markers (1., 2., -, •, etc.)
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^[-•]\s*", "", line)
        cleaned_lines.append(line)

    # Join with periods and spaces for better semantic search
    cleaned = ". ".join(cleaned_lines)
    # Remove extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
