import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import requests
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from tqdm import tqdm

from mmry.client import MemoryClient
from mmry.config import LLMConfig, MemoryConfig, VectorDBConfig
from mmry_evals.prompts import MMRY_EVAL_SYSTEM_PROMPT

load_dotenv()

PROMPT = """
# Question:
{{QUESTION}}

# Context:
{{CONTEXT}}

# Short answer:
"""


class MmryEvalManager:
    def __init__(
        self,
        data_path="dataset/locomo10_rag.json",
        k=3,  # Number of memories to retrieve
        similarity_threshold=0.7,
        embed_model="all-MiniLM-L6-v2",
        embed_model_type="local",  # Can be "local" or "openrouter"
        collection_name="mmry_evaluation",
    ):
        # Initialize mmry client with configuration
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            self.llm_config = LLMConfig(api_key=api_key)
        else:
            self.llm_config = None

        # Use a separate collection for evaluation
        vector_config = VectorDBConfig(
            url="http://localhost:6333",
            collection_name=collection_name,
            embed_model=embed_model,
            embed_model_type=embed_model_type,  # Can be "local" or "openrouter"
            embed_api_key=api_key if embed_model_type == "openrouter" else None,
        )

        memory_config = MemoryConfig(
            llm_config=self.llm_config,
            vector_db_config=vector_config,
            similarity_threshold=similarity_threshold,
            log_path="mmry_evaluation_log.jsonl",
        )

        self.client = MemoryClient.from_config(memory_config)
        self.data_path = data_path
        self.k = k  # Number of memories to retrieve during search
        self.model = os.getenv("MODEL", "openai/gpt-3.5-turbo")

    def _convert_chat_to_text(self, chat_history: List[Dict]) -> str:
        """Convert chat history to text format for mmry."""
        text_history = []
        for turn in chat_history:
            timestamp = turn.get("timestamp", "")
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            text_history.append(f"{timestamp} | {speaker}: {text}")
        return "\n".join(text_history)

    def add_conversation_to_memory(
        self, chat_history: List[Dict], user_id: str = "mmry_eval"
    ):
        """Add conversation to mmry."""
        chat_text = self._convert_chat_to_text(chat_history)
        result = self.client.create_memory(chat_text, user_id=user_id)
        return result

    def search_memory(self, query: str, user_id: str = "mmry_eval", top_k: int = 3):
        """Search in mmry."""
        result = self.client.query_memory(query, top_k=top_k, user_id=user_id)
        return result

    def generate_response(self, question: str, context: str):
        """Generate response using OpenRouter API."""
        # Use OpenRouter API for response generation
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Create the question prompt using Jinja2 template
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": MMRY_EVAL_SYSTEM_PROMPT,
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
                result = response.json()
                t2 = time.time()
                return result["choices"][0]["message"]["content"].strip(), t2 - t1
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)  # Wait before retrying

    def process_all_conversations(
        self, output_file_path: str, user_id: str = "mmry_eval"
    ):
        """Process all conversations in the dataset."""
        with open(self.data_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)

        for key, value in tqdm(data.items(), desc="Processing conversations"):
            chat_history = value["conversation"]
            questions = value["question"]

            # Add conversation to memory
            self.add_conversation_to_memory(chat_history, user_id=user_id)

            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item["question"]
                answer = item.get("answer", "")
                category = item["category"]

                # Search memory for context
                search_result = self.search_memory(
                    question, user_id=user_id, top_k=self.k
                )
                context = ". ".join(
                    [mem["payload"]["text"] for mem in search_result["memories"]]
                )

                if not context:
                    # If no memories found, use the original conversation as context
                    context = self._convert_chat_to_text(chat_history)

                response, response_time = self.generate_response(question, context)

                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "context": context,
                        "response": response,
                        "response_time": response_time,
                    }
                )

                # Save results incrementally
                with open(output_file_path, "w+") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        # Final save
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
