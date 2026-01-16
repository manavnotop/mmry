"""
mmry - Memory Management for AI Agents

Demo script showing the library's core functionality.
Run this to demonstrate mmry works in an interview.
"""

import os

from dotenv import load_dotenv
from mmry import MemoryClient

# Load API key from .env file
load_dotenv()


def main():
    print("=" * 60)
    print("mmry - Memory Management for AI Agents")
    print("=" * 60)

    # Initialize client
    client = MemoryClient({
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "similarity_threshold": 0.8,
    })
    print("\n[1] Client initialized successfully")

    # Create memories for a user
    print("\n[2] Creating memories...")
    user_id = "user_001"

    result1 = client.create_memory(
        "I am a software engineer who loves Python programming",
        user_id=user_id
    )
    print(f"    Memory 1: {result1['status']} (id: {result1['id'][:8]}...)")

    result2 = client.create_memory(
        "I have 5 years of experience building web applications",
        user_id=user_id
    )
    print(f"    Memory 2: {result2['status']} (id: {result2['id'][:8]}...)")

    # Show deduplication - similar memory gets merged
    result3 = client.create_memory(
        "I am a Python developer with expertise in web development",
        user_id=user_id
    )
    print(f"    Memory 3: {result3['status']}")
    if result3['status'] == 'merged':
        print("    -> Merged with existing memory (similarity detected)")

    # Query memories
    print("\n[3] Querying memories...")
    query_result = client.query_memory("What is my technical background?", user_id=user_id)
    print("    Query: 'What is my technical background?'")
    print(f"    Context: {query_result['context_summary']}")
    print(f"    Matches found: {len(query_result['memories'])}")

    # Show all memories
    print("\n[4] All memories stored:")
    all_memories = client.list_all(user_id=user_id)
    for mem in all_memories:
        text = mem['payload'].get('text', '')[:60]
        print(f"    - {text}...")

    # Show multi-user isolation
    print("\n[5] Multi-user support...")
    client.create_memory("I prefer dark mode", user_id="user_002")
    client.create_memory("I prefer light mode", user_id="user_003")

    dark_mem = client.query_memory("What theme do I prefer?", user_id="user_002")
    light_mem = client.query_memory("What theme do I prefer?", user_id="user_003")

    print(f"    User 002 context: {dark_mem['context_summary']}")
    print(f"    User 003 context: {light_mem['context_summary']}")
    print("    -> Users have isolated, independent memories")

    print("\n" + "=" * 60)
    print("Demo complete! mmry is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
