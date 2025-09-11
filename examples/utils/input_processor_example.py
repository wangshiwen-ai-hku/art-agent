#!/usr/bin/env python3
"""
Example demonstrating the MultimodalInputProcessor for creating HumanMessage objects
with various modalities (text, images, audio, documents) for LangChain agents.

This example shows proper error handling and usage patterns.
"""

import sys
import base64
import logging
import os
import dotenv
from langchain.chat_models import init_chat_model

sys.path.insert(0, "..")

dotenv.load_dotenv()

from src.utils.input_processor import create_multimodal_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_LLM_CONFIG = {
    "model": "gpt-4.1-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": "https://api.openai-proxy.org/v1",
    "temperature": 0,
}
llm = init_chat_model(**DEFAULT_LLM_CONFIG)


def example_text_with_image():
    """Example: Create a message with text and image"""
    print("\n=== Text + Image Example ===")

    file_path = "../data/test_data/二次函数.jpg"
    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    message = create_multimodal_message(
        text="Can you help me understand this quadratic function graph?",
        image_data=image_data,
    )
    response = llm.invoke([message]).content

    print(f"Response: {response}")


def example_image_from_url():
    """Example: Create a message with image from URL"""
    print("\n=== Image from URL Example ===")

    message = create_multimodal_message(
        text="What's in this image?",
        image_url="https://test2.actwise.xyz/agent-imgs/wAQipvN.jpg",
    )
    response = llm.invoke([message]).content
    print(f"Response: {response}")


def main():
    """Run all examples"""
    print("MultimodalInputProcessor Examples")
    print("=" * 50)
    example_text_with_image()
    example_image_from_url()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()
