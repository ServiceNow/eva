#!/usr/bin/env python3
"""Quick test script for LLM using LiteLLM client."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eva.assistant.services.llm import LiteLLMClient
from eva.utils import router
from eva.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def test_llm(model: str, prompt: str = "Write 50 words about why the sun sets in the west."):
    """Test an LLM using LiteLLM client.

    Args:
        model: Model identifier (e.g., "gpt-4o-mini", "claude-sonnet-4.5")
        prompt: Test prompt to send to the model
        test_json: Whether to run JSON response format test
    """
    # Initialize router from EVA_MODEL_LIST
    model_list_json = os.environ.get("EVA_MODEL_LIST")
    if not model_list_json:
        logger.error("EVA_MODEL_LIST environment variable not set")
        sys.exit(1)

    # Load Model List
    try:
        model_list = json.loads(model_list_json)
        router.init(model_list)
        logger.info("✓ Router initialized")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in EVA_MODEL_LIST: {e}")
        sys.exit(1)

    logger.info(f"\n{'-' * 80}\nTesting model: {model}\nPrompt: {prompt}\n{'-' * 80}")

    try:
        # Create client and call LLM
        client = LiteLLMClient(model=model)
        messages = [{"role": "user", "content": prompt}]

        logger.info("Sending request...")
        response, stats = await client.complete(messages)

        logger.info(f"\n✓ Response:\n{response}\n\nStats:\n{stats}\n\nTest completed successfully!")

        return True

    except Exception as e:
        logger.exception(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test LLM using LiteLLM client")
    parser.add_argument("model", nargs="?", default="gpt-5.2", help="Model identifier (default: gpt-5.2)")
    parser.add_argument(
        "-p", "--prompt", default="Write 50 words about why the sun sets in the west.", help="Test prompt to send"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(level="INFO")

    # Run test
    success = asyncio.run(test_llm(args.model, args.prompt))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
