import json
import logging
import re
from collections.abc import Generator

logger = logging.getLogger(__name__)

RE_JSON_START = re.compile(r"[{[]")


def extract_and_load_json_iter(
    text: str, *, start: int = 0, strict: bool = False
) -> Generator[tuple[dict | list | None, str], None, None]:
    """Method to extract JSON objects and arrays from text, even if there is text around.

    Args:
    ----
        text: The text
        start: Start searching at this index
        strict: Whether to use strict JSON decoding

    Returns:
    -------
        Generator that yields all valid JSON objects and arrays found in the text, as well as the text that was matched.
        If there is no valid JSON object in the text, the generator will be empty.

    """
    decoder = json.JSONDecoder(strict=strict)
    while match := RE_JSON_START.search(text, start):
        start = match.start()
        try:
            json_object, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start += 1
        else:
            yield json_object, text[start:end]
            start = end


def extract_and_load_json(text: str) -> dict | list | None:
    """Method to extract and load JSON from text.

    Args:
    ----
        text: str: The text

    Returns:
    -------
        dict: The JSON object or None

    """
    json_object, _ = next(extract_and_load_json_iter(text), (None, ""))
    if json_object is None:
        logger.warning(f"Error extracting JSON from text: {text}")
    return json_object
