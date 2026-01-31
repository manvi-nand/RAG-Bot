import logging
from typing import List

from google import genai
from google.genai import types

from .config import settings

logger = logging.getLogger(__name__)


def search_web(query: str) -> List[str]:
    if not settings.google_api_key:
        logger.info("web search skipped: missing GOOGLE_API_KEY")
        return []

    client = genai.Client(api_key=settings.google_api_key)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    response = client.models.generate_content(
        model=settings.grounding_model,
        contents=query,
        config=config,
    )

    results: List[str] = []
    if response.text:
        results.append(response.text.strip())
    logger.info("web search results: %s", len(results))
    return results
