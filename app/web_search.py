import logging
from typing import List

import requests

from .config import settings

logger = logging.getLogger(__name__)

def search_web(query: str) -> List[str]:
    if not settings.tavily_api_key:
        logger.info("web search skipped: missing TAVILY_API_KEY")
        return []

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": settings.web_top_k,
    }
    response = requests.post(
        "https://api.tavily.com/search",
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in (data.get("results") or [])[: settings.web_top_k]:
        title = item.get("title", "Untitled")
        link = item.get("url", "")
        snippet = item.get("content", "")
        results.append(f"{title}\n{link}\n{snippet}".strip())
    logger.info("web search results: %s", len(results))
    return results
