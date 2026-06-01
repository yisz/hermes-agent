"""Perplexity web search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses the
Perplexity Search API (``POST /search``) which returns ranked results with
title, url, snippet, and date — no LLM-generated summaries. Search-only;
web_extract and web_crawl are not supported.

Config keys this provider responds to::

    web:
      search_backend: "perplexity"   # explicit per-capability
      backend: "perplexity"          # shared fallback

Env vars::

    PERPLEXITY_API_KEY=***           # https://www.perplexity.ai/settings/api
    PERPLEXITY_API_URL=...           # Optional: self-hosted or proxy override
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_PERPLEXITY_DEFAULT_API_URL = "https://api.perplexity.ai"


def _perplexity_base_url() -> str:
    """Return the Perplexity API base URL, configurable via PERPLEXITY_API_URL.

    Allows overriding for self-hosted or proxy deployments, matching the
    pattern used by FIRECRAWL_API_URL.
    """
    return os.getenv("PERPLEXITY_API_URL", "").strip().rstrip("/") or _PERPLEXITY_DEFAULT_API_URL


def _normalize_perplexity_search_results(data: dict) -> dict:
    """Normalize Perplexity /search response to the standard web search format.

    Perplexity returns ``{id: str, results: [{title, url, snippet, date}]}``.
    We map to ``{success: bool, data: {web: [{title, url, description, date, position}]}}``.
    """
    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    web_results = []
    for i, result in enumerate(raw_results):
        if not isinstance(result, dict):
            continue
        url = result.get("url", "") or ""
        title = result.get("title", "") or ""
        snippet = result.get("snippet", "") or ""
        date = result.get("date", "") or ""
        # Skip results missing both URL and title — not useful for the agent
        if not url and not title:
            continue
        web_results.append({
            "url": url,
            "title": title,
            "description": snippet,
            "date": date,
            "position": i + 1,
        })

    return {"success": True, "data": {"web": web_results}}


class PerplexityWebSearchProvider(WebSearchProvider):
    """Perplexity search provider (search-only, no extract).

    Both ``search`` and the normalization helpers are sync — the Perplexity
    Search API is a simple POST/JSON endpoint.
    """

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def display_name(self) -> str:
        return "Perplexity"

    def is_available(self) -> bool:
        """Return True when ``PERPLEXITY_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("PERPLEXITY_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Perplexity search.

        Uses the POST /search endpoint on the Perplexity API.
        Returns results in the standard Hermes web search format.

        Perplexity Search API returns ranked results with title, url, snippet,
        and date — no LLM-generated summaries. It is a search-only backend;
        web_extract and web_crawl are not supported.

        Args:
            query: The search query string.
            limit: Maximum number of results to return (1-20, API max is 20).

        Returns:
            dict with ``success`` and ``data.web`` list of result dicts.
        """
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return {"success": False, "error": "Interrupted"}

        api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
        if not api_key:
            return {
                "success": False,
                "error": "PERPLEXITY_API_KEY not set. Get an API key at https://www.perplexity.ai/settings/api",
            }

        limit = max(1, min(limit, 20))
        base_url = _perplexity_base_url()
        search_url = f"{base_url}/search"
        logger.info("Perplexity search: '%s' (limit=%d)", query, limit)

        payload = {
            "query": query,
            "max_results": limit,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = httpx.post(
                search_url,
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_text = exc.response.text[:200] if exc.response.text else ""
            logger.error("Perplexity search HTTP error: %s %s", status_code, error_text)

            # Provide helpful error messages for common cases
            if status_code == 401:
                return {"success": False, "error": "Perplexity authentication failed (401): check your PERPLEXITY_API_KEY"}
            if status_code == 429:
                retry_after = exc.response.headers.get("retry-after", "")
                if retry_after:
                    return {"success": False, "error": f"Perplexity rate limit exceeded (429). Retry after {retry_after} seconds."}
                return {"success": False, "error": "Perplexity rate limit exceeded (429). Please wait before retrying."}
            if status_code >= 500:
                return {"success": False, "error": f"Perplexity server error ({status_code}). Please try again later."}
            return {"success": False, "error": f"Perplexity API error {status_code}: {error_text}"}
        except httpx.RequestError as exc:
            logger.error("Perplexity search request error: %s", exc)
            return {"success": False, "error": f"Perplexity request failed: {exc}"}

        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.error("Perplexity returned invalid JSON")
            return {"success": False, "error": "Perplexity returned invalid JSON. Please try again."}

        return _normalize_perplexity_search_results(data)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Perplexity",
            "badge": "paid",
            "tag": "Search API — ranked results with title, URL, snippet, and date. No content extraction.",
            "env_vars": [
                {
                    "key": "PERPLEXITY_API_KEY",
                    "prompt": "Perplexity API key",
                    "url": "https://www.perplexity.ai/settings/api",
                },
                {
                    "key": "PERPLEXITY_API_URL",
                    "prompt": "Perplexity API base URL (optional, for self-hosted/proxy)",
                    "url": "",
                },
            ],
        }
