"""Perplexity web search plugin — bundled, auto-loaded.

Searches via the Perplexity Search API (POST /search). Search-only —
no content extraction. Requires PERPLEXITY_API_KEY.
"""

from __future__ import annotations

from plugins.web.perplexity.provider import PerplexityWebSearchProvider


def register(ctx) -> None:
    """Register the Perplexity provider with the plugin context."""
    ctx.register_web_search_provider(PerplexityWebSearchProvider())
