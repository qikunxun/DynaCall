"""Chain that interprets a prompt and extracts relevant URLs from search results then fetches their content."""

from __future__ import annotations

import asyncio
import ast
import json
import re
from typing import Any, Dict, List, Optional

from src.docstore.google_search import WebSearchAPI


def _looks_like_blocked_or_challenge_page(content: str) -> bool:
    if not content:
        return False
    lowered = content.lower()
    markers = [
        "verification required",
        "please complete this challenge",
        "access denied",
        "forbidden",
        "captcha",
        "cf-chl",
        "cloudflare",
        "bot verification",
    ]
    return any(marker in lowered for marker in markers)


def _looks_like_unusable_shell_page(url: str, content: str) -> bool:
    if not content:
        return True

    lowered_url = (url or "").lower()
    lowered = content.strip().lower()
    compact = " ".join(lowered.split())

    if "instagram.com/popular/" in lowered_url or "instagram.com/explore/" in lowered_url:
        return True

    shell_texts = {
        "instagram",
        "facebook",
        "x",
        "twitter",
        "youtube",
    }
    if compact in shell_texts:
        return True

    if len(compact) < 40 and any(domain in lowered_url for domain in ("instagram.com", "facebook.com", "x.com", "twitter.com")):
        return True

    return False


def _clean_url(url: str) -> str:
    return str(url).strip().rstrip('",\']])}>')


def _extract_urls_from_text(text: str) -> List[str]:
    if not text:
        return []
    urls = [_clean_url(m) for m in re.findall(r"https?://\S+", str(text))]
    # keep order, remove duplicates
    seen = set()
    ordered = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _normalize_search_results(raw_value: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []

    def _push_url(u: str):
        u = _clean_url(u)
        if u.startswith(("http://", "https://")):
            normalized.append({"href": u, "title": "", "body": ""})

    value = raw_value
    if isinstance(value, str):
        stripped = value.strip()
        parsed = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
                break
            except Exception:
                continue
        value = parsed if parsed is not None else value

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                href = _clean_url(item.get("href", ""))
                if href.startswith(("http://", "https://")):
                    normalized.append({
                        "href": href,
                        "title": str(item.get("title", "")),
                        "body": str(item.get("body", "")),
                    })
            elif isinstance(item, str):
                for u in _extract_urls_from_text(item):
                    _push_url(u)
    elif isinstance(value, dict):
        href = _clean_url(value.get("href", ""))
        if href.startswith(("http://", "https://")):
            normalized.append({
                "href": href,
                "title": str(value.get("title", "")),
                "body": str(value.get("body", "")),
            })
    elif isinstance(value, str):
        for u in _extract_urls_from_text(value):
            _push_url(u)

    return normalized


class URLFetch:
    """Chain that extracts relevant URLs from search results and fetches their content."""

    web_search_api: Optional[WebSearchAPI] = None
    input_key: str = "query"
    output_key: str = "content"
    max_urls: int = 3
    fetch_mode: str = "truncate"

    def __init__(
        self,
        web_search_api: Optional[WebSearchAPI] = None,
        input_key: str = "query",
        output_key: str = "content",
        max_urls: int = 3,
        fetch_mode: str = "truncate",
        **kwargs: Any,
    ):
        """Initialize the URLFetchChain."""
        self.web_search_api = web_search_api or WebSearchAPI()
        self.input_key = input_key
        self.output_key = output_key
        self.max_urls = max_urls
        self.fetch_mode = fetch_mode

    @property
    def input_keys(self) -> List[str]:
        """Expect input key."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key."""
        return [self.output_key]

    def _fetch_urls_content(
        self, 
        urls: List[str],
    ) -> List[Dict[str, str]]:
        """Fetch content from multiple URLs."""
        fetched_content = []
        for i, url in enumerate(urls[:self.max_urls], 1):
            try:
                print(f"Fetching content from URL {i}/{len(urls[:self.max_urls])}: {url}")
                
                result = self.web_search_api.fetch_url_content(url, mode=self.fetch_mode)
                
                if "error" in result:
                    print(f"Error fetching {url}: {result['error']}")
                    continue
                
                content = result.get("content", "")
                if content:
                    if _looks_like_blocked_or_challenge_page(content):
                        print(f"Blocked or challenge page detected for {url}")
                        continue
                    fetched_content.append({
                        "url": url,
                        "content": content
                    })
                    print(f"Successfully fetched {len(content)} characters")
                    
            except Exception as e:
                print(f"Exception fetching {url}: {str(e)}")
                continue
        
        return fetched_content

    async def _afetch_urls_content(
        self, 
        urls: List[str],
    ) -> List[Dict[str, str]]:
        """Async version: Fetch content from multiple URLs."""
        fetched_content = []
        
        async def fetch_single_url(url: str, index: int):
            if url.startswith('//'):
                url = 'https:' + url
            try:
                if hasattr(self.web_search_api, 'afetch_url_content'):
                    result = await self.web_search_api.afetch_url_content(url, mode=self.fetch_mode)
                else:
                    result = self.web_search_api.fetch_url_content(url, mode=self.fetch_mode)
                
                if "error" in result:
                    return {'url': url, 'content': result['error']}
                
                content = result.get("content", "")
                if content:
                    if _looks_like_blocked_or_challenge_page(content):
                        return {'url': url, 'content': "Error: blocked or challenge page"}
                    if _looks_like_unusable_shell_page(url, content):
                        return {'url': url, 'content': "Error: unusable shell page"}
                    # Truncate very long content
                    if len(content) > 10000:
                        content = content[:10000] + "... [truncated]"
                    processed_url = result.get("redirect_url", url)
                    processed_url = processed_url.replace("https://", "").replace("http://", "").split('/')[0]
                    return {
                        "url": processed_url,
                        "content": content
                    }
                    
            except Exception as e:
                print(f"Exception fetching {url}: {str(e)}\n")
                return {'url': url, 'content': f"Error: {str(e)}"}
        
        # Fetch URLs concurrently, limited to max_urls
        tasks = [fetch_single_url(url, i+1) for i, url in enumerate(urls[:self.max_urls])]
        results = await asyncio.gather(*tasks)
        
        # Process results
        for result in results:
            if result and result.get("content") and "Error" not in result.get("content", ""):
                fetched_content.append(result)

        return fetched_content

    def call(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, str]:
        """Execute the chain: fetch content from all URLs."""
        query = inputs[self.input_key]
        search_results = inputs.get("search_results", [])
        
        print(f"Starting URLFetchChain with query: {query}")
        print(f"Number of search results: {len(search_results)}")
        
        normalized_results = _normalize_search_results(search_results)

        query_urls = _extract_urls_from_text(query) if isinstance(query, str) else []
        if query_urls:
            urls = query_urls
        elif isinstance(query, str) and query.startswith(("http://", "https://")):
            urls = [_clean_url(query)]
        else:
            urls = [normalized_results[i].get("href", "") for i in range(len(normalized_results))]
        
        # Fetch content from URLs
        fetched_content = self._fetch_urls_content(urls)
        
        if not fetched_content:
            print("Failed to fetch any content.")
            return {self.output_key: "Failed to fetch content from URLs."}
        
        # Format the output
        final_output = "\n\n".join([
            f"URL: {item['url']}\nContent: {item['content']}"
            for item in fetched_content
        ])
        
        print("Chain completed successfully.")
        return {self.output_key: final_output}

    async def acall(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, str]:
        """Async version: Execute the chain."""
        query = inputs[self.input_key]
        search_results = inputs.get("search_results", [])
        
        # Parse search results if needed
        try:
            normalized_results = _normalize_search_results(search_results)
        except Exception as e:
            return {self.output_key: "Failed to parse search results."}

        # If a direct URL is provided, prefer that path even without search results.
        query_urls = _extract_urls_from_text(query) if isinstance(query, str) else []
        if query_urls:
            urls = query_urls
            titles = ["" for _ in urls]
        elif isinstance(query, str) and query.startswith(("http://", "https://")):
            urls = [_clean_url(query)]
            titles = [""]
        else:
            # If search_results are missing, try a direct search fallback.
            if not normalized_results and isinstance(query, str) and query.strip():
                try:
                    if hasattr(self.web_search_api, "asearch"):
                        normalized_results = await self.web_search_api.asearch(query)
                    else:
                        normalized_results = self.web_search_api.search(query)
                except Exception:
                    normalized_results = []
            urls = [normalized_results[i].get("href", "") for i in range(len(normalized_results))]
            titles = [normalized_results[i].get("title", "") for i in range(len(normalized_results))]

        # Fetch content from URLs
        fetched_content = await self._afetch_urls_content(urls)
        
        if not fetched_content:
            return {self.output_key: "Failed to fetch content from URLs."}
        
        # Format the output
        final_output = ""
        for i, item in enumerate(fetched_content):
            if item:
                title = titles[i] if i < len(titles) else ""
                final_output += f"\nSource: {item['url']}\nTitle: {title}\nContent: {item['content']}"
        
        return {self.output_key: final_output}
