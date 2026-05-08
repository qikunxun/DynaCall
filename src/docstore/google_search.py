"""Wrapper around web search API using aiohttp."""
import os
import re
import json
import asyncio
import random
import time
from typing import List, Optional, Union, Dict, Any
from urllib.parse import quote_plus, urljoin
import requests
import aiohttp
from bs4 import BeautifulSoup
import html2text


def _append_backend_log(line: str) -> None:
    try:
        with open("/tmp/dynacall_websearch_backend.log", "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        pass
def clean_str(p):
    
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except Exception as e:
        print(f"Error cleaning string: {e}")
        return p
class WebSearchAPI:
    """Web Search API wrapper using aiohttp, similar to Wikipedia API format."""

    def __init__(
        self, 
        benchmark: bool = False, 
        skip_retry_when_postprocess: bool = False,
        proxy: Optional[str] = None,
        search_engine: str = "duckduckgo",
        timeout: int = 30
    ) -> None:
        
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "Could not import required packages. "
                "Please install with `pip install beautifulsoup4 aiohttp html2text`."
            )
        
        self.search_results = None
        self.benchmark = benchmark
        self.all_times = []
        self.skip_retry_when_postprocess = skip_retry_when_postprocess
        self.proxy = proxy
        self.search_engine = search_engine  # "google" or "duckduckgo"
        self.timeout = timeout
        
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
        }

    def reset(self):
        
        self.all_times = []

    def get_stats(self):
        
        return {
            "all_times": self.all_times,
        }

    @staticmethod
    def _get_page_obs(page: str, max_sentences: int = 5) -> str:
        
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return " ".join(sentences[:max_sentences])

    def _get_search_url(self, keywords: str, max_results: int = 10) -> str:
        
        encoded_keywords = quote_plus(keywords)
        
        if self.search_engine == "duckduckgo":
            return f"https://duckduckgo.com/html/?q={encoded_keywords}"
        else:  # google
            return f"https://www.google.com/search?q={encoded_keywords}&num={max_results}"

    def _parse_search_results(self, html_content: str) -> List[Dict[str, str]]:
        
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        if self.search_engine == "duckduckgo":
            
            result_divs = soup.find_all('div', class_=lambda x: x and ('result' in x or 'web-result' in x))
            
            for div in result_divs:
                try:
                    
                    title_elem = div.find('a', class_='result__title')
                    if not title_elem:
                        title_elem = div.find('a', class_='result__a')
                    if not title_elem:
                        title_elem = div.find('a')
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    href = title_elem.get('href', '')
                    
                    
                    snippet_elem = div.find('a', class_='result__snippet')
                    if not snippet_elem:
                        snippet_elem = div.find(class_='result__snippet')
                    if not snippet_elem:
                        snippet_elem = div.find(class_='web-result-description')
                    if not snippet_elem:
                        snippet_elem = div.find(class_='result__description')
                    
                    body = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    if href:
                        results.append({
                            "title": clean_str(title),
                            "href": href,
                            "body": clean_str(body)
                        })
                except Exception as e:
                    print(f"Parsing DuckDuckGo Error: {e}")
                    continue
                    
        else:  # google
            
            result_divs = soup.find_all('div', class_='g')
            
            for div in result_divs:
                try:
                    title_elem = div.find('h3')
                    link_elem = div.find('a')
                    
                    if not title_elem or not link_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    href = link_elem.get('href', '')
                    
                    
                    snippet_elem = div.find('div', class_='VwiC3b')
                    if not snippet_elem:
                        snippet_elem = div.find('div', class_='s')
                    
                    body = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    
                    if href.startswith('/url?q='):
                        import re
                        match = re.search(r'/url\?q=(https?://[^&]+)', href)
                        if match:
                            href = match.group(1)
                    
                    if href and href.startswith('http'):
                        results.append({
                            "title": clean_str(title),
                            "href": href,
                            "body": clean_str(body)
                        })
                        
                except Exception as e:
                    print(f"Parsing Google Error: {e}")
                    continue
        return results

    def post_process(
        self, response_text: str, keywords: str, skip_retry_when_postprocess: bool = False
    ) -> Union[str, List[Dict[str, str]]]:
        
        try:
            results = self._parse_search_results(response_text)
            self.search_results = results
            return results
            
        except Exception as e:
            return f"Error processing search results: {str(e)}"
        
    async def apost_process(
        self, response_text: str, keywords: str, skip_retry_when_postprocess: bool = False
    ) -> Union[str, List[Dict[str, str]]]:
        
        try:
            results = self._parse_search_results(response_text)
            
            self.search_results = results
            return results
            
        except Exception as e:
            return f"Error processing search results: {str(e)}"
    def search(
        self, 
        keywords: str, 
        max_results: int = 10,
        region: Optional[str] = "wt-wt"
    ) -> Union[str, List[Dict[str, str]]]:
        
        time.sleep(random.randint(0, 500) / 1000)  
        
        keywords = str(keywords)
        langsearch_key = os.environ.get("LANGSEARCH_API_KEY", "").strip()
        if langsearch_key:
            url = "https://api.langsearch.com/v1/web-search"
            payload = json.dumps({
                "query": keywords,
                "freshness": "noLimit",
                "summary": True,
                "count": max_results
            })
            headers = {
                "Authorization": langsearch_key,
                "Content-Type": "application/json"
            }
            try:
                response = requests.request("POST", url, headers=headers, data=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                values = data.get("data", {}).get("webPages", {}).get("value", [])
                results = []
                for value in values:
                    results.append({
                        "title": clean_str(value.get("name", "")),
                        "href": value.get("url", ""),
                        "body": clean_str(value.get("snippet", "")),
                    })
                if results:
                    msg = f"[WebSearchAPI] backend=langsearch query={keywords!r} results={len(results[:max_results])}"
                    print(msg)
                    _append_backend_log(msg)
                    return results[:max_results]
            except Exception:
                msg = f"[WebSearchAPI] backend=langsearch_failed query={keywords!r}; fallback={self.search_engine}"
                print(msg)
                _append_backend_log(msg)
                pass

        msg = f"[WebSearchAPI] backend={self.search_engine} query={keywords!r} (fallback path)"
        print(msg)
        _append_backend_log(msg)
        search_url = self._get_search_url(keywords, max_results)
        session = requests.Session()
        session.headers.update(self.headers)
        response = session.get(search_url, timeout=self.timeout)
        response.raise_for_status()
        return self.post_process(response.text, keywords)
    
    
    async def asearch(
        self, 
        keywords: str, 
        region: Optional[str] = "wt-wt",
        max_results: int = 10
    ) -> Union[str, List[Dict[str, str]]]:
        
        await asyncio.sleep(random.randint(500, 1000) / 1000)  
        
        keywords = str(keywords)
        return await asyncio.to_thread(
            self.search,
            keywords,
            max_results,
            region,
        )

    def fetch_url_content(
        self, 
        url: str, 
        mode: str = "truncate",
        timeout: int = 30
    ) -> Dict[str, str]:
        time.sleep(random.randint(100, 500) / 1000)
        
        if not url.startswith(("http://", "https://")):
            return {"error": f"Invalid URL: {url}"}
        
        try:
            
            session = requests.Session()
            session.headers.update(self.headers)
            
            
            proxies = None
            if self.proxy:
                proxies = {
                    'http': self.proxy,
                    'https': self.proxy
                }
            
            
            response = session.get(
                url,
                proxies=proxies,
                timeout=timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            content = response.text
            
            
            js_patterns = [ 
                r'window\.(?:parent\.)?location\.replace\([\'"]([^\'"]+)[\'"]\)',
                r'window\.location\.href\s*=\s*[\'"]([^\'"]+)[\'"]',
                r'window\.location\s*=\s*[\'"]([^\'"]+)[\'"]',
            ]
            
            redirect_url = None
            for pattern in js_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    redirect_url = match.group(1)
                    
                    if not redirect_url.startswith(('http://', 'https://')):
                        redirect_url = urljoin(response.url, redirect_url)
                    break
            
            
            if redirect_url:
                response = session.get(
                    redirect_url,
                    proxies=proxies,
                    timeout=timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                content = response.text
            
            
            if mode == "raw":
                result = {"content": content}
                
            elif mode == "markdown":
                converter = html2text.HTML2Text()
                converter.ignore_links = False
                converter.ignore_images = True
                converter.ignore_emphasis = False
                markdown = converter.handle(content)
                result = {"content": markdown}
                
            elif mode == "truncate":
                soup = BeautifulSoup(content, "html.parser")
                
                
                for element in soup(["script", "style", "header", "footer", "nav", 
                                    "aside", "form", "iframe", "noscript"]):
                    element.decompose()
                
                
                main_content = None
                for selector in ['article', 'main', '#content', '.content', '.main-content']:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                
                if main_content:
                    text = main_content.get_text(separator="\n", strip=True)
                else:
                    
                    body = soup.find('body')
                    if body:
                        text = body.get_text(separator="\n", strip=True)
                    else:
                        text = soup.get_text(separator="\n", strip=True)
                
                
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                cleaned_text = '\n'.join(lines)
                result = {"content": cleaned_text}
                
            else:
                return {"error": f"Unsupported mode: {mode}"}
            
            
            if redirect_url:
                result['redirect_url'] = redirect_url
                
            return result

        except requests.exceptions.Timeout:
            return {"error": f"Timeout fetching {url} after {timeout} seconds"}
        except requests.exceptions.RequestException as e:
            return {"error": f"HTTP error fetching {url}: {str(e)}"}
        except Exception as e:
            return {"error": f"Error fetching {url}: {str(e)}"}
    
    async def afetch_url_content(
        self, 
        url: str, 
        mode: str = "truncate",
        timeout: int = 30
    ) -> Dict[str, str]:
        await asyncio.sleep(random.randint(100, 500) / 1000)
        if not url.startswith(("http://", "https://")):
            return {"error": f"Invalid URL: {url}"}
        try:     
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(
                    url, 
                    proxy=self.proxy,
                    timeout=timeout,
                    allow_redirects=True
                ) as response:
                    response.raise_for_status()
                    content = await response.text()
            js_patterns = [ 
                r'window\.(?:parent\.)?location\.replace\([\'"]([^\'"]+)[\'"]\)',
                r'window\.location\.href\s*=\s*[\'"]([^\'"]+)[\'"]',
                r'window\.location\s*=\s*[\'"]([^\'"]+)[\'"]',
            ]
            redirect_url = None
            for pattern in js_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    redirect_url = match.group(1)
                    
                    if not redirect_url.startswith(('http://', 'https://')):
                        redirect_url = urljoin(str(response.url), redirect_url)
                    break
            if redirect_url:
                async with aiohttp.ClientSession(headers=self.headers) as session:
                    async with session.get(
                        redirect_url, 
                        proxy=self.proxy,
                        timeout=timeout,
                        allow_redirects=True
                    ) as response:
                        response.raise_for_status()
                        content = await response.text()
            
            if mode == "raw":
                return {"content": content}
                
            elif mode == "markdown":
                converter = html2text.HTML2Text()
                converter.ignore_links = False
                converter.ignore_images = True
                converter.ignore_emphasis = False
                markdown = converter.handle(content)
                return {"content": markdown}
                
            elif mode == "truncate":
                soup = BeautifulSoup(content, "html.parser")
                
                
                for element in soup(["script", "style", "header", "footer", "nav", 
                                    "aside", "form", "iframe", "noscript"]):
                    element.decompose()
                
                
                main_content = None
                for selector in ['article', 'main', '#content', '.content', '.main-content']:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                
                if main_content:
                    text = main_content.get_text(separator="\n", strip=True)
                else:
                    
                    body = soup.find('body')
                    if body:
                        text = body.get_text(separator="\n", strip=True)
                    else:
                        text = soup.get_text(separator="\n", strip=True)
                
                
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                cleaned_text = '\n'.join(lines)
                if redirect_url:
                    return {"content": cleaned_text, 'redirect_url': redirect_url}
                else:
                    return {"content": cleaned_text}
            else:
                return {"error": f"Unsupported mode: {mode}"}

        except asyncio.TimeoutError:
            return {"error": f"Timeout fetching {url} after {timeout} seconds"}
        except aiohttp.ClientError as e:
            return {"error": f"HTTP error fetching {url}: {str(e)}"}
        except Exception as e:
            return {"error": f"Error fetching {url}: {str(e)}"}

    async def afetch_multiple_urls(
        self,
        urls: List[str],
        mode: str = "truncate",
        timeout_per_url: int = 20,
        max_concurrent: int = 3
    ) -> List[Dict[str, str]]:
        results = []
        
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(url: str):
            async with semaphore:
                return await self.afetch_url_content(url, mode, timeout_per_url)
        
        
        tasks = [fetch_with_semaphore(url) for url in urls]
        
        
        fetched_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        
        for url, result in zip(urls, fetched_results):
            if isinstance(result, Exception):
                results.append({
                    "url": url,
                    "content": "",
                    "error": f"Exception: {str(result)}"
                })
            else:
                if "error" in result:
                    results.append({
                        "url": url,
                        "content": "",
                        "error": result["error"]
                    })
                else:
                    results.append({
                        "url": url,
                        "content": result["content"],
                        "error": None
                    })
        
        return results
