"""
Web search tools for QuARA using DuckDuckGo
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class WebSearchTool:
    """Web search using DuckDuckGo"""
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.logger = logging.getLogger(__name__)
    
    async def search(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """
        Search the web using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            from ddgs import DDGS
            
            max_results = max_results or self.max_results
            
            self.logger.info(f"Searching web for: {query}")
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                
                for r in search_results:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "source": "duckduckgo"
                    })
            
            self.logger.info(f"Found {len(results)} results")
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"
            }
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_news(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """
        Search news articles using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with news results
        """
        try:
            from ddgs import DDGS
            
            max_results = max_results or self.max_results
            
            self.logger.info(f"Searching news for: {query}")
            
            results = []
            with DDGS() as ddgs:
                news_results = ddgs.news(query, max_results=max_results)
                
                for r in news_results:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("body", ""),
                        "date": r.get("date", ""),
                        "source": r.get("source", ""),
                        "type": "news"
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"
            }
        except Exception as e:
            self.logger.error(f"News search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_academic(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for academic papers and articles
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with academic search results
        """
        # Add site filters for academic sources
        academic_query = f"{query} site:arxiv.org OR site:scholar.google.com OR site:pubmed.ncbi.nlm.nih.gov OR site:jstor.org OR site:sciencedirect.com"
        
        return await self.search(academic_query, max_results)
    
    async def multi_search(self, queries: List[str]) -> Dict[str, Any]:
        """
        Perform multiple searches
        
        Args:
            queries: List of search queries
            
        Returns:
            Dictionary with all search results
        """
        all_results = []
        
        for query in queries:
            result = await self.search(query)
            if result["success"]:
                all_results.extend(result["results"])
        
        return {
            "success": True,
            "queries": queries,
            "total_results": len(all_results),
            "results": all_results
        }
    
    def summarize_results(self, search_results: Dict[str, Any]) -> str:
        """
        Create a text summary of search results
        
        Args:
            search_results: Results from search()
            
        Returns:
            Text summary
        """
        if not search_results.get("success"):
            return f"Search failed: {search_results.get('error', 'Unknown error')}"
        
        results = search_results.get("results", [])
        if not results:
            return "No results found"
        
        summary = [f"Search Results for: {search_results.get('query', '')}"]
        summary.append(f"Found {len(results)} results\n")
        
        for i, result in enumerate(results[:5], 1):
            summary.append(f"{i}. {result['title']}")
            summary.append(f"   URL: {result['url']}")
            summary.append(f"   {result['snippet'][:150]}...")
            summary.append("")
        
        return "\n".join(summary)


class WebContentFetcher:
    """Fetch and extract content from web pages"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def fetch_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch and extract main content from a URL
        
        Args:
            url: URL to fetch
            
        Returns:
            Dictionary with extracted content
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            self.logger.info(f"Fetching content from: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {
                "success": True,
                "url": url,
                "title": soup.title.string if soup.title else "",
                "content": text[:5000],  # Limit to first 5000 chars
                "length": len(text)
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "beautifulsoup4 and requests not installed. Run: pip install beautifulsoup4 requests"
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
