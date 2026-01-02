"""Web search tool for looking up information online."""
import sys
from pathlib import Path

# Ensure project root is in path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional
import logging
from config.settings import SERPER_API_KEY, TAVILY_API_KEY, DUCKDUCKGO_API_KEY

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Web search tool with multiple provider support."""
    
    def __init__(self, provider: str = "duckduckgo"):
        """Initialize web search tool.
        
        Args:
            provider: "duckduckgo", "serper", or "tavily"
        """
        self.provider = provider
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup the search provider."""
        if self.provider == "serper":
            if not SERPER_API_KEY:
                raise ValueError("SERPER_API_KEY not set. Falling back to DuckDuckGo.")
                self.provider = "duckduckgo"
        elif self.provider == "tavily":
            if not TAVILY_API_KEY:
                raise ValueError("TAVILY_API_KEY not set. Falling back to DuckDuckGo.")
                self.provider = "duckduckgo"
    
    def search(self, query: str, num_results: int = 5) -> str:
        """Search the web and return summarized results.
        
        Args:
            query: Search query
            num_results: Number of results to return
        
        Returns:
            Summarized search results
        """
        logger.info("tool=look_on_web provider=%s query=%s max_results=%s", self.provider, query, num_results)
        if self.provider == "duckduckgo":
            return self._duckduckgo_search(query, num_results)
        elif self.provider == "serper":
            return self._serper_search(query, num_results)
        elif self.provider == "tavily":
            return self._tavily_search(query, num_results)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _duckduckgo_search(self, query: str, num_results: int) -> str:
        """Search using DuckDuckGo (no API key needed)."""
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            
            if not results:
                return f"No results found for: {query}"
            
            summary = f"Search results for: {query}\n\n"
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                snippet = result.get("body", result.get("snippet", "No description"))
                url = result.get("href", "No URL")
                summary += f"{i}. {title}\n   {snippet}\n   Source: {url}\n\n"
            
            return summary.strip()
        except ImportError:
            return f"DuckDuckGo search not available. Install with: pip install duckduckgo-search"
        except Exception as e:
            return f"Error searching: {str(e)}"
    
    def _serper_search(self, query: str, num_results: int) -> str:
        """Search using Serper API."""
        try:
            import requests
            
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": num_results
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            summary = f"Search results for: {query}\n\n"
            if "organic" in data:
                for i, result in enumerate(data["organic"][:num_results], 1):
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", "No description")
                    link = result.get("link", "No URL")
                    summary += f"{i}. {title}\n   {snippet}\n   Source: {link}\n\n"
            
            return summary.strip()
        except Exception as e:
            return f"Error with Serper search: {str(e)}"
    
    def _tavily_search(self, query: str, num_results: int) -> str:
        """Search using Tavily API."""
        try:
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(query=query, max_results=num_results)
            
            summary = f"Search results for: {query}\n\n"
            if "results" in response:
                for i, result in enumerate(response["results"][:num_results], 1):
                    title = result.get("title", "No title")
                    content = result.get("content", "No description")
                    url = result.get("url", "No URL")
                    summary += f"{i}. {title}\n   {content}\n   Source: {url}\n\n"
            
            return summary.strip()
        except ImportError:
            return f"Tavily not available. Install with: pip install tavily-python"
        except Exception as e:
            return f"Error with Tavily search: {str(e)}"


def look_on_web(query: str) -> str:
    """Wrapper function for LangChain tool."""
    tool = WebSearchTool()
    return tool.search(query)


def look_on_web_with_params(query: str, max_results: int = 5) -> str:
    """Wrapper with explicit parameters for LangChain tool."""
    tool = WebSearchTool()
    return tool.search(query, num_results=max_results)


