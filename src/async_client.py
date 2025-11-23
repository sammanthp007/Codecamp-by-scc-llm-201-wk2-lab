"""Async HTTP client for parallel requests."""
import asyncio
import httpx
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AsyncGoogleBooksClient:
    """Async client for parallel book searches."""
    
    BASE_URL = "https://www.googleapis.com/books/v1/volumes"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 10,
        max_concurrent: int = 5
    ):
        """
        Initialize async client.
        
        Args:
            api_key: Optional API key
            timeout: Request timeout
            max_concurrent: Maximum concurrent requests
        """
        self.api_key = api_key
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create async HTTP client
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        start_index: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Search for books asynchronously.
        
        Args:
            query: Search query
            max_results: Max results
            start_index: Pagination offset
            
        Returns:
            API response or None
        """
        params = {
            "q": query,
            "maxResults": min(max_results, 40),
            "startIndex": start_index
        }
        
        if self.api_key:
            params["key"] = self.api_key
        
        # Use semaphore to limit concurrency
        async with self.semaphore:
            try:
                logger.info(f"Async request: {query} (index={start_index})")
                response = await self.client.get(self.BASE_URL, params=params)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Status {response.status_code} for query: {query}")
                    return None
            
            except Exception as e:
                logger.error(f"Async request failed: {e}")
                return None
    
    async def search_multiple(
        self,
        queries: List[str],
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search multiple queries in parallel.
        
        Args:
            queries: List of search queries
            max_results: Max results per query
            
        Returns:
            List of API responses
        """
        tasks = [
            self.search(query, max_results)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    async def paginated_search(
        self,
        query: str,
        total_results: int = 40,
        results_per_page: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple pages in parallel.
        
        Args:
            query: Search query
            total_results: Total results to fetch
            results_per_page: Results per page
            
        Returns:
            List of all API responses
        """
        num_pages = (total_results + results_per_page - 1) // results_per_page
        
        tasks = [
            self.search(query, results_per_page, i * results_per_page)
            for i in range(num_pages)
        ]
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
