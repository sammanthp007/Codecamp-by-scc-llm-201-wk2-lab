"""HTTP client for Google Books API with resilience patterns."""
import time
import random
import requests
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoogleBooksClient:
    """Client for Google Books API with timeouts, retries, and backoff."""
    
    BASE_URL = "https://www.googleapis.com/books/v1/volumes"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        base_backoff: float = 1.0
    ):
        """
        Initialize Google Books API client.
        
        Args:
            api_key: Optional API key (increases rate limits)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_backoff: Base delay for exponential backoff
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        
        # Create session for connection pooling
        self.session = requests.Session()
    
    def search(
        self, 
        query: str, 
        max_results: int = 10,
        start_index: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Search for books.
        
        Args:
            query: Search query string
            max_results: Maximum results to return (1-40)
            start_index: Pagination offset
            
        Returns:
            API response JSON or None if all retries failed
        """
        params = {
            "q": query,
            "maxResults": min(max_results, 40),  # API limit
            "startIndex": start_index
        }
        
        if self.api_key:
            params["key"] = self.api_key
        
        return self._make_request_with_retry(self.BASE_URL, params)
    
    def _make_request_with_retry(
        self, 
        url: str, 
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: Request URL
            params: Query parameters
            
        Returns:
            Response JSON or None if all retries exhausted
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Request attempt {attempt + 1}/{self.max_retries}: {url}")
                
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                
                # Handle different status codes
                if response.status_code == 200:
                    logger.info(f"Success: {response.status_code}")
                    return response.json()
                
                elif response.status_code == 429:
                    # Rate limited - must retry with backoff
                    logger.warning(f"Rate limited (429) on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        self._backoff(attempt)
                        continue
                
                elif response.status_code >= 500:
                    # Server error - retryable
                    logger.warning(f"Server error ({response.status_code}) on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        self._backoff(attempt)
                        continue
                
                elif response.status_code >= 400:
                    # Client error - don't retry
                    logger.error(f"Client error ({response.status_code}): {response.text}")
                    return None
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
                    continue
            
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
                    continue
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
        
        logger.error(f"All {self.max_retries} attempts failed")
        return None
    
    def _backoff(self, attempt: int):
        """
        Sleep with exponential backoff and jitter.
        
        Args:
            attempt: Current attempt number (0-indexed)
        """
        # Exponential backoff: base * 2^attempt
        delay = self.base_backoff * (2 ** attempt)
        
        # Add jitter: random value between 0 and delay
        jitter = random.uniform(0, delay)
        total_delay = delay + jitter
        
        logger.info(f"Backing off for {total_delay:.2f} seconds")
        time.sleep(total_delay)
    
    def search_with_cache(
        self,
        query: str,
        max_results: int = 10,
        start_index: int = 0,
        cache_db=None,
        cache_ttl: int = 3600
    ) -> Optional[Dict[str, Any]]:
        """
        Search with database-backed caching.
        
        Args:
            query: Search query
            max_results: Max results
            start_index: Pagination offset
            cache_db: Database instance (optional)
            cache_ttl: Cache TTL in seconds
            
        Returns:
            API response or None
        """
        # Generate cache key
        cache_key = f"books:search:{query}:{max_results}:{start_index}"
        
        # Try cache first
        if cache_db:
            cached = cache_db.cache_get(cache_key)
            if cached:
                return cached
        
        # Cache miss - fetch from API
        response = self.search(query, max_results, start_index)
        
        # Cache the response
        if response and cache_db:
            cache_db.cache_set(cache_key, response, cache_ttl)
        
        return response
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
