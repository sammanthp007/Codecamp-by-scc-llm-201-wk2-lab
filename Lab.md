# Week 2 Lab Guide: Database & API Integration with Google Books

## Overview

In this lab, you'll build a production-ready **Book Search CLI** that demonstrates database-backed API orchestration using the **Google Books API**. You'll implement the complete data pipeline: API requests → parsing → database storage → caching → reporting.

## What You'll Build

A CLI tool that can:

- Search for books using Google Books API
- Store results in PostgreSQL with proper schema design
- Cache API responses with TTL (time-to-live)
- Handle network failures gracefully (timeouts, retries, backoff)
- Make parallel requests with concurrency control
- Generate reports and statistics from the database

## Lab Structure

This lab is divided into **3 progressive sessions**:

**Session A (Lab 1):** Build synchronous API client with resilience patterns  
**Session B (Lab 2):** Add database layer with caching  
**Session C (Lab 3):** Implement async parallelism and CLI

---

## Session A: API Client with Resilience (Lab 1)

### Learning Goals

- Make HTTP requests to Google Books API
- Parse and normalize JSON responses
- Implement timeouts, retries, and exponential backoff
- Write testable, pure functions

### Step 1: Set Up Project Structure

Create the following directory structure:

```
week2-lab/
├── src/
│   ├── __init__.py
│   ├── client.py          # HTTP client with resilience
│   ├── parse.py           # JSON parsing and normalization
│   └── models.py          # Data models
├── tests/
│   ├── __init__.py
│   ├── test_parse.py
│   └── test_client.py
├── .env.example
├── requirements.txt
└── README.md
```

### Step 2: Install Dependencies

```bash
pip install requests python-dotenv
```

### Step 3: Create Data Models

**File: `src/models.py`**

Create a simple data class to represent a book:

```python
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime


@dataclass
class Book:
    """Normalized book representation."""
    id: str
    title: str
    authors: List[str]
    published_date: Optional[str]
    description: Optional[str]
    page_count: Optional[int]
    categories: List[str]
    thumbnail: Optional[str]
    language: str

    @property
    def authors_str(self) -> str:
        """Format authors as comma-separated string."""
        return ", ".join(self.authors) if self.authors else "Unknown"

    @property
    def categories_str(self) -> str:
        """Format categories as comma-separated string."""
        return ", ".join(self.categories) if self.categories else "None"
```

**Key Concepts:**

- `@dataclass`: Automatic `__init__`, `__repr__`, `__eq__` generation
- `Optional[T]`: Fields that can be `None` (APIs don't always return all fields)
- Property methods: Clean string representation for display

### Step 4: Build JSON Parser

**File: `src/parse.py`**

```python
"""Parse and normalize Google Books API responses."""
from typing import Dict, Any, List, Optional
from src.models import Book


def parse_book(item: Dict[str, Any]) -> Optional[Book]:
    """
    Parse a single book item from Google Books API.

    Args:
        item: Single item from Google Books API response

    Returns:
        Book object or None if parsing fails
    """
    try:
        volume_info = item.get("volumeInfo", {})

        # Extract fields with safe defaults
        book_id = item.get("id", "")
        if not book_id:
            return None

        title = volume_info.get("title", "Unknown Title")
        authors = volume_info.get("authors", [])
        published_date = volume_info.get("publishedDate")
        description = volume_info.get("description")
        page_count = volume_info.get("pageCount")
        categories = volume_info.get("categories", [])
        language = volume_info.get("language", "en")

        # Extract thumbnail (prefer higher quality)
        image_links = volume_info.get("imageLinks", {})
        thumbnail = image_links.get("thumbnail") or image_links.get("smallThumbnail")

        return Book(
            id=book_id,
            title=title,
            authors=authors,
            published_date=published_date,
            description=description,
            page_count=page_count,
            categories=categories,
            thumbnail=thumbnail,
            language=language
        )
    except Exception as e:
        # Log but don't crash - APIs can be unpredictable
        print(f"Warning: Failed to parse book: {e}")
        return None


def parse_books_response(response_json: Dict[str, Any]) -> List[Book]:
    """
    Parse full Google Books API response.

    Args:
        response_json: Complete API response JSON

    Returns:
        List of Book objects (empty if no items found)
    """
    items = response_json.get("items", [])
    books = []

    for item in items:
        book = parse_book(item)
        if book:
            books.append(book)

    return books


def deduplicate_books(books: List[Book]) -> List[Book]:
    """
    Remove duplicate books by ID.

    Args:
        books: List of Book objects

    Returns:
        Deduplicated list of books
    """
    seen_ids = set()
    unique_books = []

    for book in books:
        if book.id not in seen_ids:
            seen_ids.add(book.id)
            unique_books.append(book)

    return unique_books
```

**Key Concepts:**

- **Safe dictionary access**: Use `.get()` with defaults instead of `[]`
- **Defensive parsing**: Handle missing fields gracefully
- **Pure functions**: No side effects, easy to test
- **Deduplication**: Use sets for O(1) lookup

### Step 5: Build HTTP Client with Resilience

**File: `src/client.py`**

```python
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

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
```

**Key Concepts:**

- **Timeouts**: Never hang forever - always set connect/read timeouts
- **Retries**: Only retry on transient errors (5xx, 429, timeouts, connection errors)
- **Exponential backoff**: Wait progressively longer between retries (1s, 2s, 4s...)
- **Jitter**: Add randomness to prevent thundering herd
- **Connection pooling**: Reuse TCP connections with `requests.Session`
- **Context manager**: Automatic cleanup with `with` statement

### Step 6: Write Tests

**File: `tests/test_parse.py`**

```python
"""Tests for parsing functions."""
from src.parse import parse_book, parse_books_response, deduplicate_books
from src.models import Book


def test_parse_book_complete():
    """Test parsing a book with all fields present."""
    item = {
        "id": "abc123",
        "volumeInfo": {
            "title": "Python Crash Course",
            "authors": ["Eric Matthes"],
            "publishedDate": "2019-05-03",
            "description": "A great book",
            "pageCount": 544,
            "categories": ["Programming"],
            "language": "en",
            "imageLinks": {
                "thumbnail": "http://example.com/thumb.jpg"
            }
        }
    }

    book = parse_book(item)

    assert book is not None
    assert book.id == "abc123"
    assert book.title == "Python Crash Course"
    assert book.authors == ["Eric Matthes"]
    assert book.page_count == 544


def test_parse_book_missing_fields():
    """Test parsing a book with missing optional fields."""
    item = {
        "id": "xyz789",
        "volumeInfo": {
            "title": "Mystery Book"
        }
    }

    book = parse_book(item)

    assert book is not None
    assert book.id == "xyz789"
    assert book.title == "Mystery Book"
    assert book.authors == []
    assert book.description is None
    assert book.page_count is None


def test_parse_book_no_id():
    """Test that book without ID returns None."""
    item = {
        "volumeInfo": {
            "title": "No ID Book"
        }
    }

    book = parse_book(item)
    assert book is None


def test_parse_books_response():
    """Test parsing complete API response."""
    response = {
        "items": [
            {
                "id": "1",
                "volumeInfo": {"title": "Book 1"}
            },
            {
                "id": "2",
                "volumeInfo": {"title": "Book 2"}
            }
        ]
    }

    books = parse_books_response(response)

    assert len(books) == 2
    assert books[0].title == "Book 1"
    assert books[1].title == "Book 2"


def test_deduplicate_books():
    """Test deduplication by book ID."""
    books = [
        Book("1", "Book A", [], None, None, None, [], None, "en"),
        Book("2", "Book B", [], None, None, None, [], None, "en"),
        Book("1", "Book A Duplicate", [], None, None, None, [], None, "en"),
    ]

    unique = deduplicate_books(books)

    assert len(unique) == 2
    assert unique[0].id == "1"
    assert unique[1].id == "2"
```

### Step 7: Create Simple Test Script

**File: `test_lab1.py`** (in root directory)

```python
"""Simple script to test Lab 1 functionality."""
from src.client import GoogleBooksClient
from src.parse import parse_books_response, deduplicate_books


def main():
    """Test the client and parser."""
    print("=" * 60)
    print("Lab 1 Test: Searching for Python books")
    print("=" * 60)

    # Create client
    with GoogleBooksClient(timeout=10, max_retries=3) as client:
        # Search for books
        response = client.search("python programming", max_results=5)

        if response is None:
            print("❌ Failed to fetch data")
            return

        # Parse response
        books = parse_books_response(response)
        books = deduplicate_books(books)

        print(f"\n✅ Found {len(books)} books:\n")

        for i, book in enumerate(books, 1):
            print(f"{i}. {book.title}")
            print(f"   Authors: {book.authors_str}")
            print(f"   Published: {book.published_date or 'Unknown'}")
            print(f"   Pages: {book.page_count or 'Unknown'}")
            print(f"   Categories: {book.categories_str}")
            print()


if __name__ == "__main__":
    main()
```

### Run Lab 1

```bash
python test_lab1.py
```

**Expected output:**

```
============================================================
Lab 1 Test: Searching for Python books
============================================================
2024-01-15 10:30:45 - INFO - Request attempt 1/3: https://www.googleapis.com/books/v1/volumes
2024-01-15 10:30:46 - INFO - Success: 200

✅ Found 5 books:

1. Python Crash Course
   Authors: Eric Matthes
   Published: 2019-05-03
   Pages: 544
   Categories: Programming
...
```

---

## Session B: Database Layer with Caching (Lab 2)

### Learning Goals

- Design PostgreSQL schema for book data and cache
- Implement connection pooling
- Add database-backed caching with TTL
- Write database transactions safely

### Step 1: Install PostgreSQL and Dependencies

```bash
# Install PostgreSQL (macOS)
brew install postgresql@15
brew services start postgresql@15

# Install Python dependencies
pip install psycopg2-binary python-dotenv
```

### Step 2: Create Database Schema

**File: `src/database.py`**

```python
"""Database layer for book storage and caching."""
import psycopg2
from psycopg2 import pool
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL database with connection pooling."""

    def __init__(self, connection_string: str, min_conn: int = 1, max_conn: int = 10):
        """
        Initialize database connection pool.

        Args:
            connection_string: PostgreSQL connection string
            min_conn: Minimum connections in pool
            max_conn: Maximum connections in pool
        """
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            min_conn,
            max_conn,
            connection_string
        )

        if self.connection_pool:
            logger.info("Database connection pool created successfully")
        else:
            raise Exception("Failed to create connection pool")

    def init_schema(self):
        """Create database tables if they don't exist."""
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Books table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS books (
                        id VARCHAR(255) PRIMARY KEY,
                        title TEXT NOT NULL,
                        authors TEXT[],
                        published_date VARCHAR(50),
                        description TEXT,
                        page_count INTEGER,
                        categories TEXT[],
                        thumbnail TEXT,
                        language VARCHAR(10),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Cache table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        cache_key VARCHAR(512) PRIMARY KEY,
                        response_data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL
                    )
                """)

                # Indexes for performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_books_title
                    ON books USING gin(to_tsvector('english', title))
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_expires
                    ON api_cache (expires_at)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_books_created
                    ON books (created_at DESC)
                """)

                conn.commit()
                logger.info("Database schema initialized successfully")

        finally:
            self.connection_pool.putconn(conn)

    def insert_book(self, book) -> bool:
        """
        Insert or update a book in the database.

        Args:
            book: Book object

        Returns:
            True if successful, False otherwise
        """
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO books (
                        id, title, authors, published_date, description,
                        page_count, categories, thumbnail, language, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        authors = EXCLUDED.authors,
                        published_date = EXCLUDED.published_date,
                        description = EXCLUDED.description,
                        page_count = EXCLUDED.page_count,
                        categories = EXCLUDED.categories,
                        thumbnail = EXCLUDED.thumbnail,
                        language = EXCLUDED.language,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    book.id, book.title, book.authors, book.published_date,
                    book.description, book.page_count, book.categories,
                    book.thumbnail, book.language
                ))
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to insert book: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)

    def get_book(self, book_id: str):
        """Get a book by ID."""
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, title, authors, published_date, description,
                           page_count, categories, thumbnail, language
                    FROM books WHERE id = %s
                """, (book_id,))

                row = cur.fetchone()
                if row:
                    from src.models import Book
                    return Book(*row)
                return None
        finally:
            self.connection_pool.putconn(conn)

    def search_books(self, query: str, limit: int = 10) -> List:
        """
        Search books by title (full-text search).

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of Book objects
        """
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, title, authors, published_date, description,
                           page_count, categories, thumbnail, language
                    FROM books
                    WHERE to_tsvector('english', title) @@ plainto_tsquery('english', %s)
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (query, limit))

                rows = cur.fetchall()
                from src.models import Book
                return [Book(*row) for row in rows]
        finally:
            self.connection_pool.putconn(conn)

    def cache_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached API response if not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached response or None
        """
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT response_data
                    FROM api_cache
                    WHERE cache_key = %s AND expires_at > CURRENT_TIMESTAMP
                """, (cache_key,))

                row = cur.fetchone()
                if row:
                    logger.info(f"Cache hit: {cache_key}")
                    return row[0]  # JSONB is automatically deserialized

                logger.info(f"Cache miss: {cache_key}")
                return None
        finally:
            self.connection_pool.putconn(conn)

    def cache_set(
        self,
        cache_key: str,
        response_data: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> bool:
        """
        Cache API response with TTL.

        Args:
            cache_key: Cache key
            response_data: Response to cache
            ttl_seconds: Time to live in seconds

        Returns:
            True if successful
        """
        conn = self.connection_pool.getconn()
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO api_cache (cache_key, response_data, expires_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        response_data = EXCLUDED.response_data,
                        expires_at = EXCLUDED.expires_at,
                        created_at = CURRENT_TIMESTAMP
                """, (cache_key, json.dumps(response_data), expires_at))

                conn.commit()
                logger.info(f"Cached response: {cache_key} (TTL: {ttl_seconds}s)")
                return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to cache response: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Count books
                cur.execute("SELECT COUNT(*) FROM books")
                book_count = cur.fetchone()[0]

                # Count cache entries
                cur.execute("SELECT COUNT(*) FROM api_cache WHERE expires_at > CURRENT_TIMESTAMP")
                cache_count = cur.fetchone()[0]

                # Count expired cache
                cur.execute("SELECT COUNT(*) FROM api_cache WHERE expires_at <= CURRENT_TIMESTAMP")
                expired_count = cur.fetchone()[0]

                return {
                    "total_books": book_count,
                    "cached_responses": cache_count,
                    "expired_cache_entries": expired_count
                }
        finally:
            self.connection_pool.putconn(conn)

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries."""
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM api_cache
                    WHERE expires_at <= CURRENT_TIMESTAMP
                """)
                deleted = cur.rowcount
                conn.commit()
                logger.info(f"Cleaned up {deleted} expired cache entries")
                return deleted
        finally:
            self.connection_pool.putconn(conn)

    def close(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
```

### Step 3: Update Client to Use Cache

**File: `src/client.py`** (add caching method)

```python
# Add to GoogleBooksClient class:

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
```

### Step 4: Configuration Management

**File: `src/config.py`**

```python
"""Configuration management."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""

    # Database
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "booksdb")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    @property
    def DATABASE_URL(self):
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # API
    GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

    # Defaults
    DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "10"))
    DEFAULT_MAX_RETRIES = int(os.getenv("DEFAULT_MAX_RETRIES", "3"))
    DEFAULT_CACHE_TTL = int(os.getenv("DEFAULT_CACHE_TTL", "3600"))
```

**File: `.env.example`**

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=booksdb
DB_USER=postgres
DB_PASSWORD=your_password_here

# Google Books API (optional - increases rate limits)
GOOGLE_BOOKS_API_KEY=

# Client Configuration
DEFAULT_TIMEOUT=10
DEFAULT_MAX_RETRIES=3
DEFAULT_CACHE_TTL=3600
```

### Step 5: Test Lab 2

**File: `test_lab2.py`**

```python
"""Test Lab 2: Database and caching."""
from src.client import GoogleBooksClient
from src.parse import parse_books_response
from src.database import Database
from src.config import Config
import time


def main():
    """Test database and caching."""
    config = Config()

    print("=" * 60)
    print("Lab 2 Test: Database-backed caching")
    print("=" * 60)

    # Initialize database
    print("\n1. Initializing database...")
    with Database(config.DATABASE_URL) as db:
        db.init_schema()
        print("✅ Database schema created")

        # Create client
        with GoogleBooksClient() as client:
            query = "machine learning"

            # First request (cache miss)
            print(f"\n2. First request for '{query}' (should hit API)...")
            start = time.time()
            response = client.search_with_cache(
                query,
                max_results=5,
                cache_db=db,
                cache_ttl=300
            )
            first_duration = time.time() - start

            if response:
                books = parse_books_response(response)
                print(f"✅ Found {len(books)} books in {first_duration:.2f}s")

                # Store books in database
                for book in books:
                    db.insert_book(book)
                print(f"✅ Stored {len(books)} books in database")

            # Second request (cache hit)
            print(f"\n3. Second request for '{query}' (should hit cache)...")
            start = time.time()
            response = client.search_with_cache(
                query,
                max_results=5,
                cache_db=db,
                cache_ttl=300
            )
            second_duration = time.time() - start

            if response:
                print(f"✅ Retrieved from cache in {second_duration:.2f}s")
                print(f"⚡ Speed-up: {first_duration/second_duration:.1f}x faster!")

            # Show stats
            print("\n4. Database statistics:")
            stats = db.get_stats()
            print(f"   Total books: {stats['total_books']}")
            print(f"   Cached responses: {stats['cached_responses']}")
            print(f"   Expired cache: {stats['expired_cache_entries']}")

            # Test search
            print(f"\n5. Searching database for '{query}'...")
            db_books = db.search_books(query, limit=3)
            print(f"✅ Found {len(db_books)} books in local database")
            for book in db_books:
                print(f"   - {book.title} by {book.authors_str}")


if __name__ == "__main__":
    main()
```

### Setup Database

```bash
# Create database
createdb booksdb

# Copy and configure environment
cp .env.example .env
# Edit .env with your database password

# Run test
python test_lab2.py
```

---

## Session C: Async Parallelism & CLI (Lab 3)

### Learning Goals

- Implement async/await for concurrent requests
- Use Semaphore to cap concurrency
- Build a production CLI with argparse
- Generate reports from database

### Step 1: Install Async Dependencies

```bash
pip install httpx asyncio
```

### Step 2: Create Async Client

**File: `src/async_client.py`**

```python
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
```

### Step 3: Build CLI

**File: `explorer.py`** (main CLI script)

```python
#!/usr/bin/env python3
"""Book Explorer CLI - Database & API Integration."""
import argparse
import asyncio
import sys
import json
from tabulate import tabulate
from src.client import GoogleBooksClient
from src.async_client import AsyncGoogleBooksClient
from src.database import Database
from src.parse import parse_books_response, deduplicate_books
from src.config import Config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_database(config: Config) -> Database:
    """Initialize database."""
    db = Database(config.DATABASE_URL)
    db.init_schema()
    return db


async def search_books_async(args, config: Config):
    """Search for books using async client."""
    db = setup_database(config)

    try:
        async with AsyncGoogleBooksClient(
            api_key=config.GOOGLE_BOOKS_API_KEY,
            timeout=config.DEFAULT_TIMEOUT,
            max_concurrent=args.parallel
        ) as client:

            logger.info(f"Searching for: {args.query}")
            logger.info(f"Parallel requests: {args.parallel}")
            logger.info(f"Cache TTL: {args.cache_ttl}s")

            # Check cache first
            cache_key = f"books:search:{args.query}:{args.limit}:0"
            cached = db.cache_get(cache_key)

            if cached and not args.no_cache:
                logger.info("✅ Cache hit - using cached data")
                response = cached
            else:
                logger.info("⚠️  Cache miss - fetching from API")

                # Fetch from API (paginated if needed)
                if args.limit > 40:
                    responses = await client.paginated_search(
                        args.query,
                        total_results=args.limit,
                        results_per_page=40
                    )
                    # Merge responses
                    all_items = []
                    for resp in responses:
                        all_items.extend(resp.get("items", []))
                    response = {"items": all_items}
                else:
                    response = await client.search(args.query, args.limit)

                # Cache the response
                if response and not args.no_cache:
                    db.cache_set(cache_key, response, args.cache_ttl)

            # Parse books
            books = parse_books_response(response)
            books = deduplicate_books(books)
            books = books[:args.limit]  # Limit results

            logger.info(f"Found {len(books)} books")

            # Store in database
            for book in books:
                db.insert_book(book)
            logger.info(f"Stored {len(books)} books in database")

            # Display results
            display_books(books, args.format)

    finally:
        db.close()


def search_books_sync(args, config: Config):
    """Search for books using sync client."""
    db = setup_database(config)

    try:
        with GoogleBooksClient(
            api_key=config.GOOGLE_BOOKS_API_KEY,
            timeout=config.DEFAULT_TIMEOUT,
            max_retries=config.DEFAULT_MAX_RETRIES
        ) as client:

            # Use caching
            response = client.search_with_cache(
                args.query,
                max_results=args.limit,
                cache_db=db if not args.no_cache else None,
                cache_ttl=args.cache_ttl
            )

            if not response:
                logger.error("Failed to fetch data")
                return

            # Parse and store
            books = parse_books_response(response)
            books = deduplicate_books(books)

            for book in books:
                db.insert_book(book)

            display_books(books, args.format)

    finally:
        db.close()


def display_books(books, format_type: str):
    """Display books in specified format."""
    if format_type == "table":
        headers = ["Title", "Authors", "Published", "Pages", "Categories"]
        rows = [
            [
                book.title[:50] + "..." if len(book.title) > 50 else book.title,
                book.authors_str[:30] + "..." if len(book.authors_str) > 30 else book.authors_str,
                book.published_date or "Unknown",
                book.page_count or "N/A",
                book.categories_str[:30] + "..." if len(book.categories_str) > 30 else book.categories_str
            ]
            for book in books
        ]
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    elif format_type == "json":
        books_dict = [
            {
                "id": book.id,
                "title": book.title,
                "authors": book.authors,
                "published_date": book.published_date,
                "description": book.description,
                "page_count": book.page_count,
                "categories": book.categories,
                "thumbnail": book.thumbnail,
                "language": book.language
            }
            for book in books
        ]
        print(json.dumps(books_dict, indent=2))

    elif format_type == "compact":
        for i, book in enumerate(books, 1):
            print(f"{i}. {book.title} - {book.authors_str}")


def show_stats(args, config: Config):
    """Show database statistics."""
    db = setup_database(config)

    try:
        stats = db.get_stats()

        print("\n" + "=" * 50)
        print("DATABASE STATISTICS")
        print("=" * 50)
        print(f"Total books stored: {stats['total_books']}")
        print(f"Cached API responses: {stats['cached_responses']}")
        print(f"Expired cache entries: {stats['expired_cache_entries']}")
        print("=" * 50 + "\n")

        # Cleanup if requested
        if args.cleanup:
            deleted = db.cleanup_expired_cache()
            print(f"✅ Cleaned up {deleted} expired cache entries\n")

    finally:
        db.close()


def export_data(args, config: Config):
    """Export database data."""
    db = setup_database(config)

    try:
        books = db.search_books("", limit=args.limit or 1000)

        if args.format == "json":
            data = [
                {
                    "id": book.id,
                    "title": book.title,
                    "authors": book.authors,
                    "published_date": book.published_date,
                    "page_count": book.page_count,
                    "categories": book.categories,
                    "language": book.language
                }
                for book in books
            ]

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"✅ Exported {len(books)} books to {args.output}")
            else:
                print(json.dumps(data, indent=2))

        elif args.format == "csv":
            import csv

            output_file = args.output or "books_export.csv"
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Title", "Authors", "Published", "Pages", "Categories", "Language"])

                for book in books:
                    writer.writerow([
                        book.id,
                        book.title,
                        book.authors_str,
                        book.published_date or "",
                        book.page_count or "",
                        book.categories_str,
                        book.language
                    ])

            logger.info(f"✅ Exported {len(books)} books to {output_file}")

    finally:
        db.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Book Explorer - Database & API Integration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search with defaults
  %(prog)s search "python programming"

  # Parallel search with custom cache TTL
  %(prog)s search "machine learning" --limit 20 --parallel 5 --cache-ttl 7200

  # Export data
  %(prog)s export --format json --output books.json

  # Show statistics
  %(prog)s stats --cleanup
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for books")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    search_parser.add_argument("--format", choices=["table", "json", "compact"], default="table", help="Output format")
    search_parser.add_argument("--parallel", type=int, default=5, help="Concurrent requests (default: 5)")
    search_parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL in seconds (default: 3600)")
    search_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    search_parser.add_argument("--async", dest="use_async", action="store_true", help="Use async client")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--cleanup", action="store_true", help="Clean up expired cache")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export database data")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json", help="Export format")
    export_parser.add_argument("--output", help="Output file (default: stdout for JSON)")
    export_parser.add_argument("--limit", type=int, help="Limit results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = Config()

    try:
        if args.command == "search":
            if args.use_async:
                asyncio.run(search_books_async(args, config))
            else:
                search_books_sync(args, config)

        elif args.command == "stats":
            show_stats(args, config)

        elif args.command == "export":
            export_data(args, config)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 4: Final Dependencies

**File: `requirements.txt`**

```
requests>=2.31.0
httpx>=0.25.0
psycopg2-binary>=2.9.9
python-dotenv>=1.0.0
tabulate>=0.9.0
```

### Step 5: Run Complete System

```bash
# Install all dependencies
pip install -r requirements.txt

# Setup database
createdb booksdb

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run searches
python explorer.py search "python programming" --format table
python explorer.py search "machine learning" --limit 20 --parallel 5 --async
python explorer.py search "data science" --cache-ttl 7200

# View statistics
python explorer.py stats

# Export data
python explorer.py export --format json --output books.json
python explorer.py export --format csv --output books.csv

# Clean up cache
python explorer.py stats --cleanup
```

---

## Testing Your Implementation

Run the test suite:

```bash
# Unit tests
pytest tests/

# Integration test
python test_lab1.py  # Test API client
python test_lab2.py  # Test database + caching
```

---

## Success Criteria

By completing this lab, you should be able to:

✅ Make HTTP requests with proper timeout and retry logic  
✅ Parse complex JSON responses safely  
✅ Design PostgreSQL schemas with proper indexes  
✅ Implement database-backed caching with TTL  
✅ Use async/await for concurrent API requests  
✅ Limit concurrency with Semaphore  
✅ Build a production CLI with argparse  
✅ Handle errors gracefully at every layer  
✅ Generate reports from database queries

---

## Common Issues & Solutions

**Database connection fails:**

```bash
# Check PostgreSQL is running
brew services list

# Restart PostgreSQL
brew services restart postgresql@15

# Verify connection
psql -d booksdb -c "SELECT 1"
```

**Rate limiting (429 errors):**

- Reduce `--parallel` value (try 2-3)
- Increase cache TTL
- Add Google Books API key to `.env`

**Async is confusing:**

- Start with sync client first
- Read logs carefully to understand execution order
- Remember: `await` pauses execution until I/O completes

**Tests failing:**

- Check database is initialized: `python -c "from src.database import Database; from src.config import Config; db = Database(Config().DATABASE_URL); db.init_schema()"`
- Verify .env file exists and is configured
- Check imports are correct

---

## Next Steps

After completing this lab:

1. **Add more APIs**: Try combining Google Books with Open Library or Goodreads
2. **Enhance caching**: Implement cache invalidation strategies
3. **Add monitoring**: Track API usage, cache hit rates, query performance
4. **Scale up**: Test with larger datasets (1000+ books)
5. **Optimize queries**: Use EXPLAIN ANALYZE to find slow queries

This lab gives you the complete foundation for building production LLM applications that integrate external data sources reliably and efficiently.
