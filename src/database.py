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
                if query:
                    cur.execute("""
                        SELECT id, title, authors, published_date, description,
                               page_count, categories, thumbnail, language
                        FROM books
                        WHERE to_tsvector('english', title) @@ plainto_tsquery('english', %s)
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (query, limit))
                else:
                    # Return all books if no query
                    cur.execute("""
                        SELECT id, title, authors, published_date, description,
                               page_count, categories, thumbnail, language
                        FROM books
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))
                
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
