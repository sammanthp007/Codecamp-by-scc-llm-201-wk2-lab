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
