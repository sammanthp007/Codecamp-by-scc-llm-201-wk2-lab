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


if __name__ == "__main__":
    # Run tests
    test_parse_book_complete()
    test_parse_book_missing_fields()
    test_parse_book_no_id()
    test_parse_books_response()
    test_deduplicate_books()
    print("✅ All tests passed!")
