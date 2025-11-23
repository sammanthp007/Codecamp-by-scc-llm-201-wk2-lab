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
