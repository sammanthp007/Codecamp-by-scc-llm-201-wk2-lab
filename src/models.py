"""Data models for books."""
from dataclasses import dataclass
from typing import Optional, List


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
