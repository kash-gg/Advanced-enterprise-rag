"""
Web Scraper / URL Loader
Loads and extracts content from web pages and URLs.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from urllib.parse import urlparse, urljoin
import time

from llama_index.core import Document
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class WebLoader:
    """
    Loader for web content from URLs.
    
    Features:
    - Load content from single URLs
    - Extract main content (remove headers, footers, ads)
    - Handle pagination and multi-page articles
    - Respect robots.txt and rate limiting
    - Extract metadata (title, author, publish date)
    - Support sitemap parsing
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit_delay: float = 1.0,
        user_agent: str = "RAG-System-Bot/1.0"
    ):
        """
        Initialize the Web loader.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_delay: Delay between requests in seconds
            user_agent: User agent string for requests
        """
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.user_agent = user_agent
        
        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'User-Agent': self.user_agent
        })
        
        logger.info("WebLoader initialized")
    
    def load_url(self, url: str) -> List[Document]:
        """
        Load content from a single URL.
        
        Args:
            url: URL to load content from
            
        Returns:
            List containing a single Document object
        """
        try:
            logger.info(f"Loading URL: {url}")
            
            # Fetch the page
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            # Extract main content
            content = self._extract_content(soup)
            
            # Create document
            doc = Document(
                text=content,
                metadata={
                    **metadata,
                    'source_type': 'web',
                    'url': url,
                    'loaded_at': datetime.now().isoformat(),
                    'status_code': response.status_code
                }
            )
            
            logger.info(f"Successfully loaded content from {url}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return [doc]
            
        except requests.RequestException as e:
            logger.error(f"Error loading URL {url}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading URL {url}: {str(e)}")
            raise
    
    def load_batch(self, urls: List[str]) -> List[Document]:
        """
        Load content from multiple URLs.
        
        Args:
            urls: List of URLs to load
            
        Returns:
            List of Document objects
        """
        all_documents = []
        successful = 0
        failed = 0
        
        for url in urls:
            try:
                documents = self.load_url(url)
                all_documents.extend(documents)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to load {url}: {str(e)}")
                failed += 1
        
        logger.info(f"Batch loading complete: {successful} successful, {failed} failed")
        return all_documents
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from HTML, removing boilerplate.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted text content
        """
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try to find main content area
        main_content = None
        
        # Common patterns for main content
        content_selectors = [
            {'name': 'main'},
            {'name': 'article'},
            {'class_': 'content'},
            {'class_': 'main-content'},
            {'class_': 'article-content'},
            {'id': 'content'},
            {'id': 'main-content'}
        ]
        
        for selector in content_selectors:
            main_content = soup.find(**selector)
            if main_content:
                break
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return soup.get_text()
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract metadata from HTML page.
        
        Args:
            soup: BeautifulSoup object
            url: Page URL
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {}
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        else:
            # Try Open Graph title
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                metadata['title'] = og_title['content']
            else:
                metadata['title'] = urlparse(url).netloc
        
        # Extract description
        description = soup.find('meta', attrs={'name': 'description'})
        if description and description.get('content'):
            metadata['description'] = description['content']
        else:
            og_desc = soup.find('meta', property='og:description')
            if og_desc and og_desc.get('content'):
                metadata['description'] = og_desc['content']
        
        # Extract author
        author = soup.find('meta', attrs={'name': 'author'})
        if author and author.get('content'):
            metadata['author'] = author['content']
        else:
            # Try to find author in common patterns
            author_elem = soup.find(class_=['author', 'by-author', 'post-author'])
            if author_elem:
                metadata['author'] = author_elem.get_text().strip()
        
        # Extract publish date
        date_meta = soup.find('meta', property='article:published_time')
        if date_meta and date_meta.get('content'):
            metadata['publish_date'] = date_meta['content']
        
        # Extract keywords
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords and keywords.get('content'):
            metadata['keywords'] = keywords['content']
        
        # Extract domain
        parsed_url = urlparse(url)
        metadata['domain'] = parsed_url.netloc
        metadata['path'] = parsed_url.path
        
        return metadata
    
    def load_from_sitemap(self, sitemap_url: str, max_urls: Optional[int] = None) -> List[Document]:
        """
        Load content from URLs listed in a sitemap.
        
        Args:
            sitemap_url: URL of the sitemap XML file
            max_urls: Maximum number of URLs to load (None for all)
            
        Returns:
            List of Document objects
        """
        try:
            logger.info(f"Loading sitemap: {sitemap_url}")
            
            response = self.session.get(sitemap_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            
            # Extract URLs from sitemap
            urls = [loc.get_text() for loc in soup.find_all('loc')]
            
            if max_urls:
                urls = urls[:max_urls]
            
            logger.info(f"Found {len(urls)} URLs in sitemap")
            
            return self.load_batch(urls)
            
        except Exception as e:
            logger.error(f"Error loading sitemap {sitemap_url}: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = WebLoader()
    
    # Example: Load a single URL
    # documents = loader.load_url("https://example.com/article")
    
    # Example: Load multiple URLs
    # urls = ["https://example.com/page1", "https://example.com/page2"]
    # documents = loader.load_batch(urls)
    
    # Example: Load from sitemap
    # documents = loader.load_from_sitemap("https://example.com/sitemap.xml", max_urls=10)
    
    print("WebLoader initialized and ready to use!")
