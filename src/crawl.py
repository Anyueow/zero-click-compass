"""
Web crawling module for downloading and sanitizing web pages.
"""
import os
import time
import requests
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from .utils import (
    retry_on_failure, sanitize_text, extract_domain, 
    is_valid_url, get_env_var, create_data_dir, save_jsonl,
    logger, safe_request
)

class WebCrawler:
    """Crawler for downloading and sanitizing web pages."""
    
    def __init__(self, max_pages: int = 100, delay: float = 1.0):
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.data_dir = create_data_dir()
        
        # Initialize Selenium for JavaScript-heavy pages
        self.driver = None
        self._setup_selenium()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver for JavaScript rendering."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            logger.info("Selenium WebDriver initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Selenium: {e}")
            self.driver = None
    
    def crawl_website(self, start_url: str, max_depth: int = 2) -> List[Dict]:
        """Crawl a website starting from the given URL."""
        if not is_valid_url(start_url):
            raise ValueError(f"Invalid URL: {start_url}")
        
        logger.info(f"Starting crawl of {start_url}")
        pages = []
        urls_to_visit = [(start_url, 0)]  # (url, depth)
        
        while urls_to_visit and len(pages) < self.max_pages:
            url, depth = urls_to_visit.pop(0)
            
            if url in self.visited_urls or depth > max_depth:
                continue
            
            self.visited_urls.add(url)
            
            try:
                page_data = self._crawl_page(url)
                if page_data:
                    pages.append(page_data)
                    logger.info(f"Crawled page {len(pages)}/{self.max_pages}: {url}")
                    
                    # Find more URLs to crawl
                    if depth < max_depth:
                        new_urls = self._extract_links(page_data['html'], url)
                        for new_url in new_urls:
                            if (new_url not in self.visited_urls and 
                                extract_domain(new_url) == extract_domain(start_url)):
                                urls_to_visit.append((new_url, depth + 1))
                
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
        
        # Save crawled pages
        output_file = os.path.join(self.data_dir, "crawled_pages.jsonl")
        save_jsonl(pages, output_file)
        logger.info(f"Saved {len(pages)} pages to {output_file}")
        
        return pages
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _crawl_page(self, url: str) -> Optional[Dict]:
        """Crawl a single page and extract content."""
        # Try regular requests first
        response = safe_request(url)
        if not response:
            return None
        
        html_content = response.text
        
        # If page seems to have JavaScript content, try Selenium
        if self.driver and self._needs_javascript(html_content):
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(2)  # Let JavaScript load
                html_content = self.driver.page_source
            except Exception as e:
                logger.warning(f"Selenium failed for {url}: {e}")
        
        # Parse and extract content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract text content
        text_content = soup.get_text()
        text_content = sanitize_text(text_content)
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description.get('content', '') if meta_description else ""
        
        # Extract headings
        headings = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for heading in soup.find_all(tag):
                headings.append({
                    'level': int(tag[1]),
                    'text': sanitize_text(heading.get_text())
                })
        
        return {
            'url': url,
            'title': title_text,
            'description': description,
            'text': text_content,
            'html': html_content,
            'headings': headings,
            'domain': extract_domain(url),
            'crawled_at': time.time()
        }
    
    def _needs_javascript(self, html_content: str) -> bool:
        """Check if page likely needs JavaScript rendering."""
        indicators = [
            'react', 'vue', 'angular', 'spa', 'single-page',
            'loading', 'spinner', 'dynamic', 'ajax'
        ]
        html_lower = html_content.lower()
        return any(indicator in html_lower for indicator in indicators)
    
    def _extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract all links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            if is_valid_url(absolute_url):
                links.append(absolute_url)
        
        return list(set(links))  # Remove duplicates
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            logger.info("Selenium WebDriver closed")

def crawl_single_page(url: str) -> Optional[Dict]:
    """Crawl a single page for quick testing."""
    crawler = WebCrawler(max_pages=1)
    try:
        pages = crawler.crawl_website(url, max_depth=0)
        return pages[0] if pages else None
    finally:
        crawler.close()

def crawl_multiple_pages(urls: List[str]) -> List[Dict]:
    """Crawl multiple pages from a list of URLs."""
    crawler = WebCrawler(max_pages=len(urls))
    try:
        pages = []
        for url in urls:
            page = crawler._crawl_page(url)
            if page:
                pages.append(page)
            time.sleep(crawler.delay)
        
        # Save results
        output_file = os.path.join(crawler.data_dir, "crawled_pages.jsonl")
        save_jsonl(pages, output_file)
        
        return pages
    finally:
        crawler.close() 