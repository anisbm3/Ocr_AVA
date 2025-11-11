"""
Web Scraper Module - Extract information from candidate online profiles
"""
import requests
import re
import logging
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

@dataclass
class WebScrapingResult:
    """Web scraping result"""
    url: str
    content: str
    title: str
    success: bool
    error_message: Optional[str] = None
    scraped_data: Dict[str, Any] = None

class WebScraper:
    """Web scraper for extracting additional candidate information"""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session() if requests else None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_urls_from_cv(self, cv_text: str) -> List[str]:
        """Extract URLs from CV text"""
        urls = []

        # Common URL patterns
        url_patterns = [
            r'https?://[^\s<>"\']+',
            r'www\.[^\s<>"\']+',
            r'github\.com/[^\s<>"\']+',
            r'linkedin\.com/in/[^\s<>"\']+',
            r'[a-zA-Z0-9-]+\.github\.io[^\s<>"\']*'
        ]

        for pattern in url_patterns:
            matches = re.findall(pattern, cv_text, re.IGNORECASE)
            for match in matches:
                # Clean up the URL
                url = match.strip('.,;:!?')
                if not url.startswith('http'):
                    url = 'https://' + url
                urls.append(url)

        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def scrape_github(self, url: str) -> WebScrapingResult:
        """Scrape GitHub profile and repositories"""
        try:
            if not self.session:
                return WebScrapingResult(url, "", "", False, "Requests not available")

            response = self.session.get(url, headers=self.headers, timeout=self.timeout)

            if response.status_code == 200 and BeautifulSoup:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract GitHub profile information
                profile_data = {}

                # Get profile name and bio
                name_elem = soup.find('span', {'class': 'p-name'})
                bio_elem = soup.find('div', {'class': 'p-note'})

                profile_data['name'] = name_elem.get_text().strip() if name_elem else ""
                profile_data['bio'] = bio_elem.get_text().strip() if bio_elem else ""

                # Get repositories information
                repos = []
                repo_elements = soup.find_all('a', {'data-hovercard-type': 'repository'})
                for repo in repo_elements[:10]:  # Limit to first 10 repos
                    repo_name = repo.get_text().strip()
                    if repo_name:
                        repos.append(repo_name)

                profile_data['repositories'] = repos
                profile_data['public_repos'] = len(repos)

                # Get languages used
                languages = []
                lang_elements = soup.find_all('span', {'class': 'color-fg-default'})
                for lang in lang_elements:
                    lang_text = lang.get_text().strip()
                    if lang_text and len(lang_text) < 20:  # Likely a programming language
                        languages.append(lang_text)

                profile_data['languages'] = list(set(languages))[:10]  # Unique languages, max 10

                # Compile content
                content_parts = []
                if profile_data['name']:
                    content_parts.append(f"Name: {profile_data['name']}")
                if profile_data['bio']:
                    content_parts.append(f"Bio: {profile_data['bio']}")
                if profile_data['repositories']:
                    content_parts.append(f"Repositories: {', '.join(profile_data['repositories'][:5])}")
                if profile_data['languages']:
                    content_parts.append(f"Languages: {', '.join(profile_data['languages'])}")

                content = ". ".join(content_parts)

                return WebScrapingResult(
                    url=url,
                    content=content,
                    title=soup.find('title').get_text() if soup.find('title') else "",
                    success=True,
                    scraped_data=profile_data
                )
            else:
                return WebScrapingResult(url, "", "", False, f"HTTP {response.status_code}")

        except Exception as e:
            return WebScrapingResult(url, "", "", False, str(e))

    def scrape_linkedin(self, url: str) -> WebScrapingResult:
        """Scrape LinkedIn profile (limited due to anti-scraping measures)"""
        try:
            if not self.session:
                return WebScrapingResult(url, "", "", False, "Requests not available")

            response = self.session.get(url, headers=self.headers, timeout=self.timeout)

            if response.status_code == 200 and BeautifulSoup:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract basic information (LinkedIn heavily restricts scraping)
                title_tag = soup.find('title')
                title = title_tag.get_text() if title_tag else ""

                # Look for any visible text content
                content = soup.get_text()[:1000]  # First 1000 chars

                return WebScrapingResult(
                    url=url,
                    content=content,
                    title=title,
                    success=True,
                    scraped_data={
                        "platform": "linkedin",
                        "profile_title": title,
                        "limited_access": True,
                        "note": "LinkedIn restricts automated access"
                    }
                )
            else:
                return WebScrapingResult(url, "", "", False, f"HTTP {response.status_code}")

        except Exception as e:
            return WebScrapingResult(url, "", "", False, str(e))

    def scrape_personal_website(self, url: str) -> WebScrapingResult:
        """Scrape personal website or portfolio"""
        try:
            if not self.session:
                return WebScrapingResult(url, "", "", False, "Requests not available")

            response = self.session.get(url, headers=self.headers, timeout=self.timeout)

            if response.status_code == 200 and BeautifulSoup:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get title
                title = soup.find('title').get_text() if soup.find('title') else ""

                # Extract main content
                content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                content_parts = []

                for elem in content_elements:
                    text = elem.get_text().strip()
                    if text and len(text) > 10 and len(text) < 500:  # Filter meaningful content
                        content_parts.append(text)

                content = ". ".join(content_parts[:20])  # Limit to first 20 meaningful paragraphs

                # Look for skills, projects, experience keywords
                skills_found = []
                skill_keywords = [
                    'python', 'javascript', 'java', 'react', 'angular', 'vue', 'node',
                    'django', 'flask', 'express', 'mongodb', 'postgresql', 'mysql',
                    'aws', 'azure', 'docker', 'kubernetes', 'git', 'machine learning',
                    'data science', 'artificial intelligence', 'web development'
                ]

                content_lower = content.lower()
                for skill in skill_keywords:
                    if skill in content_lower:
                        skills_found.append(skill)

                return WebScrapingResult(
                    url=url,
                    content=content,
                    title=title,
                    success=True,
                    scraped_data={
                        "platform": "personal_website",
                        "skills_found": skills_found,
                        "content_length": len(content)
                    }
                )
            else:
                return WebScrapingResult(url, "", "", False, f"HTTP {response.status_code}")

        except Exception as e:
            return WebScrapingResult(url, "", "", False, str(e))

    def scrape_url(self, url: str) -> WebScrapingResult:
        """Scrape URL based on its type"""
        try:
            domain = urlparse(url).netloc.lower()

            if 'linkedin.com' in domain:
                return self.scrape_linkedin(url)
            elif 'github.com' in domain:
                return self.scrape_github(url)
            else:
                return self.scrape_personal_website(url)

        except Exception as e:
            return WebScrapingResult(url, "", "", False, str(e))