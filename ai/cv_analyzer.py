import asyncio
import json
import logging
import time
import numpy as np
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
from pathlib import Path
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# Enhanced error handling for imports
DEPENDENCIES = {
    'sentence_transformers': False,
    'torch': False,
    'chromadb': False,
    'faiss': False,
    'transformers': False,
    'beautifulsoup4': False,
    'requests': False
}

try:
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES['sentence_transformers'] = True
except ImportError:
    SentenceTransformer = None

try:
    import torch
    import torch.nn.functional as F
    DEPENDENCIES['torch'] = True
except ImportError:
    torch = None
    F = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    DEPENDENCIES['chromadb'] = True
except ImportError:
    chromadb = None

try:
    import faiss
    DEPENDENCIES['faiss'] = True
except ImportError:
    faiss = None

try:
    from transformers import AutoTokenizer, AutoModel
    DEPENDENCIES['transformers'] = True
except ImportError:
    AutoTokenizer = None
    AutoModel = None

try:
    from bs4 import BeautifulSoup
    DEPENDENCIES['beautifulsoup4'] = True
except ImportError:
    BeautifulSoup = None

try:
    import requests
    DEPENDENCIES['requests'] = True
except ImportError:
    requests = None

# Import base agent
from .base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    success: bool
    model_used: str
    tokens_used: int
    processing_time: float
    error_message: Optional[str] = None

@dataclass
class RAGSearchResult:
    """RAG search result with metadata"""
    documents: List[str]
    scores: List[float]
    metadata: List[Dict[str, Any]]
    query: str
    total_found: int

@dataclass
class WebScrapingResult:
    """Web scraping result"""
    url: str
    content: str
    title: str
    success: bool
    error_message: Optional[str] = None
    scraped_data: Dict[str, Any] = None

@dataclass
class EnhancedCVAnalysis:
    """Enhanced CV analysis result"""
    candidate_name: str
    basic_score: float
    rag_enhanced_score: float
    final_score: float
    skills_analysis: Dict[str, Any]
    experience_analysis: Dict[str, Any]
    web_data: Dict[str, Any]
    llm_insights: str
    recommendations: List[str]
    confidence: float
    processing_time: float

class DependencyChecker:
    """Check and report on required dependencies"""

    @staticmethod
    def check_all() -> Dict[str, bool]:
        """Check all dependencies and return status"""
        return DEPENDENCIES.copy()

    @staticmethod
    def get_missing() -> List[str]:
        """Get list of missing dependencies"""
        return [dep for dep, available in DEPENDENCIES.items() if not available]

    @staticmethod
    def is_ready_for_rag() -> bool:
        """Check if minimum dependencies for RAG are available"""
        required = ['sentence_transformers']
        return all(DEPENDENCIES.get(dep, False) for dep in required)

    @staticmethod
    def print_status():
        """Print dependency status"""
        print("=== Dependency Status ===")
        for dep, available in DEPENDENCIES.items():
            status = "✓" if available else "✗"
            print(f"{status} {dep}: {'Available' if available else 'Missing'}")

        missing = DependencyChecker.get_missing()
        if missing:
            print(f"\nTo install missing dependencies:")
            print(f"pip install {' '.join(missing)}")

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

    def scrape_multiple_urls(self, urls: List[str]) -> List[WebScrapingResult]:
        """Scrape multiple URLs"""
        results = []
        for url in urls[:5]:  # Limit to 5 URLs to avoid overloading
            logger.info(f"Scraping URL: {url}")
            result = self.scrape_url(url)
            results.append(result)
            time.sleep(1)  # Be respectful to servers
        return results

class LMStudioClient:
    """Enhanced LM Studio client with better error handling"""

    def __init__(self, base_url: str = "http://localhost:1234", timeout: int = 120):
        self.base_url = base_url
        self.timeout = timeout
        self.session = None
        self.available_models = []
        self.is_connected = False

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(limit=10)
        )
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def initialize(self):
        """Initialize and check LM Studio connection"""
        try:
            self.is_connected = await self.health_check()
            if self.is_connected:
                self.available_models = await self.get_available_models()
                logger.info(f"LM Studio connected. Models: {self.available_models}")
            else:
                logger.warning("LM Studio not available - will use fallback methods")
        except Exception as e:
            logger.error(f"LM Studio initialization error: {e}")
            self.is_connected = False

    async def health_check(self) -> bool:
        """Check if LM Studio is available"""
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"LM Studio health check failed: {e}")
            return False

    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model['id'] for model in data.get('data', [])]
                return []
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

    async def generate_completion(self, prompt: str, system_prompt: str = None,
                                model: str = None, max_tokens: int = 1024,
                                temperature: float = 0.3) -> LLMResponse:
        """Generate completion with enhanced error handling"""
        start_time = time.time()

        if not self.is_connected:
            return LLMResponse(
                content="",
                success=False,
                model_used="none",
                tokens_used=0,
                processing_time=time.time() - start_time,
                error_message="LM Studio not connected"
            )

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Use first available model if none specified
            model_to_use = model or (self.available_models[0] if self.available_models else "local-model")

            payload = {
                "model": model_to_use,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            async with self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload
            ) as response:

                processing_time = time.time() - start_time

                if response.status == 200:
                    data = await response.json()
                    return LLMResponse(
                        content=data['choices'][0]['message']['content'],
                        success=True,
                        model_used=data.get('model', model_to_use),
                        tokens_used=data.get('usage', {}).get('total_tokens', 0),
                        processing_time=processing_time
                    )
                else:
                    error_text = await response.text()
                    return LLMResponse(
                        content="",
                        success=False,
                        model_used=model_to_use,
                        tokens_used=0,
                        processing_time=processing_time,
                        error_message=f"HTTP {response.status}: {error_text}"
                    )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"LM Studio completion error: {e}")
            return LLMResponse(
                content="",
                success=False,
                model_used="error",
                tokens_used=0,
                processing_time=processing_time,
                error_message=str(e)
            )

class SimpleVectorStore:
    """Fixed vector store with proper embedding handling"""

    def __init__(self, persist_directory: str = "./simple_vectordb"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        self.embedding_model = None
        self.documents = {}  # collection_name -> List[Dict]
        self.embeddings = {}  # collection_name -> np.array

        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """Initialize embedding model with fallback"""
        if DEPENDENCIES['sentence_transformers']:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("SentenceTransformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")
                self.embedding_model = None
        else:
            logger.warning("SentenceTransformers not available - using keyword matching")

    def create_collection(self, name: str) -> bool:
        """Create a new collection"""
        try:
            self.documents[name] = []
            self.embeddings[name] = None  # Initialize as None instead of empty list
            logger.info(f"Collection '{name}' created")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False

    def add_documents(self, collection_name: str, documents: List[str],
                     metadatas: List[Dict[str, Any]] = None) -> bool:
        """Add documents to collection with fixed embedding handling"""
        try:
            if collection_name not in self.documents:
                self.create_collection(collection_name)

            # Prepare metadata
            if not metadatas:
                metadatas = [{"id": len(self.documents[collection_name]) + i}
                           for i in range(len(documents))]

            # Store documents with metadata
            for doc, meta in zip(documents, metadatas):
                self.documents[collection_name].append({
                    "content": doc,
                    "metadata": meta
                })

            # Generate embeddings if model available
            if self.embedding_model and documents:
                try:
                    doc_embeddings = self.embedding_model.encode(documents)

                    # Handle first batch of embeddings
                    if self.embeddings[collection_name] is None:
                        self.embeddings[collection_name] = doc_embeddings
                        logger.info(f"Created embeddings for {len(documents)} documents in {collection_name}")
                    else:
                        # Properly concatenate existing embeddings with new ones
                        self.embeddings[collection_name] = np.vstack([
                            self.embeddings[collection_name],
                            doc_embeddings
                        ])
                        logger.info(f"Added {len(documents)} documents with embeddings to {collection_name}")

                except Exception as e:
                    logger.warning(f"Failed to generate embeddings: {e}")
                    # Fall back to keyword search for this collection
                    self.embeddings[collection_name] = None
            else:
                logger.info(f"Added {len(documents)} documents (no embeddings) to {collection_name}")

            return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def similarity_search(self, collection_name: str, query: str,
                         n_results: int = 5) -> RAGSearchResult:
        """Perform similarity search with proper error handling"""
        try:
            if collection_name not in self.documents:
                return RAGSearchResult([], [], [], query, 0)

            docs_data = self.documents[collection_name]
            if not docs_data:
                return RAGSearchResult([], [], [], query, 0)

            # Try embedding-based search first if embeddings exist and are valid
            if (self.embedding_model and
                collection_name in self.embeddings and
                self.embeddings[collection_name] is not None and
                len(self.embeddings[collection_name]) > 0):
                return self._embedding_search(collection_name, query, n_results)
            else:
                # Fallback to keyword search
                return self._keyword_search(collection_name, query, n_results)

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return RAGSearchResult([], [], [], query, 0)

    def _embedding_search(self, collection_name: str, query: str, n_results: int) -> RAGSearchResult:
        """Embedding-based similarity search with proper error handling"""
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            doc_embeddings = self.embeddings[collection_name]

            # Ensure we have valid embeddings
            if doc_embeddings is None or len(doc_embeddings) == 0:
                logger.warning(f"No valid embeddings found for collection {collection_name}, falling back to keyword search")
                return self._keyword_search(collection_name, query, n_results)

            # Calculate similarities
            similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            # Get top results
            top_indices = np.argsort(similarities)[::-1][:n_results]

            docs_data = self.documents[collection_name]
            results = []
            scores = []
            metadata = []

            for idx in top_indices:
                if idx < len(docs_data):
                    results.append(docs_data[idx]["content"])
                    scores.append(float(similarities[idx]))
                    metadata.append(docs_data[idx]["metadata"])

            return RAGSearchResult(results, scores, metadata, query, len(docs_data))

        except Exception as e:
            logger.error(f"Embedding search error: {e}")
            return self._keyword_search(collection_name, query, n_results)

    def _keyword_search(self, collection_name: str, query: str, n_results: int) -> RAGSearchResult:
        """Keyword-based similarity search"""
        try:
            query_words = set(query.lower().split())
            docs_data = self.documents[collection_name]

            scored_docs = []
            for doc_data in docs_data:
                doc_words = set(doc_data["content"].lower().split())
                overlap = len(query_words.intersection(doc_words))
                score = overlap / max(len(query_words), 1)
                scored_docs.append((doc_data, score))

            # Sort by score and take top results
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = scored_docs[:n_results]

            results = [doc_data["content"] for doc_data, _ in top_docs]
            scores = [score for _, score in top_docs]
            metadata = [doc_data["metadata"] for doc_data, _ in top_docs]

            return RAGSearchResult(results, scores, metadata, query, len(docs_data))

        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return RAGSearchResult([], [], [], query, 0)

class EnhancedRAGProcessor:
    """RAG processor with improved reliability and web scraping"""

    def __init__(self, vector_store: SimpleVectorStore, llm_client: LMStudioClient):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.web_scraper = WebScraper()
        self.knowledge_initialized = False

    async def initialize_knowledge_base(self):
        """Initialize knowledge base with job-related information"""
        try:
            # Create collections
            self.vector_store.create_collection("job_skills")
            self.vector_store.create_collection("experience_patterns")
            self.vector_store.create_collection("education_requirements")
            self.vector_store.create_collection("web_data_context")

            # Add job skills knowledge
            skills_docs = [
                "Python programming requires experience with frameworks like Django, Flask, FastAPI, data science libraries like pandas, numpy, scikit-learn",
                "JavaScript development includes React, Angular, Vue.js, Node.js, Express, TypeScript, modern build tools",
                "Data science involves machine learning, pandas, numpy, scikit-learn, TensorFlow, PyTorch, statistics, data visualization",
                "DevOps requires Docker, Kubernetes, AWS, Azure, GCP, CI/CD pipelines, infrastructure as code, monitoring",
                "Database skills include SQL, PostgreSQL, MongoDB, Redis, database design, optimization, data modeling",
                "Frontend development needs HTML, CSS, JavaScript, responsive design, user experience, accessibility",
                "Backend development requires API design, database integration, security, scalability, microservices",
                "Mobile development uses React Native, Flutter, Swift, Kotlin, cross-platform frameworks",
                "Machine learning engineering involves MLOps, model deployment, feature engineering, model monitoring",
                "Cloud architecture requires system design, scalability, security, cost optimization, cloud services"
            ]

            self.vector_store.add_documents("job_skills", skills_docs)

            # Add experience patterns
            experience_docs = [
                "Senior level requires 5+ years of relevant experience, technical leadership, architecture decisions",
                "Mid-level positions need 2-5 years of experience, independent work, code reviews, mentoring",
                "Junior roles suitable for 0-2 years experience, learning attitude, basic programming skills",
                "Lead positions require team management, technical expertise, project planning, stakeholder communication",
                "Architect roles need system design, strategic thinking, technology evaluation, cross-team collaboration",
                "Principal engineer roles require deep technical expertise, industry influence, innovation leadership"
            ]

            self.vector_store.add_documents("experience_patterns", experience_docs)

            # Add education requirements
            education_docs = [
                "Computer Science degree provides strong programming foundation, algorithms, data structures, software engineering",
                "Engineering degrees offer problem-solving skills, analytical thinking, mathematics, technical knowledge",
                "Bootcamp graduates bring practical, focused skills, recent technology knowledge, career change motivation",
                "Self-taught developers show initiative, continuous learning, passion for technology, adaptability",
                "Advanced degrees indicate research experience, deep technical knowledge, academic rigor, specialization",
                "Certifications demonstrate specific technology expertise, continuous learning, professional development"
            ]

            self.vector_store.add_documents("education_requirements", education_docs)

            # Add web data context
            web_context_docs = [
                "GitHub repositories demonstrate coding skills, project experience, collaboration, code quality",
                "LinkedIn profiles show professional network, career progression, recommendations, industry involvement",
                "Personal websites showcase portfolio, communication skills, personal branding, technical abilities",
                "Open source contributions indicate community involvement, collaboration skills, code quality standards",
                "Technical blog posts demonstrate knowledge sharing, communication skills, continuous learning"
            ]

            self.vector_store.add_documents("web_data_context", web_context_docs)

            self.knowledge_initialized = True
            logger.info("Enhanced knowledge base initialized successfully")

        except Exception as e:
            logger.error(f"Knowledge base initialization error: {e}")
            self.knowledge_initialized = False

    async def analyze_with_rag(self, cv_text: str, job_description: str) -> Dict[str, Any]:
        """Perform RAG-enhanced analysis with web scraping"""
        try:
            if not self.knowledge_initialized:
                await self.initialize_knowledge_base()

            # Extract URLs from CV and scrape web data
            urls = self.web_scraper.extract_urls_from_cv(cv_text)
            web_results = []

            if urls:
                logger.info(f"Found {len(urls)} URLs in CV: {urls}")
                web_results = self.web_scraper.scrape_multiple_urls(urls)

            # Extract key information for RAG queries
            job_skills = self._extract_job_skills(job_description)
            job_level = self._extract_job_level(job_description)

            # Perform RAG searches
            skills_search = self.vector_store.similarity_search(
                "job_skills", f"skills required: {' '.join(job_skills)}", 3
            )

            experience_search = self.vector_store.similarity_search(
                "experience_patterns", f"experience level: {job_level}", 2
            )

            # Search web data context if we have web data
            web_search = RAGSearchResult([], [], [], "", 0)
            if web_results:
                web_platforms = [result.scraped_data.get('platform', 'website') if result.scraped_data else 'website'
                               for result in web_results if result.success]
                if web_platforms:
                    web_search = self.vector_store.similarity_search(
                        "web_data_context", f"web presence: {' '.join(web_platforms)}", 2
                    )

            # Generate LLM analysis with web data
            llm_analysis = await self._generate_llm_analysis(
                cv_text, job_description, skills_search, experience_search, web_results
            )

            return {
                "skills_context": skills_search,
                "experience_context": experience_search,
                "web_context": web_search,
                "web_results": web_results,
                "llm_analysis": llm_analysis,
                "rag_confidence": self._calculate_rag_confidence(skills_search, experience_search, web_search)
            }

        except Exception as e:
            logger.error(f"RAG analysis error: {e}")
            return {
                "skills_context": RAGSearchResult([], [], [], "", 0),
                "experience_context": RAGSearchResult([], [], [], "", 0),
                "web_context": RAGSearchResult([], [], [], "", 0),
                "web_results": [],
                "llm_analysis": LLMResponse("", False, "", 0, 0.0, str(e)),
                "rag_confidence": 0.0
            }

    def _extract_job_skills(self, job_description: str) -> List[str]:
        """Extract key skills from job description"""
        common_skills = [
            "python", "java", "javascript", "react", "angular", "vue", "typescript",
            "docker", "kubernetes", "aws", "azure", "gcp", "sql", "mongodb", "postgresql",
            "machine learning", "data science", "api", "rest", "graphql", "node.js",
            "django", "flask", "express", "spring", "git", "ci/cd", "devops",
            "html", "css", "bootstrap", "tailwind", "pandas", "numpy", "tensorflow",
            "pytorch", "scikit-learn", "redis", "elasticsearch", "microservices"
        ]

        job_lower = job_description.lower()
        found_skills = [skill for skill in common_skills if skill in job_lower]
        return found_skills

    def _extract_job_level(self, job_description: str) -> str:
        """Extract job level from description"""
        job_lower = job_description.lower()

        if "senior" in job_lower or "sr." in job_lower:
            return "senior"
        elif "junior" in job_lower or "jr." in job_lower:
            return "junior"
        elif "lead" in job_lower or "principal" in job_lower:
            return "lead"
        elif "intern" in job_lower:
            return "intern"
        else:
            return "mid-level"

    async def _generate_llm_analysis(self, cv_text: str, job_description: str,
                                   skills_context: RAGSearchResult,
                                   experience_context: RAGSearchResult,
                                   web_results: List[WebScrapingResult]) -> LLMResponse:
        """Generate LLM analysis with RAG context and web data"""
        try:
            # Build context from RAG results
            context_parts = []

            if skills_context.documents:
                context_parts.append("Skills Context:")
                context_parts.extend(skills_context.documents)

            if experience_context.documents:
                context_parts.append("\nExperience Context:")
                context_parts.extend(experience_context.documents)

            # Add web scraping results
            if web_results:
                context_parts.append("\nWeb Profile Data:")
                for result in web_results:
                    if result.success and result.content:
                        platform = result.scraped_data.get('platform', 'website') if result.scraped_data else 'website'
                        context_parts.append(f"{platform.title()}: {result.content[:300]}")

            context = "\n".join(context_parts)

            system_prompt = """You are an expert HR analyst. Analyze the candidate's CV against the job requirements using the provided context and web profile data. Focus on:
1. Skills alignment and gaps
2. Experience level match
3. Web presence and additional qualifications
4. Portfolio and project quality (from web data)
5. Overall fit assessment with specific recommendations

Be thorough and consider both CV content and additional web profile information."""

            prompt = f"""
Job Description:
{job_description[:1000]}

Candidate CV:
{cv_text[:1200]}

Context from Knowledge Base and Web Profiles:
{context}

Provide a comprehensive analysis of this candidate's fit for the position, including insights from their online presence.
"""

            return await self.llm_client.generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=800,
                temperature=0.3
            )

        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return LLMResponse("", False, "", 0, 0.0, str(e))

    def _calculate_rag_confidence(self, skills_search: RAGSearchResult,
                                experience_search: RAGSearchResult,
                                web_search: RAGSearchResult) -> float:
        """Calculate confidence in RAG results including web data"""
        try:
            total_scores = skills_search.scores + experience_search.scores + web_search.scores
            if not total_scores:
                return 0.0

            avg_score = sum(total_scores) / len(total_scores)
            doc_count_factor = min(len(total_scores) / 7.0, 1.0)  # Adjusted for web data

            return avg_score * doc_count_factor
        except:
            return 0.0

class EnhancedRAGCVAgent(BaseAgent):
    """Enhanced CV analysis agent with web scraping and improved RAG integration"""

    def __init__(self, lm_studio_url: str = "http://localhost:1234"):
        super().__init__(agent_name="EnhancedRAGCVAgent", max_concurrent_tasks=3)

        self.lm_studio_url = lm_studio_url
        self.vector_store = SimpleVectorStore()
        self.llm_client = None
        self.rag_processor = None
        self.web_scraper = WebScraper()
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize agent with dependency checking"""
        try:
            # Check dependencies
            DependencyChecker.print_status()

            # Initialize LLM client
            self.llm_client = LMStudioClient(self.lm_studio_url)

            # Test LLM connection
            async with self.llm_client as client:
                pass  # Connection test happens in __aenter__

            # Initialize RAG processor
            self.rag_processor = EnhancedRAGProcessor(self.vector_store, self.llm_client)
            await self.rag_processor.initialize_knowledge_base()

            self.is_initialized = True
            logger.info("EnhancedRAGCVAgent with web scraping initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            self.is_initialized = False
            return False

    async def analyze_cv(self, cv_text: str, job_description: str) -> EnhancedCVAnalysis:
        """Perform enhanced CV analysis with web scraping"""
        start_time = time.time()

        try:
            if not self.is_initialized:
                await self.initialize()

            # Basic analysis
            basic_score = self._calculate_basic_score(cv_text, job_description)
            candidate_name = self._extract_candidate_name(cv_text)
            skills_analysis = self._analyze_skills(cv_text, job_description)
            experience_analysis = self._analyze_experience(cv_text, job_description)

            # Web data collection
            urls = self.web_scraper.extract_urls_from_cv(cv_text)
            web_data = {
                "urls_found": urls,
                "scraped_results": [],
                "additional_skills": [],
                "web_score_boost": 0.0
            }

            # RAG-enhanced analysis with web scraping
            rag_enhanced_score = basic_score
            llm_insights = "Basic analysis only - RAG not available"
            recommendations = ["Review basic qualifications"]

            if self.rag_processor and self.rag_processor.knowledge_initialized:
                async with self.llm_client:
                    rag_results = await self.rag_processor.analyze_with_rag(cv_text, job_description)

                    # Process web scraping results
                    if rag_results["web_results"]:
                        web_data["scraped_results"] = [
                            {
                                "url": result.url,
                                "success": result.success,
                                "platform": result.scraped_data.get('platform', 'unknown') if result.scraped_data else 'unknown',
                                "content_preview": result.content[:200] if result.content else "",
                                "skills_found": result.scraped_data.get('skills_found', []) if result.scraped_data else []
                            }
                            for result in rag_results["web_results"]
                        ]

                        # Extract additional skills from web data
                        for result in rag_results["web_results"]:
                            if result.success and result.scraped_data:
                                additional_skills = result.scraped_data.get('skills_found', [])
                                web_data["additional_skills"].extend(additional_skills)

                        # Calculate web score boost
                        successful_scrapes = sum(1 for result in rag_results["web_results"] if result.success)
                        web_data["web_score_boost"] = min(successful_scrapes * 5.0, 15.0)  # Max 15% boost

                    if rag_results["llm_analysis"].success:
                        llm_insights = rag_results["llm_analysis"].content
                        rag_confidence = rag_results["rag_confidence"]
                        rag_enhanced_score = (basic_score * 0.6) + (rag_confidence * 25) + web_data["web_score_boost"]
                        recommendations = self._generate_recommendations(rag_results, basic_score, web_data)

            # Calculate final score
            final_score = min((basic_score + rag_enhanced_score) / 2, 100.0)

            processing_time = time.time() - start_time

            return EnhancedCVAnalysis(
                candidate_name=candidate_name,
                basic_score=basic_score,
                rag_enhanced_score=rag_enhanced_score,
                final_score=final_score,
                skills_analysis=skills_analysis,
                experience_analysis=experience_analysis,
                web_data=web_data,
                llm_insights=llm_insights,
                recommendations=recommendations,
                confidence=min(final_score / 100.0, 1.0),
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"CV analysis error: {e}")

            return EnhancedCVAnalysis(
                candidate_name="Unknown",
                basic_score=0.0,
                rag_enhanced_score=0.0,
                final_score=0.0,
                skills_analysis={"error": str(e)},
                experience_analysis={"error": str(e)},
                web_data={"error": str(e)},
                llm_insights=f"Analysis failed: {str(e)}",
                recommendations=["Unable to analyze - check system configuration"],
                confidence=0.0,
                processing_time=processing_time
            )

    def _calculate_basic_score(self, cv_text: str, job_description: str) -> float:
        """Calculate basic matching score"""
        try:
            # Keyword matching
            job_words = set(job_description.lower().split())
            cv_words = set(cv_text.lower().split())

            # Remove common words
            stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            job_words -= stop_words
            cv_words -= stop_words

            if not job_words:
                return 0.0

            overlap = len(job_words.intersection(cv_words))
            basic_score = (overlap / len(job_words)) * 100

            return min(basic_score, 100.0)

        except Exception as e:
            logger.error(f"Basic score calculation error: {e}")
            return 0.0

    def _extract_candidate_name(self, cv_text: str) -> str:
        """Extract candidate name from CV"""
        try:
            lines = cv_text.split('\n')[:5]
            for line in lines:
                line = line.strip()
                # Look for name-like patterns
                if 2 <= len(line.split()) <= 4 and len(line) < 50:
                    # Check if it looks like a name (mostly alphabetic)
                    alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / len(line)
                    if alpha_ratio > 0.8:
                        return line
            return "Unknown Candidate"
        except:
            return "Unknown Candidate"

    def _analyze_skills(self, cv_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze skills match"""
        try:
            # Extended technical skills database
            skills_db = [
                "python", "java", "javascript", "typescript", "react", "angular", "vue",
                "docker", "kubernetes", "aws", "azure", "gcp", "sql", "mongodb", "postgresql",
                "machine learning", "ai", "data science", "api", "rest", "graphql", "node.js",
                "django", "flask", "express", "spring", "git", "ci/cd", "devops",
                "html", "css", "bootstrap", "tailwind", "pandas", "numpy", "tensorflow",
                "pytorch", "scikit-learn", "redis", "elasticsearch", "microservices",
                "c++", "c#", "go", "rust", "php", "ruby", "swift", "kotlin"
            ]

            cv_lower = cv_text.lower()
            job_lower = job_description.lower()

            cv_skills = [skill for skill in skills_db if skill in cv_lower]
            job_skills = [skill for skill in skills_db if skill in job_lower]

            matching_skills = list(set(cv_skills).intersection(set(job_skills)))
            missing_skills = list(set(job_skills) - set(cv_skills))

            skills_score = (len(matching_skills) / max(len(job_skills), 1)) * 100

            return {
                "cv_skills": cv_skills,
                "job_skills": job_skills,
                "matching_skills": matching_skills,
                "missing_skills": missing_skills,
                "skills_score": skills_score
            }

        except Exception as e:
            logger.error(f"Skills analysis error: {e}")
            return {"error": str(e), "skills_score": 0.0}

    def _analyze_experience(self, cv_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze experience match"""
        try:
            # Extract years of experience
            experience_patterns = [
                r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)',
                r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)'
            ]

            cv_years = 0.0
            for pattern in experience_patterns:
                matches = re.findall(pattern, cv_text.lower())
                if matches:
                    cv_years = max(float(match) for match in matches)
                    break

            # Extract required experience from job
            job_years = 0.0
            for pattern in experience_patterns:
                matches = re.findall(pattern, job_description.lower())
                if matches:
                    job_years = max(float(match) for match in matches)
                    break

            # Calculate experience match
            if job_years == 0:
                exp_score = 100.0
            elif cv_years >= job_years:
                exp_score = 100.0
            else:
                exp_score = (cv_years / job_years) * 100

            return {
                "cv_experience_years": cv_years,
                "required_experience_years": job_years,
                "experience_score": exp_score,
                "experience_gap": max(0, job_years - cv_years)
            }

        except Exception as e:
            logger.error(f"Experience analysis error: {e}")
            return {"error": str(e), "experience_score": 0.0}

    def _generate_recommendations(self, rag_results: Dict[str, Any], basic_score: float,
                                web_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis including web data"""
        try:
            recommendations = []

            # Score-based recommendations
            final_score = basic_score + web_data.get("web_score_boost", 0)
            if final_score >= 80:
                recommendations.append("HIGHLY RECOMMENDED - Strong candidate with excellent qualifications")
            elif final_score >= 65:
                recommendations.append("RECOMMENDED - Good match with solid potential")
            else:
                recommendations.append("CONSIDER WITH CAUTION - Significant gaps present")

            # Web presence recommendations
            if web_data.get("scraped_results"):
                successful_scrapes = sum(1 for result in web_data["scraped_results"] if result["success"])
                if successful_scrapes > 0:
                    recommendations.append(f"Strong online presence verified ({successful_scrapes} profiles found)")

                    # GitHub specific recommendations
                    github_results = [r for r in web_data["scraped_results"]
                                    if r["platform"] == "github" and r["success"]]
                    if github_results:
                        recommendations.append("Active GitHub profile demonstrates coding skills and project experience")

            # Additional skills from web
            if web_data.get("additional_skills"):
                additional_count = len(set(web_data["additional_skills"]))
                if additional_count > 0:
                    recommendations.append(f"Web profiles reveal {additional_count} additional technical skills")

            # RAG-based recommendations
            if rag_results.get("rag_confidence", 0) > 0.7:
                recommendations.append("AI analysis indicates strong cultural and technical fit")
            elif rag_results.get("rag_confidence", 0) < 0.3:
                recommendations.append("AI analysis suggests reviewing alternative candidates")

            return recommendations

        except Exception as e:
            logger.error(f"Recommendations generation error: {e}")
            return ["Review manually due to analysis limitations"]

    async def execute_task(self, task_data: Dict[str, Any]) -> AgentResult:
        """Execute enhanced CV analysis task"""
        try:
            cv_text = task_data.get("cv_text")
            job_description = task_data.get("job_description")

            if not cv_text or not job_description:
                raise ValueError("cv_text and job_description are required")

            result = await self.analyze_cv(cv_text, job_description)

            return self.create_success_result(
                data=asdict(result),
                confidence_score=result.confidence
            )

        except Exception as e:
            return self.create_error_result(
                error_message=str(e),
                data={"agent": self.agent_name}
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics"""
        base_stats = super().get_performance_stats()

        base_stats.update({
            "lm_studio_url": self.lm_studio_url,
            "is_initialized": self.is_initialized,
            "dependencies": DependencyChecker.check_all(),
            "rag_ready": DependencyChecker.is_ready_for_rag(),
            "web_scraping_enabled": DEPENDENCIES.get('beautifulsoup4', False) and DEPENDENCIES.get('requests', False)
        })

        return base_stats


# Diagnostic functions
async def diagnose_rag_system():
    """Comprehensive RAG system diagnosis"""
    print("=== RAG System Diagnostics ===\n")

    # Check dependencies
    DependencyChecker.print_status()

    # Test vector store
    print("\n=== Vector Store Test ===")
    try:
        vector_store = SimpleVectorStore()
        vector_store.create_collection("test")

        test_docs = ["Python is a programming language", "Machine learning uses data"]
        success = vector_store.add_documents("test", test_docs)

        if success:
            search_result = vector_store.similarity_search("test", "programming", 1)
            print(f"✓ Vector store working - found {len(search_result.documents)} documents")
        else:
            print("✗ Vector store failed to add documents")

    except Exception as e:
        print(f"✗ Vector store error: {e}")

    # Test web scraping
    print("\n=== Web Scraping Test ===")
    try:
        scraper = WebScraper()
        test_cv = "Check out my GitHub: https://github.com/testuser and LinkedIn: https://linkedin.com/in/testuser"
        urls = scraper.extract_urls_from_cv(test_cv)
        print(f"✓ URL extraction working - found {len(urls)} URLs: {urls}")

        if DEPENDENCIES.get('beautifulsoup4') and DEPENDENCIES.get('requests'):
            print("✓ Web scraping dependencies available")
        else:
            print("✗ Web scraping dependencies missing (beautifulsoup4, requests)")

    except Exception as e:
        print(f"✗ Web scraping test error: {e}")

    # Test LM Studio connection
    print("\n=== LM Studio Connection Test ===")
    try:
        client = LMStudioClient()
        async with client:
            if client.is_connected:
                print(f"✓ LM Studio connected - Models: {client.available_models}")

                # Test completion
                response = await client.generate_completion("Say hello")
                if response.success:
                    print(f"✓ LLM generation working: {response.content[:50]}...")
                else:
                    print(f"✗ LLM generation failed: {response.error_message}")
            else:
                print("✗ LM Studio not connected")
                print("  Make sure LM Studio is running on http://localhost:1234")
                print("  Load a model in LM Studio before testing")

    except Exception as e:
        print(f"✗ LM Studio test error: {e}")

    # Test full RAG pipeline
    print("\n=== Full RAG Pipeline Test ===")
    try:
        agent = EnhancedRAGCVAgent()
        initialized = await agent.initialize()

        if initialized:
            print("✓ RAG agent initialized successfully")

            # Test analysis with web scraping
            sample_cv = """John Smith, Python developer with 3 years experience
            GitHub: https://github.com/johnsmith
            LinkedIn: https://linkedin.com/in/johnsmith"""
            sample_job = "Looking for Python developer with Django experience"

            result = await agent.analyze_cv(sample_cv, sample_job)
            print(f"✓ Analysis completed - Score: {result.final_score:.1f}%")

            if result.web_data.get("urls_found"):
                print(f"✓ Web scraping detected {len(result.web_data['urls_found'])} URLs")

        else:
            print("✗ RAG agent initialization failed")

    except Exception as e:
        print(f"✗ RAG pipeline error: {e}")


async def test_enhanced_agent():
    """Test the enhanced RAG agent with web scraping"""
    print("=== Testing Enhanced RAG CV Agent with Web Scraping ===\n")

    # Run diagnostics first
    await diagnose_rag_system()

    print("\n=== Sample Analysis with Web Data ===")

    sample_cv = """
    Jane Doe
    Senior Software Engineer
    Email: jane.doe@email.com
    Phone: +1-555-0123
    GitHub: https://github.com/janedoe
    LinkedIn: https://linkedin.com/in/janedoe
    Portfolio: https://janedoe.dev

    EXPERIENCE:
    Senior Python Developer (2020-2024) - 4 years
    - Built web applications using Django and Flask
    - Implemented machine learning models with TensorFlow
    - Managed AWS cloud infrastructure
    - Led team of 3 junior developers

    SKILLS:
    Python, Django, Flask, TensorFlow, AWS, Docker, PostgreSQL, React

    EDUCATION:
    Master's in Computer Science
    """

    sample_job = """
    Senior Python Developer Position

    We need an experienced Python developer for our growing team.

    REQUIREMENTS:
    - 3+ years Python experience
    - Django or Flask framework
    - Machine learning experience preferred
    - AWS cloud platform
    - Team leadership experience
    - Strong online presence and portfolio

    RESPONSIBILITIES:
    - Develop scalable web applications
    - Implement ML algorithms
    - Mentor junior developers
    """

    try:
        agent = EnhancedRAGCVAgent()
        result = await agent.analyze_cv(sample_cv, sample_job)

        print(f"Candidate: {result.candidate_name}")
        print(f"Basic Score: {result.basic_score:.1f}%")
        print(f"RAG Enhanced Score: {result.rag_enhanced_score:.1f}%")
        print(f"Final Score: {result.final_score:.1f}%")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Processing Time: {result.processing_time:.2f}s")

        print(f"\nSkills Analysis:")
        skills = result.skills_analysis
        if "matching_skills" in skills:
            print(f"  Matching: {skills['matching_skills']}")
            print(f"  Missing: {skills['missing_skills']}")

        print(f"\nWeb Data Analysis:")
        web_data = result.web_data
        if "urls_found" in web_data:
            print(f"  URLs Found: {web_data['urls_found']}")
            print(f"  Successful Scrapes: {sum(1 for r in web_data.get('scraped_results', []) if r['success'])}")
            print(f"  Web Score Boost: +{web_data.get('web_score_boost', 0):.1f}%")

            if web_data.get("additional_skills"):
                print(f"  Additional Skills from Web: {web_data['additional_skills']}")

        print(f"\nLLM Insights:")
        print(f"  {result.llm_insights}")

        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")

    except Exception as e:
        print(f"Test error: {e}")


if __name__ == "__main__":
    # Install missing dependencies reminder
    missing = DependencyChecker.get_missing()
    if missing:
        print("Missing dependencies detected. Install with:")
        print(f"pip install {' '.join(missing)}")
        print()

    asyncio.run(test_enhanced_agent())