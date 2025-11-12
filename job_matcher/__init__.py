"""
Job Matcher Agent - Match Structured Resume to Job Description
Uses parsed CV JSON to perform intelligent job matching with web scraping
"""
import json
import logging
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Import web scraper if available
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from web_scraper import WebScraper
    WEB_SCRAPING_AVAILABLE = True
except Exception as e:
    WEB_SCRAPING_AVAILABLE = False
    WebScraper = None
    logger.warning(f"Web scraping not available: {e}")


class JobMatcherAgent:
    """
    Matches structured resume JSON against job requirements
    Provides detailed scoring and recommendations
    """
    
    def __init__(self, llm_client=None, enable_web_scraping=True, web_scraping_timeout=10):
        """
        Initialize Job Matcher
        
        Args:
            llm_client: Optional LLMClient for enhanced analysis
            enable_web_scraping: Whether to scrape candidate's online profiles
            web_scraping_timeout: Timeout for web scraping requests in seconds
        """
        self.llm_client = llm_client
        self.enable_web_scraping = enable_web_scraping and WEB_SCRAPING_AVAILABLE
        self.web_scraping_timeout = web_scraping_timeout
        self.web_scraper = WebScraper(timeout=web_scraping_timeout) if self.enable_web_scraping else None
        
        if self.enable_web_scraping:
            logger.info(f"✅ Job Matcher Agent initialized with web scraping enabled (timeout: {web_scraping_timeout}s)")
        else:
            logger.info("✅ Job Matcher Agent initialized (web scraping disabled)")

    
    async def match_cv_to_job(
        self,
        resume_data: Dict[str, Any],
        job_title: str,
        job_description: str,
        job_requirements: str,
        github_url: str = "",
        linkedin_url: str = ""
    ) -> Dict[str, Any]:
        """
        Complete job matching pipeline
        
        Args:
            resume_data: Structured resume JSON from CVParserAgent
            job_title: Job position title
            job_description: Job description text
            job_requirements: Required skills and qualifications
        
        Returns:
            Detailed matching analysis with score and recommendations
        """
        logger.info(f"🎯 Starting job matching for: {job_title}")
        
        # Extract candidate info
        personal_info = resume_data.get("personalInformation", {})
        candidate_name = personal_info.get("fullName", "Candidate")
        
        # Perform matching analysis
        skills_analysis = await self._analyze_skills_match(resume_data, job_requirements, job_description)
        experience_analysis = self._analyze_experience_match(resume_data, job_description)
        education_analysis = self._analyze_education_match(resume_data, job_requirements)
        
        # Perform web scraping if enabled
        web_scraping_results = None
        web_score_boost = 0.0
        if self.enable_web_scraping:
            web_scraping_results = self._scrape_candidate_profiles(resume_data, github_url, linkedin_url)
            web_score_boost = self._calculate_web_score_boost(web_scraping_results)

        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            skills_analysis,
            experience_analysis,
            education_analysis,
            web_score_boost
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            skills_analysis,
            experience_analysis,
            education_analysis,
            overall_score,
            web_scraping_results
        )
        
        # Determine decision
        decision = self._make_hiring_decision(overall_score)
        
        # Build comprehensive result
        result = {
            "candidate_name": candidate_name,
            "job_title": job_title,
            "overall_score": overall_score,
            "decision": decision,
            "skills_analysis": skills_analysis,
            "experience_analysis": experience_analysis,
            "education_analysis": education_analysis,
            "web_scraping_results": web_scraping_results,
            "web_score_boost": web_score_boost,
            "recommendations": recommendations,
            "matched_at": datetime.utcnow().isoformat()
        }
        
        # Add LLM-enhanced analysis if available
        if self.llm_client:
            try:
                llm_analysis = await self._get_llm_analysis(
                    resume_data, job_title, job_description, job_requirements,
                    skills_analysis, experience_analysis, education_analysis, overall_score, web_scraping_results
                )
                result["llm_insights"] = llm_analysis
            except Exception as e:
                logger.warning(f"LLM analysis unavailable: {e}")
                result["llm_insights"] = "LLM analysis unavailable - using rule-based matching"
        
        logger.info(f"✅ Matching complete - Score: {overall_score:.1f}%")
        return result
    
    async def _analyze_skills_match(
        self,
        resume_data: Dict[str, Any],
        job_requirements: str,
        job_description: str
    ) -> Dict[str, Any]:
        """Analyze skills matching between CV and job using LLM"""
        
        # Get candidate skills
        candidate_skills = resume_data.get("skills", [])
        
        logger.info(f"🔍 Analyzing skills match for {len(candidate_skills)} candidate skills")
        
        # If LLM is available, use it for intelligent skill matching
        if self.llm_client:
            logger.info("✅ LLM client available - using intelligent skill matching")
            try:
                result = await self._llm_skill_analysis(candidate_skills, job_requirements, job_description)
                logger.info(f"✅ LLM skill analysis complete - {len(result.get('matched_skills', []))} skills matched")
                return result
            except Exception as e:
                logger.error(f"❌ LLM skill analysis failed: {e}, falling back to basic matching")
        else:
            logger.warning("⚠️ LLM client not available - using basic skill matching")
        
        # Fallback to simple matching if LLM unavailable
        return self._basic_skill_match(candidate_skills, job_requirements, job_description)
    
    async def _llm_skill_analysis(
        self,
        candidate_skills: List[str],
        job_requirements: str,
        job_description: str
    ) -> Dict[str, Any]:
        """Use LLM to intelligently match skills"""
        
        logger.info(f"🤖 Starting LLM skill analysis for {len(candidate_skills)} skills")
        
        # Create a mapping for easy lookup
        candidate_skills_str = '\n'.join([f"  - {skill}" for skill in candidate_skills])
        
        prompt = f"""You are an expert technical recruiter analyzing skill matches between a candidate's resume and job requirements.

CANDIDATE SKILLS (EXACT NAMES - DO NOT MODIFY THESE):
{candidate_skills_str}

JOB REQUIREMENTS:
{job_requirements}

JOB DESCRIPTION:
{job_description}

MATCHING RULES:
1. Match flexibly: "JavaScript" = "Javascript" = "JS", "Node.js" = "Node" = "NodeJS", "CI/CD" = "CICD" = "Ci/Cd"
2. Match synonyms: "React.js" = "React" = "ReactJS", "TypeScript" = "Typescript" = "TS"
3. Match cloud: "AWS" = "Aws" = "Amazon Web Services", "Azure" = "Microsoft Azure", "GCP" = "Gcp" = "Google Cloud"
4. When a skill matches, use the EXACT TEXT from the CANDIDATE SKILLS list above
5. If you see "Typescript" in candidate skills and job needs "TypeScript", mark "Typescript" as matched
6. If you see "Aws" in candidate skills and job needs "AWS", mark "Aws" as matched

OUTPUT FORMAT (JSON only, no explanation):
{{
    "required_skills": ["all technical skills mentioned in job, use your own names"],
    "matched_skills": ["EXACT candidate skill names that match - copy from candidate list above"],
    "missing_skills": ["required skills NOT found in candidate list"],
    "skills_score": <0-100 number>,
    "analysis": "one sentence summary"
}}

CRITICAL: For matched_skills, you MUST copy the exact text from the candidate skills list. Do not change case, spelling, or format!"""

        try:
            logger.info("📤 Sending prompt to LLM...")
            response = await self.llm_client.generate(prompt, max_tokens=800, temperature=0.3)
            logger.info(f"📥 Received LLM response: {response[:200]}...")
            
            # Parse LLM response
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Validate and normalize the response
                required_skills = result.get("required_skills", [])
                matched_skills = result.get("matched_skills", [])
                missing_skills = result.get("missing_skills", [])
                skills_score = result.get("skills_score", 0)
                analysis = result.get("analysis", "")
                
                logger.info(f"✅ Parsed LLM response - Required: {len(required_skills)}, Matched: {len(matched_skills)}, Missing: {len(missing_skills)}")
                
                return {
                    "candidate_skills": candidate_skills,
                    "required_skills": required_skills,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "skills_score": min(float(skills_score), 100.0),
                    "skills_count": len(candidate_skills),
                    "match_percentage": len(matched_skills) / max(len(required_skills), 1) * 100 if required_skills else 0,
                    "llm_analysis": analysis
                }
            else:
                logger.error(f"❌ Could not parse JSON from LLM response: {response[:500]}")
                raise ValueError("Could not parse JSON from LLM response")
                
        except Exception as e:
            logger.error(f"❌ LLM skill analysis error: {e}")
            # Fallback to basic matching
            return self._basic_skill_match(candidate_skills, job_requirements, job_description)
    
    def _basic_skill_match(
        self,
        candidate_skills: List[str],
        job_requirements: str,
        job_description: str
    ) -> Dict[str, Any]:
        """Simple fallback skill matching without LLM"""
        
        logger.info("⚠️ Using basic skill matching (LLM unavailable)")
        
        combined_text = f"{job_requirements}\n{job_description}".lower()
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        
        # Simple matching: check if candidate skill appears in job text
        matched_skills = []
        for i, skill_lower in enumerate(candidate_skills_lower):
            if skill_lower in combined_text or re.search(r'\b' + re.escape(skill_lower) + r'\b', combined_text):
                matched_skills.append(candidate_skills[i])
        
        logger.info(f"📊 Basic matching found {len(matched_skills)}/{len(candidate_skills)} skills")
        
        # Basic score calculation
        skills_score = (len(matched_skills) / max(len(candidate_skills), 1)) * 100 if candidate_skills else 50.0
        
        return {
            "candidate_skills": candidate_skills,
            "required_skills": [],
            "matched_skills": matched_skills,
            "missing_skills": [],
            "skills_score": min(skills_score, 100.0),
            "skills_count": len(candidate_skills),
            "match_percentage": skills_score
        }
    
    def _analyze_experience_match(
        self,
        resume_data: Dict[str, Any],
        job_description: str
    ) -> Dict[str, Any]:
        """Analyze work experience matching"""
        
        work_experience = resume_data.get("workExperience", [])
        
        # Calculate total years of experience
        total_years = self._calculate_total_experience(work_experience)
        
        # Extract required years from job description
        required_years = self._extract_required_experience(job_description)
        
        # Calculate experience score
        if required_years > 0:
            if total_years >= required_years:
                experience_score = 100.0
            else:
                experience_score = (total_years / required_years) * 100
        else:
            experience_score = 85.0  # Default if no experience requirement
        
        # Analyze role relevance
        relevant_roles = self._find_relevant_roles(work_experience, job_description)
        
        return {
            "total_years": total_years,
            "required_years": required_years,
            "experience_score": min(experience_score, 100.0),
            "experience_gap": max(0, required_years - total_years),
            "work_entries": len(work_experience),
            "relevant_roles": relevant_roles
        }
    
    def _analyze_education_match(
        self,
        resume_data: Dict[str, Any],
        job_requirements: str
    ) -> Dict[str, Any]:
        """Analyze education matching"""
        
        education = resume_data.get("education", [])
        
        # Check for degree requirements
        required_degree = self._extract_required_degree(job_requirements)
        
        # Find candidate's highest degree
        candidate_degree = self._get_highest_degree(education)
        
        # Calculate education score
        education_score = self._score_education(candidate_degree, required_degree)
        
        return {
            "candidate_degree": candidate_degree,
            "required_degree": required_degree,
            "education_score": education_score,
            "education_entries": len(education),
            "meets_requirement": education_score >= 70.0
        }
    
    def _calculate_total_experience(self, work_experience: List[Dict]) -> float:
        """Calculate total years of experience from work history"""
        total_months = 0
        
        for job in work_experience:
            start_date = job.get("startDate", "")
            end_date = job.get("endDate", "Present")
            
            # Simple year extraction
            start_year = self._extract_year(start_date)
            end_year = self._extract_year(end_date)
            
            if start_year and end_year:
                years = end_year - start_year
                total_months += years * 12
        
        return total_months / 12.0
    
    def _extract_year(self, date_str: str) -> int:
        """Extract year from date string"""
        if not date_str or date_str.lower() == "present":
            return datetime.now().year
        
        # Try to find 4-digit year
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        if year_match:
            return int(year_match.group(0))
        
        return 0
    
    def _extract_required_experience(self, job_description: str) -> float:
        """Extract required years of experience from job description"""
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)',
            r'(\d+)\+?\s*(?:years?|yrs?)',
            r'minimum\s+(\d+)\s*(?:years?|yrs?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, job_description.lower())
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _extract_required_degree(self, job_requirements: str) -> str:
        """Extract required degree level"""
        text_lower = job_requirements.lower()
        
        if "phd" in text_lower or "doctorate" in text_lower:
            return "PhD"
        elif "master" in text_lower or "msc" in text_lower or "mba" in text_lower:
            return "Master"
        elif "bachelor" in text_lower or "bsc" in text_lower or "ba" in text_lower:
            return "Bachelor"
        else:
            return "None"
    
    def _get_highest_degree(self, education: List[Dict]) -> str:
        """Get candidate's highest degree"""
        degree_levels = {"phd": 4, "doctorate": 4, "master": 3, "msc": 3, "mba": 3, 
                        "bachelor": 2, "bsc": 2, "ba": 2, "associate": 1}
        
        highest = 0
        highest_degree = "None"
        
        for edu in education:
            degree = edu.get("degree", "").lower()
            for key, level in degree_levels.items():
                if key in degree and level > highest:
                    highest = level
                    highest_degree = key.upper()
        
        return highest_degree
    
    def _score_education(self, candidate_degree: str, required_degree: str) -> float:
        """Score education match"""
        degree_map = {"PHD": 4, "DOCTORATE": 4, "MASTER": 3, "MSC": 3, "MBA": 3,
                     "BACHELOR": 2, "BSC": 2, "BA": 2, "ASSOCIATE": 1, "NONE": 0}
        
        candidate_level = degree_map.get(candidate_degree.upper(), 0)
        required_level = degree_map.get(required_degree.upper(), 0)
        
        if required_level == 0:
            return 85.0  # No specific requirement
        elif candidate_level >= required_level:
            return 100.0
        else:
            return (candidate_level / required_level) * 100
    
    def _find_relevant_roles(self, work_experience: List[Dict], job_description: str) -> List[str]:
        """Find work experiences relevant to the job"""
        relevant = []
        job_lower = job_description.lower()
        
        for job in work_experience:
            title = job.get("jobTitle", "").lower()
            company = job.get("company", "").lower()
            description = " ".join(job.get("description", [])).lower()
            
            # Check if job title or description matches
            if any(keyword in title or keyword in description 
                   for keyword in ["developer", "engineer", "manager", "analyst", "designer"]):
                relevant.append(job.get("jobTitle", "Unknown Role"))
        
        return relevant[:5]  # Return top 5
    
    def _calculate_overall_score(
        self,
        skills_analysis: Dict,
        experience_analysis: Dict,
        education_analysis: Dict,
        web_score_boost: float = 0.0
    ) -> float:
        """Calculate weighted overall matching score"""
        
        # Weights
        skills_weight = 0.5      # 50% - Most important
        experience_weight = 0.3  # 30%
        education_weight = 0.2   # 20%
        
        base_score = (
            skills_analysis["skills_score"] * skills_weight +
            experience_analysis["experience_score"] * experience_weight +
            education_analysis["education_score"] * education_weight
        )
        
        # Add web scraping boost (up to +10%)
        overall = base_score + web_score_boost
        
        return min(overall, 100.0)
    
    def _scrape_candidate_profiles(self, resume_data: Dict[str, Any], github_url: str = "", linkedin_url: str = "") -> Dict[str, Any]:
        """Scrape candidate's online profiles from links in resume"""
        if not self.web_scraper:
            return None
        
        logger.info("🌐 Starting web scraping of candidate profiles...")
        
        personal_info = resume_data.get("personalInformation", {})
        links = personal_info.get("links", [])
        
        # Add direct URLs if provided
        if github_url and github_url.strip():
            if github_url not in links:
                links.insert(0, github_url.strip())  # Add to beginning for priority
        
        if linkedin_url and linkedin_url.strip():
            if linkedin_url not in links:
                links.insert(0, linkedin_url.strip())  # Add to beginning for priority
        
        scraping_results = {
            "total_links": len(links),
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "profiles": [],
            "additional_skills": [],
            "projects_found": 0,
            "repositories_count": 0,
            "direct_github_provided": bool(github_url and github_url.strip()),
            "direct_linkedin_provided": bool(linkedin_url and linkedin_url.strip())
        }
        
        if not links:
            logger.info("No links found in resume to scrape")
            return scraping_results
        
        for link in links[:5]:  # Limit to first 5 links
            try:
                # Determine platform
                if "github.com" in link.lower():
                    result = self.web_scraper.scrape_github(link)
                    if result.success:
                        scraping_results["successful_scrapes"] += 1
                        scraping_results["profiles"].append({
                            "platform": "GitHub",
                            "url": link,
                            "data": result.scraped_data
                        })
                        
                        # Extract additional info
                        if result.scraped_data:
                            scraping_results["repositories_count"] = result.scraped_data.get("public_repos", 0)
                            # Extract skills from repos
                            repos = result.scraped_data.get("top_repos", [])
                            for repo in repos:
                                lang = repo.get("language", "")
                                if lang and lang not in scraping_results["additional_skills"]:
                                    scraping_results["additional_skills"].append(lang)
                    else:
                        scraping_results["failed_scrapes"] += 1
                        
                elif "linkedin.com" in link.lower():
                    result = self.web_scraper.scrape_linkedin(link)
                    if result.success:
                        scraping_results["successful_scrapes"] += 1
                        scraping_results["profiles"].append({
                            "platform": "LinkedIn",
                            "url": link,
                            "data": result.scraped_data,
                            "note": "Limited access due to LinkedIn restrictions"
                        })
                    else:
                        scraping_results["failed_scrapes"] += 1
                        
                else:
                    # Try generic scraping
                    logger.debug(f"Skipping unsupported platform: {link}")
                    
            except Exception as e:
                logger.error(f"Error scraping {link}: {e}")
                scraping_results["failed_scrapes"] += 1
        
        logger.info(f"✅ Web scraping complete: {scraping_results['successful_scrapes']} successful, {scraping_results['failed_scrapes']} failed")
        return scraping_results
    
    def _calculate_web_score_boost(self, web_results: Dict[str, Any]) -> float:
        """Calculate score boost based on web scraping results"""
        if not web_results:
            return 0.0
        
        boost = 0.0
        
        # Boost for having active profiles
        if web_results["successful_scrapes"] > 0:
            boost += 2.0  # +2% for having online presence
        
        # Boost for GitHub activity
        repos_count = web_results.get("repositories_count", 0)
        if repos_count > 0:
            # +1% per 5 repos, max +5%
            boost += min(repos_count / 5.0, 5.0)
        
        # Boost for additional skills found
        additional_skills = len(web_results.get("additional_skills", []))
        if additional_skills > 0:
            boost += min(additional_skills * 0.5, 3.0)  # +0.5% per skill, max +3%
        
        return min(boost, 10.0)  # Cap at +10%
    
    def _generate_recommendations(
        self,
        skills_analysis: Dict,
        experience_analysis: Dict,
        education_analysis: Dict,
        overall_score: float,
        web_results: Dict = None
    ) -> List[str]:
        """Generate hiring recommendations"""
        recommendations = []
        
        # Skills recommendations
        if skills_analysis["skills_score"] >= 80:
            recommendations.append("✅ Strong skills match - Candidate has most required skills")
        elif skills_analysis["skills_score"] >= 60:
            recommendations.append("⚠️ Moderate skills match - Some training may be needed")
        else:
            recommendations.append("❌ Skills gap identified - Significant training required")
        
        # Experience recommendations
        if experience_analysis["experience_score"] >= 80:
            recommendations.append("✅ Experience level meets requirements")
        elif experience_analysis["experience_gap"] > 0:
            recommendations.append(f"⚠️ Experience gap: {experience_analysis['experience_gap']:.1f} years below requirement")
        
        # Education recommendations
        if education_analysis["meets_requirement"]:
            recommendations.append("✅ Education requirements met")
        else:
            recommendations.append("⚠️ Education level below requirement")
        
        # Web presence recommendations
        if web_results:
            if web_results["successful_scrapes"] > 0:
                recommendations.append(f"🌐 Active online presence verified ({web_results['successful_scrapes']} profiles)")
                
                if web_results.get("direct_github_provided"):
                    recommendations.append("🔗 Direct GitHub URL provided for enhanced analysis")
                
                if web_results.get("direct_linkedin_provided"):
                    recommendations.append("🔗 Direct LinkedIn URL provided for enhanced analysis")
                
                if web_results.get("repositories_count", 0) > 0:
                    recommendations.append(f"💻 GitHub activity: {web_results['repositories_count']} public repositories")
                
                if web_results.get("additional_skills"):
                    recommendations.append(f"🔍 Additional skills found via web scraping: {', '.join(web_results['additional_skills'][:5])}")
            else:
                recommendations.append("ℹ️ Limited online presence found")
        
        # Overall recommendation
        if overall_score >= 80:
            recommendations.append("🎯 STRONG CANDIDATE - Highly recommended for interview")
        elif overall_score >= 60:
            recommendations.append("👍 GOOD CANDIDATE - Recommended for interview")
        elif overall_score >= 40:
            recommendations.append("🤔 MODERATE FIT - Consider for interview if other candidates unavailable")
        else:
            recommendations.append("❌ WEAK MATCH - Not recommended at this time")
        
        return recommendations
    
    def _make_hiring_decision(self, overall_score: float) -> str:
        """Make hiring decision based on score"""
        if overall_score >= 75:
            return "STRONG MATCH - Proceed to Interview"
        elif overall_score >= 60:
            return "GOOD MATCH - Consider for Interview"
        elif overall_score >= 40:
            return "MODERATE MATCH - Review Carefully"
        else:
            return "WEAK MATCH - Not Recommended"
    
    async def _get_llm_analysis(
        self,
        resume_data: Dict,
        job_title: str,
        job_description: str,
        job_requirements: str,
        skills_analysis: Dict = None,
        experience_analysis: Dict = None,
        education_analysis: Dict = None,
        overall_score: float = 0.0,
        web_results: Dict = None
    ) -> str:
        """Get LLM-enhanced analysis with full scoring and web context"""
        
        if not self.llm_client:
            return "LLM analysis not available"
        
        # Create enhanced prompt for LLM with detailed analysis context
        # Get detailed candidate information
        personal_info = resume_data.get('personalInformation', {})
        candidate_name = personal_info.get('fullName', 'N/A')
        candidate_email = personal_info.get('email', 'N/A')
        candidate_summary = personal_info.get('summary', 'N/A')

        # Get work experience details
        work_exp = resume_data.get('workExperience', [])
        experience_summary = []
        for job in work_exp[:3]:  # Top 3 roles
            title = job.get('jobTitle', 'Unknown')
            company = job.get('company', 'Unknown')
            duration = f"{job.get('startDate', '')} - {job.get('endDate', 'Present')}"
            experience_summary.append(f"{title} at {company} ({duration})")

        # Get education details
        education = resume_data.get('education', [])
        education_summary = []
        for edu in education[:2]:  # Top 2 degrees
            degree = edu.get('degree', 'Unknown')
            institution = edu.get('institution', 'Unknown')
            education_summary.append(f"{degree} from {institution}")

        # Get skills analysis details
        skills_score = skills_analysis.get('skills_score', 0) if skills_analysis else 0
        matched_skills = skills_analysis.get('matched_skills', []) if skills_analysis else []
        missing_skills = skills_analysis.get('missing_skills', []) if skills_analysis else []
        candidate_skills = skills_analysis.get('candidate_skills', []) if skills_analysis else []

        # Get experience analysis details
        exp_score = experience_analysis.get('experience_score', 0) if experience_analysis else 0
        total_years = experience_analysis.get('total_years', 0) if experience_analysis else 0
        required_years = experience_analysis.get('required_years', 0) if experience_analysis else 0
        relevant_roles = experience_analysis.get('relevant_roles', []) if experience_analysis else []

        # Get education analysis details
        edu_score = education_analysis.get('education_score', 0) if education_analysis else 0
        candidate_degree = education_analysis.get('candidate_degree', 'None') if education_analysis else 'None'
        required_degree = education_analysis.get('required_degree', 'None') if education_analysis else 'None'

        # Get web scraping context if available
        web_context = ""
        if web_results and web_results.get('successful_scrapes', 0) > 0:
            repos = web_results.get('repositories_count', 0)
            additional_skills = web_results.get('additional_skills', [])
            web_context = f"""
ONLINE PRESENCE:
- GitHub Repositories: {repos}
- Additional Skills Found: {', '.join(additional_skills[:5]) if additional_skills else 'None'}
- Profiles Verified: {web_results['successful_scrapes']}"""

        # Create ultra-comprehensive prompt for job matching
        prompt = f"""You are a senior technical recruiter with 15+ years of experience in software engineering hiring. Analyze this candidate's fit for the position using a structured, data-driven approach.

JOB POSITION ANALYSIS:
Title: {job_title}
Description: {job_description[:1000]}
Requirements: {job_requirements[:800]}

CANDIDATE PROFILE:
Name: {candidate_name}
Professional Summary: {candidate_summary[:400]}

TECHNICAL SKILLS ({len(candidate_skills)}):
{', '.join(candidate_skills[:25])}

WORK EXPERIENCE ({len(work_exp)} positions):
{chr(10).join(f"• {exp}" for exp in experience_summary)}

EDUCATION:
{chr(10).join(f"• {edu}" for edu in education_summary)}

MATCHING SCORES & ANALYSIS:
• Overall Fit: {overall_score:.1f}%
• Technical Skills: {skills_score:.1f}% ({len(matched_skills)}/{len(matched_skills)+len(missing_skills)} skills matched)
• Experience Level: {exp_score:.1f}% ({total_years:.1f} years vs {required_years:.1f} required)
• Education Match: {edu_score:.1f}% ({candidate_degree} vs {required_degree} required)

MATCHED SKILLS: {', '.join(matched_skills[:10])}
MISSING SKILLS: {', '.join(missing_skills[:10]) if missing_skills else 'None identified'}
RELEVANT ROLES: {', '.join(relevant_roles[:5]) if relevant_roles else 'None identified'}{web_context}

EXECUTIVE HIRING ANALYSIS FRAMEWORK:

1. TECHNICAL COMPETENCY ASSESSMENT:
   - Evaluate skill match quality and criticality
   - Assess experience depth and relevance
   - Identify skill gaps and training requirements

2. CULTURAL & TEAM FIT ANALYSIS:
   - Based on experience progression and role stability
   - Assess growth trajectory and adaptability
   - Evaluate collaboration indicators from background

3. RISK ASSESSMENT:
   - Flight risk based on career progression
   - Onboarding complexity and time-to-productivity
   - Potential red flags in employment history

4. INTERVIEW STRATEGY & QUESTIONS:
   - Technical deep-dive questions for key skills
   - Behavioral questions for soft skills assessment
   - Reference check questions to validate claims

5. COMPENSATION & NEGOTIATION CONSIDERATIONS:
   - Market rate assessment based on skills/experience
   - Negotiation leverage points
   - Retention risk factors

6. FINAL RECOMMENDATION:
   - Hire/No-hire decision with confidence level
   - Specific conditions for offer (if applicable)
   - Alternative role suggestions if not a fit

Provide a comprehensive hiring memorandum with specific, actionable insights. Be direct about concerns and enthusiastic about strengths. Include exact interview questions and reference check points."""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=1200, temperature=0.7)
            return response
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return f"LLM analysis unavailable: {str(e)}"
