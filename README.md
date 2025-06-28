# JobSeeker
An AI built for the purpose of hunting jobs.

You're asking for the full blueprint, which is a fantastic next step\! This will be a multi-module Python project, designed for robustness and cloud deployment. I'll provide a high-level project layout, illustrative code snippets for key functionalities, and discuss the services involved.

**Important Note:** The code snippets provided are illustrative and simplified. Real-world implementations will require more robust error handling, configuration, logging, and specific selectors for job boards. Always check a website's `robots.txt` and terms of service before scraping.

-----

## Project Layout (Modular Structure)

```
ai-job-hunter/
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile                  # For containerization
├── config/
│   ├── __init__.py
│   └── settings.py             # Centralized configuration (API keys, DB settings, etc.)
├── data/
│   ├── resumes/                # Store your original CVs (e.g., my_cv_v1.pdf, my_cv_v2.docx)
│   └── processed_jobs/         # Store raw job data (e.g., job_data_linkedin_20250628.json)
├── src/
│   ├── __init__.py
│   ├── job_scraper/
│   │   ├── __init__.py
│   │   ├── linkedin_scraper.py # Specific scraper for LinkedIn
│   │   ├── indeed_scraper.py   # Specific scraper for Indeed
│   │   └── base_scraper.py     # Base class/utility for scrapers
│   ├── resume_parser/
│   │   ├── __init__.py
│   │   └── parser.py           # Logic to parse PDF/DOCX into structured data
│   ├── matching_engine/
│   │   ├── __init__.py
│   │   ├── matcher.py          # Core logic for job-resume matching
│   │   └── llm_integrator.py   # Handles LLM API calls and prompt engineering
│   ├── cv_tailor/
│   │   ├── __init__.py
│   │   └── tailor.py           # Generates tailored CV sections/cover letters
│   ├── database/
│   │   ├── __init__.py
│   │   └── manager.py          # Handles database interactions (save jobs, CVs, etc.)
│   ├── notifications/
│   │   ├── __init__.py
│   │   └── notifier.py         # Sends email/Slack/Discord notifications
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── main_scheduler.py   # Orchestrates running tasks periodically
│   └── app/                    # If you build a simple API or UI for interaction
│       ├── __init__.py
│       └── main.py             # e.g., FastAPI/Flask app endpoint
├── tests/
│   ├── __init__.py
│   ├── test_scraper.py
│   ├── test_parser.py
│   └── test_matcher.py
└── main.py                     # Entry point for local execution/testing (or a specific cloud entry)
```

-----

## Code Examples (Illustrative Snippets)

### 1\. `requirements.txt`

```
requests
beautifulsoup4
selenium  # Or playwright if preferred for dynamic sites
lxml
spacy
python-docx
PyMuPDF # For PDF parsing
APScheduler
openai    # Or google-generativeai, anthropic for LLM APIs
python-dotenv # For local environment variables
psycopg2-binary # If using PostgreSQL
```

### 2\. `config/settings.py`

```python
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# General settings
APP_NAME = "AI Job Hunter"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./jobs.db") # Default to SQLite

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEED_API_KEY = os.getenv("INDEED_API_KEY") # If an API is available, otherwise relying on scraping
LINKEDIN_USERNAME = os.getenv("LINKEDIN_USERNAME") # For Selenium-based login
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")

# Notification settings
NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Scraper settings
SCRAPE_INTERVAL_HOURS = int(os.getenv("SCRAPE_INTERVAL_HOURS", "24"))
MAX_JOB_DESCRIPTIONS = int(os.getenv("MAX_JOB_DESCRIPTIONS", "50")) # Per run

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
```

### 3\. `src/job_scraper/base_scraper.py` (Abstract Base Class)

```python
from abc import ABC, abstractmethod
import logging
import time
import random

logger = logging.getLogger(__name__)

class JobScraper(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    @abstractmethod
    def scrape_jobs(self, keywords: str, location: str, pages: int = 1) -> list[dict]:
        """
        Abstract method to be implemented by specific job board scrapers.
        Returns a list of dictionaries, each representing a job posting.
        """
        pass

    def _apply_rate_limit(self):
        """Applies a random delay to avoid overwhelming servers."""
        delay = random.uniform(2, 5) # Delay between 2 and 5 seconds
        logger.info(f"Applying rate limit delay of {delay:.2f} seconds...")
        time.sleep(delay)

    def _get_page_content(self, url: str) -> str | None:
        """Helper to fetch page content, handles basic errors."""
        try:
            self._apply_rate_limit()
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
```

### 4\. `src/job_scraper/indeed_scraper.py` (Simplified Example)

```python
import requests
from bs4 import BeautifulSoup
from .base_scraper import JobScraper
import logging

logger = logging.getLogger(__name__)

class IndeedScraper(JobScraper):
    def __init__(self):
        super().__init__("https://uk.indeed.com") # Adjust for your region

    def scrape_jobs(self, keywords: str, location: str, pages: int = 1) -> list[dict]:
        job_listings = []
        search_url_template = f"{self.base_url}/jobs?q={keywords}&l={location}&start="

        for page_num in range(pages):
            start_index = page_num * 10 # Indeed typically shows 10 jobs per page
            url = f"{search_url_template}{start_index}"
            logger.info(f"Scraping Indeed page: {url}")
            html_content = self._get_page_content(url)
            if not html_content:
                continue

            soup = BeautifulSoup(html_content, 'lxml') # Use 'lxml' for faster parsing
            job_cards = soup.select(".jobsearch-ResultsList > li") # Specific selector for job cards

            if not job_cards:
                logger.warning(f"No job cards found on {url}. End of results or selector changed.")
                break

            for card in job_cards:
                try:
                    title_element = card.select_one(".jobTitle")
                    job_title = title_element.get_text(strip=True) if title_element else "N/A"

                    company_element = card.select_one(".companyName")
                    company_name = company_element.get_text(strip=True) if company_element else "N/A"

                    location_element = card.select_one(".companyLocation")
                    job_location = location_element.get_text(strip=True) if location_element else "N/A"

                    link_element = card.select_one("a.jcs-JobTitle")
                    job_url = self.base_url + link_element["href"] if link_element and "href" in link_element.attrs else "N/A"

                    # You'd need to go to the job_url to get the full description,
                    # or try to extract from the card's data attributes if possible.
                    # For simplicity, we'll just get main details here.
                    # A more advanced scraper would then visit each job_url for description.
                    description = "Description requires visiting job page." # Placeholder

                    job_listings.append({
                        "title": job_title,
                        "company": company_name,
                        "location": job_location,
                        "url": job_url,
                        "description": description,
                        "source": "Indeed"
                    })
                except Exception as e:
                    logger.error(f"Error parsing job card: {e}")
                    continue
        return job_listings

# Example of how to use Selenium/Playwright if needed (not integrated above)
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By

# class LinkedInScraper(JobScraper):
#     def __init__(self):
#         super().__init__("https://www.linkedin.com")
#         self.driver = self._init_webdriver()

#     def _init_webdriver(self):
#         options = webdriver.ChromeOptions()
#         options.add_argument("--headless") # Run in background
#         options.add_argument("--no-sandbox")
#         options.add_argument("--disable-dev-shm-usage")
#         service = ChromeService(ChromeDriverManager().install())
#         return webdriver.Chrome(service=service, options=options)

#     def scrape_jobs(self, keywords: str, location: str, pages: int = 1) -> list[dict]:
#         # ... implement LinkedIn-specific logic using self.driver.get(), find_element(By.CSS_SELECTOR), etc.
#         # Remember to close the driver: self.driver.quit()
#         pass
```

### 5\. `src/resume_parser/parser.py`

```python
import spacy
from docx import Document
import fitz # PyMuPDF
import re
import logging

logger = logging.getLogger(__name__)

# Load a pre-trained spaCy model
try:
    nlp = spacy.load("en_core_web_lg") # 'lg' for better NER, download with: python -m spacy download en_core_web_lg
except OSError:
    logger.error("SpaCy model 'en_core_web_lg' not found. Please run 'python -m spacy download en_core_web_lg'")
    nlp = spacy.load("en_core_web_sm") # Fallback to smaller model

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path: str) -> str:
    text = []
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text.append(para.text)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {docx_path}: {e}")
    return "\n".join(text)

def parse_resume(file_path: str) -> dict:
    """
    Parses a resume file (PDF or DOCX) and extracts structured information.
    """
    text = ""
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        logger.error(f"Unsupported file format: {file_path}")
        return {}

    doc = nlp(text)

    # Basic entity extraction using spaCy NER and custom logic
    parsed_data = {
        "raw_text": text,
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "summary": None,
        "sections": {} # Store text by detected section
    }

    # Extract Name (often the first PERSON entity)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not parsed_data["name"]:
            parsed_data["name"] = ent.text
            break

    # Extract Email and Phone using regex
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        parsed_data["email"] = email_match.group(0)

    phone_match = re.search(r"(\+\d{1,3}\s?)?(\(?\d{2,}\)?[\s.-]?\d{3,}[\s.-]?\d{4,})", text)
    if phone_match:
        parsed_data["phone"] = phone_match.group(0)

    # --- More advanced NLP for skills, experience, education would go here ---
    # This often involves custom NER models, rule-based matching, or more sophisticated text chunking.
    # For example, to get skills, you might have a predefined list of tech skills
    # and check if they appear in the text.

    # Placeholder for skill extraction:
    tech_skills = ["Python", "JavaScript", "SQL", "AWS", "Azure", "Linux", "Cybersecurity", "Networking", "Penetration Testing", "SIEM"]
    parsed_data["skills"] = [skill for skill in tech_skills if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE)]

    # Placeholder for section extraction:
    # This is a very basic attempt and would need significant refinement.
    sections_pattern = re.compile(r"(Education|Experience|Skills|Summary|Certifications|Projects)\s*[\n:]", re.IGNORECASE)
    matches = list(sections_pattern.finditer(text))
    
    current_section = "summary"
    section_start_idx = 0
    
    for i, match in enumerate(matches):
        section_name = match.group(1).lower()
        section_text = text[section_start_idx:match.start()].strip()
        if section_text:
            parsed_data["sections"][current_section] = section_text
        current_section = section_name
        section_start_idx = match.end()

    # Get the last section
    remaining_text = text[section_start_idx:].strip()
    if remaining_text:
        parsed_data["sections"][current_section] = remaining_text
    
    # Example: If 'Experience' section is found, you'd then parse it further
    if "experience" in parsed_data["sections"]:
        # This would be a more complex parsing step, potentially using regex for dates and bullet points
        # For demo, just taking a slice
        parsed_data["experience"].append({"raw_text": parsed_data["sections"]["experience"][:200] + "..."}) # truncated example

    logger.info(f"Successfully parsed resume from {file_path}. Extracted name: {parsed_data.get('name')}")
    return parsed_data

```

### 6\. `src/matching_engine/llm_integrator.py`

```python
import openai # Using OpenAI as an example, adjust for Google Gemini or Anthropic
from openai import OpenAI
import logging
from config.settings import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

class LLMIntegrator:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE

    def _call_llm(self, messages: list[dict], max_tokens: int = 1000) -> str:
        """Helper to call the LLM API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except openai.APIErr as e:
            logger.error(f"LLM API Error: {e}")
            return "An error occurred while generating the response."
        except Exception as e:
            logger.error(f"Unexpected error calling LLM: {e}")
            return "An unexpected error occurred."

    def generate_tailored_summary(self, job_description: str, parsed_resume: dict) -> str:
        """Generates a tailored professional summary for the CV."""
        prompt = f"""
        You are an expert resume writer. Given the following job description and a candidate's resume,
        create a concise (2-4 sentences) professional summary that highlights the candidate's most relevant
        skills and experiences for *this specific job*.

        Job Description:
        {job_description}

        Candidate's Core Skills: {', '.join(parsed_resume.get('skills', []))}
        Candidate's Experience Snapshot: {parsed_resume.get('sections', {}).get('experience', 'No experience details found.')[:500]}...

        Professional Summary:
        """
        messages = [
            {"role": "system", "content": "You are a helpful and expert resume writer."},
            {"role": "user", "content": prompt}
        ]
        return self._call_llm(messages, max_tokens=200)

    def suggest_bullet_point_improvements(self, job_description: str, resume_experience_bullet: str) -> str:
        """Suggests improvements for a single resume bullet point."""
        prompt = f"""
        Given the job description below and one of my resume's experience bullet points,
        suggest how to rephrase or enhance this bullet point to make it more relevant and impactful
        for the job. Focus on action verbs, quantifiable achievements, and matching keywords.
        Provide only the improved bullet point, or 'No change needed' if perfect.

        Job Description:
        {job_description}

        My Resume Bullet Point:
        {resume_experience_bullet}

        Improved Bullet Point:
        """
        messages = [
            {"role": "system", "content": "You are a helpful and expert resume optimizer."},
            {"role": "user", "content": prompt}
        ]
        return self._call_llm(messages, max_tokens=100)

    def generate_cover_letter(self, job_description: str, parsed_resume: dict, company_name: str, job_title: str) -> str:
        """Generates a full cover letter."""
        prompt = f"""
        Write a professional cover letter for the position of '{job_title}' at '{company_name}'.
        Highlight the candidate's most relevant skills and experiences from their resume,
        directly addressing the key requirements in the job description.
        The cover letter should be concise, professional, and express strong interest.
        Start with a formal salutation (e.g., "Dear Hiring Manager,").

        Job Description:
        {job_description}

        Candidate's Resume Data:
        Name: {parsed_resume.get('name', 'Candidate Name')}
        Email: {parsed_resume.get('email', 'N/A')}
        Phone: {parsed_resume.get('phone', 'N/A')}
        Skills: {', '.join(parsed_resume.get('skills', []))}
        Summary: {parsed_resume.get('sections', {}).get('summary', 'No summary.')}
        Experience (first 500 chars): {parsed_resume.get('sections', {}).get('experience', 'No experience details found.')[:500]}...
        Education (first 200 chars): {parsed_resume.get('sections', {}).get('education', 'No education details found.')[:200]}...

        Cover Letter:
        """
        messages = [
            {"role": "system", "content": "You are an expert at writing compelling cover letters."},
            {"role": "user", "content": prompt}
        ]
        return self._call_llm(messages, max_tokens=800)

```

### 7\. `src/matching_engine/matcher.py` (Basic Matching Logic)

```python
import logging
from .llm_integrator import LLMIntegrator

logger = logging.getLogger(__name__)

class JobMatcher:
    def __init__(self):
        self.llm_integrator = LLMIntegrator()

    def calculate_match_score(self, job_description: str, parsed_resume: dict) -> float:
        """
        Calculates a simple match score based on keyword presence and LLM evaluation.
        This is a highly simplified example. Real matching is more complex.
        """
        job_description_lower = job_description.lower()
        resume_text_lower = parsed_resume.get("raw_text", "").lower()
        skills = [s.lower() for s in parsed_resume.get("skills", [])]

        score = 0
        total_keywords = 0

        # Simple keyword matching from job description
        keywords_to_check = ["cybersecurity", "security analyst", "incident response",
                             "compliance", "linux", "cloud security", "threat intelligence", "firewall",
                             "penetration testing", "risk management", "azure", "aws"] # Expand this list significantly!

        for keyword in keywords_to_check:
            total_keywords += 1
            if keyword in job_description_lower and (keyword in resume_text_lower or keyword in skills):
                score += 1

        # Basic LLM relevance check (can be expanded to a true/false match)
        llm_match_prompt = f"""
        Given the following job description and a candidate's resume content,
        rate the overall fit of the candidate for the job on a scale of 0 to 1.
        A 0 indicates no fit, and a 1 indicates a perfect fit.
        Focus on skills, experience, and educational alignment.
        Provide ONLY the numerical score (e.g., 0.75).

        Job Description:
        {job_description}

        Candidate Resume Summary: {parsed_resume.get('sections', {}).get('summary', '')}
        Candidate Skills: {', '.join(parsed_resume.get('skills', []))}
        Candidate Experience Highlights: {parsed_resume.get('sections', {}).get('experience', '')[:300]}...
        """
        llm_response = self.llm_integrator._call_llm(
            [{"role": "system", "content": "You are a helpful job matching assistant."},
             {"role": "user", "content": llm_match_prompt}],
            max_tokens=10 # Expect a short numerical answer
        )

        try:
            llm_numeric_score = float(llm_response)
            score += (llm_numeric_score * total_keywords) # Weigh LLM score
        except ValueError:
            logger.warning(f"Could not parse LLM response to float: {llm_response}")
            llm_numeric_score = 0.0

        if total_keywords == 0:
            return llm_numeric_score # Avoid division by zero
        
        # Combine keyword score with LLM score
        combined_score = (score / total_keywords + llm_numeric_score) / 2
        return min(1.0, max(0.0, combined_score)) # Ensure score is between 0 and 1

8. src/cv_tailor/tailor.py
Python

import logging
from src.matching_engine.llm_integrator import LLMIntegrator
import os

logger = logging.getLogger(__name__)

class CVTailor:
    def __init__(self):
        self.llm_integrator = LLMIntegrator()

    def generate_tailored_cv_content(self, job_description: str, parsed_resume: dict) -> dict:
        """
        Generates tailored content for various CV sections based on the job description.
        """
        tailored_content = {}

        # 1. Tailor Professional Summary
        logger.info("Generating tailored summary...")
        tailored_content["summary"] = self.llm_integrator.generate_tailored_summary(
            job_description, parsed_resume
        )

        # 2. Suggest experience bullet point improvements (iterative)
        logger.info("Suggesting experience bullet point improvements...")
        improved_experience_bullets = []
        # This part assumes your parsed_resume.experience contains a list of strings or dicts
        # For this example, let's just take the raw experience text and ask LLM to rephrase.
        raw_experience = parsed_resume.get('sections', {}).get('experience', '').split('\n')
        for bullet in raw_experience:
            if bullet.strip(): # Skip empty lines
                improved_bullet = self.llm_integrator.suggest_bullet_point_improvements(
                    job_description, bullet
                )
                improved_experience_bullets.append(improved_bullet)
        tailored_content["experience_bullets_suggestions"] = improved_experience_bullets


        # 3. Generate tailored cover letter
        # You'll need job title and company name from the scraped job data
        # For simplicity, let's assume they are extracted in the main loop
        job_title_placeholder = "Cyber Security Analyst" # Replace with actual from job_data
        company_name_placeholder = "Acme Corp" # Replace with actual from job_data

        logger.info("Generating tailored cover letter...")
        tailored_content["cover_letter"] = self.llm_integrator.generate_cover_letter(
            job_description, parsed_resume, company_name_placeholder, job_title_placeholder
        )

        return tailored_content

    def save_tailored_content(self, tailored_data: dict, job_id: str, output_dir: str = "data/tailored_output"):
        """Saves the tailored content to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        if tailored_data.get("summary"):
            with open(os.path.join(output_dir, f"{job_id}_summary.txt"), "w") as f:
                f.write(tailored_data["summary"])
            logger.info(f"Saved tailored summary for {job_id}")

        # Save improved bullet points
        if tailored_data.get("experience_bullets_suggestions"):
            with open(os.path.join(output_dir, f"{job_id}_experience_suggestions.txt"), "w") as f:
                for bullet in tailored_data["experience_bullets_suggestions"]:
                    f.write(bullet + "\n")
            logger.info(f"Saved experience suggestions for {job_id}")

        # Save cover letter
        if tailored_data.get("cover_letter"):
            with open(os.path.join(output_dir, f"{job_id}_cover_letter.txt"), "w") as f:
                f.write(tailored_data["cover_letter"])
            logger.info(f"Saved tailored cover letter for {job_id}")

        # You might also want to generate a new DOCX or PDF here,
        # which is more complex and usually involves using templates and libraries
        # like `python-docx` or `reportlab`. For LLM-generated content,
        # saving as text files is simpler for review.
9. src/database/manager.py (Simplified SQLite Example)
Python

import sqlite3
import json
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                title TEXT,
                company TEXT,
                location TEXT,
                url TEXT UNIQUE,
                description TEXT,
                source TEXT,
                posted_date TEXT,
                scraped_date TEXT,
                match_score REAL,
                tailored_summary TEXT,
                tailored_cover_letter TEXT,
                status TEXT DEFAULT 'new' -- 'new', 'reviewed', 'applied'
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database tables ensured.")

    def save_job(self, job_data: dict):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO jobs (id, title, company, location, url, description, source, posted_date, scraped_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                str(hash(job_data['url'] + job_data['title'])), # Simple ID from hash of URL+Title
                job_data.get('title'),
                job_data.get('company'),
                job_data.get('location'),
                job_data.get('url'),
                job_data.get('description'),
                job_data.get('source'),
                job_data.get('posted_date', 'N/A')
            ))
            conn.commit()
            logger.info(f"Saved job: {job_data.get('title')} at {job_data.get('company')}")
        except sqlite3.IntegrityError:
            logger.warning(f"Job already exists (URL: {job_data.get('url')}). Skipping.")
        except Exception as e:
            logger.error(f"Error saving job: {e}")
        finally:
            conn.close()

    def get_unprocessed_jobs(self) -> list[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE match_score IS NULL") # Or status = 'new'
        jobs = []
        for row in cursor.fetchall():
            # Convert row to dictionary for easier access
            job = {
                "id": row[0], "title": row[1], "company": row[2], "location": row[3],
                "url": row[4], "description": row[5], "source": row[6],
                "posted_date": row[7], "scraped_date": row[8], "match_score": row[9],
                "tailored_summary": row[10], "tailored_cover_letter": row[11], "status": row[12]
            }
            jobs.append(job)
        conn.close()
        return jobs

    def update_job_match_and_tailoring(self, job_id: str, score: float, tailored_summary: str, tailored_cover_letter: str):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE jobs
            SET match_score = ?, tailored_summary = ?, tailored_cover_letter = ?, status = 'processed'
            WHERE id = ?
        """, (score, tailored_summary, tailored_cover_letter, job_id))
        conn.commit()
        conn.close()
        logger.info(f"Updated job {job_id} with match score and tailored content.")

    def get_top_matched_jobs(self, limit: int = 5) -> list[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE match_score IS NOT NULL ORDER BY match_score DESC LIMIT ?", (limit,))
        jobs = []
        for row in cursor.fetchall():
            job = {
                "id": row[0], "title": row[1], "company": row[2], "location": row[3],
                "url": row[4], "description": row[5], "source": row[6],
                "posted_date": row[7], "scraped_date": row[8], "match_score": row[9],
                "tailored_summary": row[10], "tailored_cover_letter": row[11], "status": row[12]
            }
            jobs.append(job)
        conn.close()
        return jobs

10. src/notifications/notifier.py (Simplified Email Example)
Python

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from config.settings import NOTIFICATION_EMAIL, EMAIL_PASSWORD

logger = logging.getLogger(__name__)

class Notifier:
    def __init__(self):
        self.sender_email = NOTIFICATION_EMAIL
        self.sender_password = EMAIL_PASSWORD
        self.smtp_server = "smtp.gmail.com" # Or your email provider's SMTP server
        self.smtp_port = 587

    def send_email(self, recipient_email: str, subject: str, body: str):
        if not self.sender_email or not self.sender_password:
            logger.warning("Email notification not configured (sender_email/password missing). Skipping email.")
            return

        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls() # Secure the connection
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            logger.info(f"Email sent successfully to {recipient_email} with subject: '{subject}'")
        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {e}")

    def notify_new_jobs(self, jobs: list[dict], recipient_email: str):
        if not jobs:
            return

        subject = f"{len(jobs)} New Highly Matched Job(s) Found!"
        body = "Here are the details for the highly matched jobs:\n\n"
        for job in jobs:
            body += f"Title: {job['title']}\n"
            body += f"Company: {job['company']}\n"
            body += f"Location: {job['location']}\n"
            body += f"URL: {job['url']}\n"
            body += f"Match Score: {job['match_score']:.2f}\n"
            body += f"Tailored Summary (Snippet): {job['tailored_summary'][:150]}...\n"
            body += f"Tailored Cover Letter (Snippet): {job['tailored_cover_letter'][:200]}...\n"
            body += "-" * 30 + "\n"
        
        body += "\nCheck your tailored_output folder for full documents."
        self.send_email(recipient_email, subject, body)

11. src/scheduler/main_scheduler.py (Orchestration Logic)
Python

import logging
import os
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler # Use BlockingScheduler for a single, always-on process
# from apscheduler.schedulers.background import BackgroundScheduler # Use for web apps or other concurrent processes

from config.settings import SCRAPE_INTERVAL_HOURS, DATABASE_URL, NOTIFICATION_EMAIL
from src.job_scraper.indeed_scraper import IndeedScraper # Import your specific scrapers
from src.resume_parser.parser import parse_resume
from src.matching_engine.matcher import JobMatcher
from src.cv_tailor.tailor import CVTailor
from src.database.manager import DatabaseManager
from src.notifications.notifier import Notifier

logger = logging.getLogger(__name__)

# --- Global Instances (initialized once) ---
db_manager = DatabaseManager(DATABASE_URL)
indeed_scraper = IndeedScraper() # Add other scrapers as you implement them
job_matcher = JobMatcher()
cv_tailor = CVTailor()
notifier = Notifier()

# Define your resume path (this would ideally be configurable or uploaded via a UI)
YOUR_RESUME_PATH = "data/resumes/my_cyber_cv.pdf" # Make sure you have this file!
YOUR_TARGET_EMAIL = NOTIFICATION_EMAIL # Where notifications will be sent

def scrape_and_process_jobs_task():
    logger.info(f"[{datetime.now()}] Starting scheduled job scraping and processing...")

    # 1. Parse your resume once at the start of the process
    logger.info(f"Parsing resume: {YOUR_RESUME_PATH}")
    parsed_resume = parse_resume(YOUR_RESUME_PATH)
    if not parsed_resume:
        logger.error("Failed to parse resume. Cannot proceed with job matching.")
        return

    # 2. Scrape Jobs (add more scrapers as needed)
    keywords = "cybersecurity analyst" # Example keywords
    location = "Edinburgh" # Example location, adjust for your target
    
    scraped_jobs = []
    scraped_jobs.extend(indeed_scraper.scrape_jobs(keywords, location, pages=2)) # Scrape 2 pages from Indeed
    # scraped_jobs.extend(linkedin_scraper.scrape_jobs(keywords, location, pages=1)) # If you implement LinkedIn

    logger.info(f"Found {len(scraped_jobs)} jobs across all sources.")

    # 3. Save new jobs to DB and get unprocessed ones
    for job in scraped_jobs:
        db_manager.save_job(job) # Save all scraped jobs, manager handles duplicates

    unprocessed_jobs = db_manager.get_unprocessed_jobs()
    logger.info(f"Processing {len(unprocessed_jobs)} new/unprocessed jobs.")

    highly_matched_jobs = []

    for job in unprocessed_jobs:
        # Fetch full job description if only partial was scraped
        # (For Indeed example, 'description' was a placeholder. Here you'd visit 'job['url']')
        full_job_description = job['description'] # Assume this now has the full description or fetch it.
        # if job['source'] == 'Indeed' and 'Description requires visiting job page.' in job['description']:
        #     # Need to implement logic here to visit job['url'] and scrape full description
        #     # This can be tricky and requires careful handling (e.g., using Selenium)
        #     pass


        # 4. Match Job to Resume
        match_score = job_matcher.calculate_match_score(full_job_description, parsed_resume)
        job['match_score'] = match_score # Update in memory for logic below

        if match_score >= 0.7: # Define your threshold for "highly matched"
            logger.info(f"Job '{job['title']}' has high match score: {match_score:.2f}. Tailoring CV...")
            
            # 5. Tailor CV/Cover Letter Content
            tailored_content = cv_tailor.generate_tailored_cv_content(full_job_description, parsed_resume)
            
            # 6. Save tailored content to DB and files
            db_manager.update_job_match_and_tailoring(
                job['id'],
                match_score,
                tailored_content.get('summary', ''),
                tailored_content.get('cover_letter', '')
            )
            cv_tailor.save_tailored_content(tailored_content, job['id'])
            
            job['tailored_summary'] = tailored_content.get('summary', '') # For notification
            job['tailored_cover_letter'] = tailored_content.get('cover_letter', '') # For notification
            highly_matched_jobs.append(job)
        else:
            logger.info(f"Job '{job['title']}' match score: {match_score:.2f}. Not high enough.")
            # Update score even if not highly matched, to avoid reprocessing
            db_manager.update_job_match_and_tailoring(job['id'], match_score, "", "") # Store empty tailoring

    # 7. Notify User
    if highly_matched_jobs:
        logger.info(f"Notifying user about {len(highly_matched_jobs)} highly matched jobs.")
        notifier.notify_new_jobs(highly_matched_jobs, YOUR_TARGET_EMAIL)
    else:
        logger.info("No highly matched jobs found in this run.")

    logger.info(f"[{datetime.now()}] Job processing complete.")


def start_scheduler():
    scheduler = BlockingScheduler()
    # Schedule the job to run every X hours
    scheduler.add_job(scrape_and_process_jobs_task, 'interval', hours=SCRAPE_INTERVAL_HOURS)
    logger.info(f"Scheduler started, will run every {SCRAPE_INTERVAL_HOURS} hours.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down.")
        scheduler.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    scrape_and_process_jobs_task() # Run once immediately on startup for testing/initial run
    start_scheduler()
Services Involved & Cloud Deployment
Local Development Environment
Python 3.9+: The language runtime.

Virtual Environment (venv): Isolate project dependencies.

Git: For version control.

IDE (VS Code, PyCharm): For coding and debugging.

.env file: For local management of environment variables/secrets.

SQLite: For local database testing.

Cloud Deployment (Google Cloud Platform Example)
Given your goal of "leaving it in the cloud running," a serverless or containerized approach is ideal for cost-effectiveness and scalability.

Containerization (Docker):

Purpose: Package your application and all its dependencies into a single, portable unit. This ensures that your code runs consistently across different environments (local, staging, production).

Dockerfile:

Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (can also be done in a separate step or pre-built into image)
RUN python -m spacy download en_core_web_lg

# Expose a port if you were to run a web service (e.g., app/main.py)
# EXPOSE 8080

# Command to run the application
# If running scheduler as a persistent process:
CMD ["python", "src/scheduler/main_scheduler.py"]
# If running as a web service for API endpoint:
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "src.app.main:app"]
Cloud Platform: Google Cloud Platform (GCP)

Compute:

Google Cloud Run: Recommended for this specific use case. It's a serverless platform for containerized applications. You deploy your Docker image, and it handles scaling (to zero, meaning you only pay when code is running) and infrastructure.

For the scheduler: You would run src/scheduler/main_scheduler.py as a continuously running Cloud Run service (configure min instances to 1, or use Cloud Scheduler to trigger it periodically if the jobs are short-lived functions).

For an API: If you later add a web interface (src/app/main.py), Cloud Run is perfect for that too.

Google Compute Engine (VMs): More control, but requires managing servers (updates, scaling). Not ideal for cost efficiency for an intermittent background task like this.

Database:

Google Cloud SQL (PostgreSQL/MySQL): For a production-grade relational database. More robust and scalable than SQLite.

Google Firestore (NoSQL): If your data model is more flexible (e.g., storing varied job posting formats).

Storage:

Google Cloud Storage (GCS): For storing raw CVs, scraped job data, and generated tailored CV documents. Very cost-effective for large files.

Secrets Management:

Google Secret Manager: Securely store API keys (OpenAI, LinkedIn, email passwords etc.) and database credentials. Your application retrieves them at runtime.

Scheduling:

Google Cloud Scheduler: If you choose to run your processing as a Cloud Function or as an ephemeral Cloud Run job, you'd use Cloud Scheduler to trigger it at a set interval (e.g., daily).

Alternative (as per code): The BlockingScheduler within a single Cloud Run instance means the scheduler runs inside your container. This works but means your container is always "on" (and incurring minimal cost, even if at idle minimum instances). For more cost-efficiency, you could break the job down into smaller, event-driven Cloud Functions triggered by Cloud Scheduler.

Logging and Monitoring:

Google Cloud Logging / Cloud Monitoring: Automatically collects logs and metrics from your deployed services. Essential for debugging and monitoring performance.

Deployment Automation (GitHub Actions):

Purpose: Automate the process of building your Docker image and deploying it to Cloud Run whenever you push changes to your GitHub repository.

.github/workflows/deploy.yml (Example):

YAML

name: Deploy to Cloud Run

on:
  push:
    branches:
      - main # Trigger on pushes to the main branch

env:
  PROJECT_ID: your-gcp-project-id # Replace with your GCP Project ID
  SERVICE_NAME: ai-job-hunter-scheduler # Name of your Cloud Run service
  REGION: europe-west2 # GCP region (e.g., europe-west2)

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: 'read'
      id-token: 'write'

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Authenticate Google Cloud CLI
      uses: google-github-actions/auth@v2
      with:
        workload_identity_provider: projects/YOUR_PROJECT_NUMBER/locations/global/workloadIdentityPools/YOUR_POOL_ID/providers/YOUR_PROVIDER_ID # Replace these
        service_account: your-service-account@your-gcp-project-id.iam.gserviceaccount.com # Replace with your service account email

    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v2
      with:
        project_id: ${{ env.PROJECT_ID }}

    - name: Build and Push Docker image to Artifact Registry
      run: |
        gcloud auth configure-docker ${REGION}-docker.pkg.dev
        docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${SERVICE_NAME}/${SERVICE_NAME}:latest .
        docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${SERVICE_NAME}/${SERVICE_NAME}:latest

    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy ${{ env.SERVICE_NAME }} \
          --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${SERVICE_NAME}/${SERVICE_NAME}:latest \
          --region ${{ env.REGION }} \
          --platform managed \
          --allow-unauthenticated \ # Consider removing if your API is not public facing
          --set-env-vars=DATABASE_URL=${{ secrets.DATABASE_URL }},OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }},NOTIFICATION_EMAIL=${{ secrets.NOTIFICATION_EMAIL }},EMAIL_PASSWORD=${{ secrets.EMAIL_PASSWORD }},LLM_MODEL=${{ secrets.LLM_MODEL }} \ # Pass secrets
          --cpu=1 --memory=2Gi \ # Adjust resources based on your needs
          --min-instances=1 --max-instances=1 # For a scheduler, you typically want 1 min instance
Workload Identity Federation: This is the secure way to authenticate GitHub Actions with GCP. It requires setting up a Workload Identity Pool and Provider in GCP IAM.

GitHub Secrets: Store your DATABASE_URL, OPENAI_API_KEY, etc., as repository secrets in GitHub. Never commit these directly to your code!

This complete setup provides a robust, scalable, and automated solution for your AI job hunter. It's a significant undertaking but, works well.



