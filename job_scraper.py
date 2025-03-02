import os
import time
import json
import re
import random
import asyncio
import math
import concurrent.futures
import pandas as pd
import nltk
import torch
import logging
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pyresparser import ResumeParser
from sentence_transformers import SentenceTransformer, util

# Playwright
from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure Generative AI
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY_HERE")

# =============================================================================
# Configuration & Global Settings
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading BAAI/bge-large-en on device: {device}")
model = SentenceTransformer("BAAI/bge-large-en").to(device)

STOPWORDS = set(stopwords.words('english'))

# -- Adjust concurrency & pacing --
MAX_SCRAPER_WORKERS = 4   # For job search queries
MAX_DESC_WORKERS = 8      # For job descriptions
REQUEST_DELAY = 1.5       # Base delay (seconds)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; rv:110.0) Gecko/20100101 Firefox/110.0"
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

# =============================================================================
# 1) Batch Missing Keywords + Suggestions + Tips
# =============================================================================
def batch_analyze_missing_keywords(cv_text, job_descriptions, batch_size=2, cooldown=5):
    """
    Compare the given CV to multiple job descriptions in fewer AI calls.

    Updated prompt to also return:
      - missingKeywords (3-20 items)
      - resumeSuggestions (string)
      - applicationTips (string)
    """
    all_results = [None] * len(job_descriptions)
    num_batches = math.ceil(len(job_descriptions) / batch_size)

    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, len(job_descriptions))
        batch = job_descriptions[start_i:end_i]

        # Build a prompt for the entire batch
        job_blocks = []
        for i, desc in enumerate(batch):
            global_idx = start_i + i
            job_blocks.append(f"JobIndex {global_idx}:\n{desc}")
        job_descriptions_text = "\n\n".join(job_blocks)

        # Updated prompt:
        prompt = f"""
You are an expert career advisor. Analyze this CV against the following job descriptions. For each job, provide:
1. Missing skills/keywords (3-20 items)
2. Resume/cover letter improvements
3. Application strategy tips

CV (Same for all jobs):
{cv_text}

Job Descriptions in Batch:
{job_descriptions_text}

Return a JSON array with entries containing:
- jobIndex (number)
- missingKeywords (array)
- resumeSuggestions (string)
- applicationTips (string)

Example:
[
  {{
    "jobIndex": 0,
    "missingKeywords": ["AWS", "Agile"],
    "resumeSuggestions": "Add AWS certification section...",
    "applicationTips": "Highlight any Agile experience first..."
  }}
]
"""
        model_obj = genai.GenerativeModel("gemini-2.0-flash")
        response = model_obj.generate_content(prompt)

        if response:
            raw_response_text = response.text.strip()
            #logger.info(f"Raw AI Response for Batch {batch_idx+1}/{num_batches}:\n{raw_response_text}")

            # Clean response by removing markdown-style code blocks (```json ... ```)
            cleaned_response = re.sub(r"```[jJ][sS][oO][nN]?\n(.*?)\n```", r"\1", raw_response_text, flags=re.DOTALL).strip()

            try:
                data = json.loads(cleaned_response)

                for item in data:
                    idx_in_all = item.get("jobIndex")
                    if isinstance(idx_in_all, int) and 0 <= idx_in_all < len(job_descriptions):
                        all_results[idx_in_all] = {
                            "missingKeywords": item.get("missingKeywords", []),
                            "resumeSuggestions": item.get("resumeSuggestions", ""),
                            "applicationTips": item.get("applicationTips", "")
                        }
                logger.info(f"Processed batch {batch_idx+1}/{num_batches} with {len(batch)} jobs")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for batch {batch_idx+1}: {e}")
                logger.error(f"Cleaned API Response:\n{cleaned_response}")

        else:
            logger.warning(f"Empty response from AI for batch {batch_idx+1}")

        time.sleep(cooldown)

    return all_results

# =============================================================================
# 2) Other Helper Functions
# =============================================================================
def get_embedding(text: str):
    """Encode text using SentenceTransformer."""
    return model.encode(text, convert_to_tensor=True, device=device)

def tokenize_and_filter_nouns(text, keep_adjectives=False):
    tokens = re.split(r'\W+', text.lower())
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    tagged = nltk.pos_tag(tokens)

    valid_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
    if keep_adjectives:
        valid_tags.update({'JJ', 'JJR', 'JJS'})
    filtered = [token for (token, tag) in tagged if tag in valid_tags]
    return set(filtered)

def hybrid_similarity(resume_text, job_text, embedding_weight=0.7):
    """
    Combine SentenceTransformer embedding similarity with TF-IDF overlap.
    """
    # 1) Embedding-based
    resume_emb = get_embedding(resume_text)
    job_emb = get_embedding(job_text)
    emb_score = float(util.cos_sim(resume_emb, job_emb)[0][0])

    # 2) TF-IDF overlap
    texts = [resume_text, job_text]
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    resume_vec = tfidf_matrix.toarray()[0]
    job_vec = tfidf_matrix.toarray()[1]
    dot = float((resume_vec * job_vec).sum())
    norm_product = (resume_vec**2).sum()**0.5 * (job_vec**2).sum()**0.5
    tfidf_sim = 0.0 if norm_product == 0 else dot / norm_product

    # Weighted combination
    return embedding_weight * emb_score + (1 - embedding_weight) * tfidf_sim

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF via PyPDF2."""
    logger.info(f"Extracting text from PDF: {pdf_path}")
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                logger.warning(f"Page {i} had no text.")
        return ' '.join(pages_text)

def process_resume(pdf_path):
    """Parse resume for skills + full text."""
    logger.info(f"Processing resume: {pdf_path}")
    try:
        data = ResumeParser(pdf_path).get_extracted_data()
        logger.info(f"ResumeParser extracted skills: {data.get('skills', [])}")
        resume_text = extract_text_from_pdf(pdf_path)
        return {
            'skills': data.get('skills', []),
            'text': resume_text
        }
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise

def build_url(role, filters, page):
    """
    Build a Reed URL with optional filters and page number.
    """
    base_url = f"https://www.reed.co.uk/jobs/{role}-jobs"
    params = []
    if 'salaryFrom' in filters and filters['salaryFrom']:
        params.append(f"salaryFrom={filters['salaryFrom']}")
    if 'salaryTo' in filters and filters['salaryTo']:
        params.append(f"salaryTo={filters['salaryTo']}")
    if 'date' in filters and filters['date']:
        params.append(f"dateCreatedOffSet={filters['date']}")

    if page > 1:
        params.append(f"pageno={page}")
    if params:
        return f"{base_url}?{'&'.join(params)}"
    else:
        return base_url

# =============================================================================
# 3) PLAYWRIGHT ASYNC SCRAPING (Parallel)
# =============================================================================
async def process_job_card(card, query):
    """Extract data from a single job card element."""
    try:
        title_el = await card.query_selector('h2.job-card_jobResultHeading__title__IQ8iT a')
        company_el = await card.query_selector('a.gtmJobListingPostedBy')
        meta_els = await card.query_selector_all('ul.job-card_jobMetadata__gjkG3 li')

        title = await title_el.inner_text() if title_el else None
        link = await title_el.get_attribute('href') if title_el else None
        company = await company_el.inner_text() if company_el else None

        salary = await meta_els[0].inner_text() if len(meta_els) > 0 else None
        location = await meta_els[1].inner_text() if len(meta_els) > 1 else None

        if link and link.startswith("/"):
            link = f"https://www.reed.co.uk{link}"
        return {
            'SearchTerm': query,
            'Title': title.strip() if title else None,
            'Company': company.strip() if company else None,
            'Salary': salary.strip() if salary else None,
            'Location': location.strip() if location else None,
            'Link': link
        }
    except Exception as e:
        logger.error(f"Error processing job card: {str(e)}", exc_info=True)
        return None

async def scrape_single_query_playwright(context, query, filters):
    """Scrapes all pages for a single query."""
    jobs = []
    page = await context.new_page()
    page_number = 1

    try:
        while True:
            url = build_url(query, filters, page_number)
            logger.info(f"[{query}] Scraping page {page_number} - {url}")
            
            try:
                await page.goto(url, timeout=20000, wait_until="domcontentloaded")
                await asyncio.sleep(REQUEST_DELAY * (0.6 + random.random() * 0.4))
                await page.wait_for_selector('article[data-qa="job-card"]', timeout=10000)
            except Exception as e:
                if page_number == 1:
                    logger.warning(f"[{query}] Initial page failed: {str(e)}")
                break

            job_cards = await page.query_selector_all('article[data-qa="job-card"]')
            if not job_cards:
                logger.info(f"[{query}] No more jobs on page {page_number}")
                break

            card_tasks = [process_job_card(card, query) for card in job_cards]
            card_results = await asyncio.gather(*card_tasks)
            new_jobs = [res for res in card_results if res is not None]
            jobs += new_jobs

            # Progress logging for every 50 new jobs
            if len(jobs) % 50 == 0:
                logger.info(f"PROGRESS: Identified {len(jobs)} jobs so far...")

            logger.info(f"[{query}] Page {page_number} yielded {len(new_jobs)} jobs")
            page_number += 1

    finally:
        await page.close()
        logger.info(f"[{query}] Finished scraping, total jobs: {len(jobs)}")

    return jobs

async def scrape_reed_jobs_today_playwright(cleaned_queries, filters=None):
    """Parallel job search for multiple queries."""
    filters = filters or {}
    all_jobs = []

    async with async_playwright() as p:
        logger.info("Launching browser for job search")
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]
        )

        context_pool = [
            await browser.new_context(
                user_agent=get_random_user_agent(),
                viewport={"width": 1920, "height": 1080}
            )
            for _ in range(MAX_SCRAPER_WORKERS)
        ]

        queue = asyncio.Queue()
        for q in cleaned_queries:
            await queue.put(q)

        sem = asyncio.Semaphore(MAX_SCRAPER_WORKERS * 2)

        async def worker(context):
            while not queue.empty():
                query = await queue.get()
                logger.info(f"Starting query: {query}")
                async with sem:
                    try:
                        job_list = await scrape_single_query_playwright(context, query, filters)
                        all_jobs.extend(job_list)
                        logger.info(f"Completed query: {query} ({len(job_list)} jobs)")
                    except Exception as e:
                        logger.error(f"Query {query} failed: {str(e)}", exc_info=True)
                    finally:
                        queue.task_done()

        tasks = []
        for ctx in context_pool:
            tasks.append(asyncio.create_task(worker(ctx)))

        await queue.join()
        await asyncio.gather(*tasks)

        for ctx in context_pool:
            await ctx.close()
        await browser.close()

    logger.info(f"Total jobs scraped: {len(all_jobs)}")
    return pd.DataFrame(all_jobs)

async def scrape_single_description_playwright(context, link):
    """Scrape a single job description."""
    page = await context.new_page()
    desc_text = None
    try:
        logger.debug(f"Scraping description from: {link}")
        await page.goto(link, timeout=30000, wait_until="domcontentloaded")
        await asyncio.sleep(REQUEST_DELAY * (0.8 + 0.4 * random.random()))

        desc_el = await page.query_selector('div[class*="job-details_jobDescription__"]')
        if desc_el:
            desc_text = await desc_el.inner_text()
            logger.debug(f"Successfully scraped description from {link}")
        else:
            logger.warning(f"No description element found at {link}")
    except Exception as e:
        logger.error(f"Failed to get description from {link}: {e}", exc_info=True)
    finally:
        await page.close()
    return desc_text.strip() if desc_text else None

async def scrape_job_descriptions_playwright(links):
    """Parallel job description scraping."""
    logger.info(f"Starting description scraping for {len(links)} jobs")
    start_time = time.time()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]
        )

        context_pool = [
            await browser.new_context(
                user_agent=get_random_user_agent(),
                viewport={"width": 1920, "height": 1080}
            )
            for _ in range(MAX_DESC_WORKERS)
        ]

        sem = asyncio.Semaphore(MAX_DESC_WORKERS * 2)
        results = [None] * len(links)

        async def fetch_desc(idx, link, context):
            async with sem:
                desc = await scrape_single_description_playwright(context, link)
                results[idx] = desc
                # Modified logging every 10 items
                if (idx + 1) % 10 == 0:
                    logger.info(f"PROGRESS: Descriptions scraped {idx+1}/{len(links)}")

        tasks = []
        for idx, link in enumerate(links):
            ctx = context_pool[idx % len(context_pool)]
            tasks.append(fetch_desc(idx, link, ctx))

        await asyncio.gather(*tasks)

        for ctx in context_pool:
            await ctx.close()
        await browser.close()

    logger.info(f"Description scraping completed in {time.time()-start_time:.2f}s")
    return results

# =============================================================================
# 4) Main Analysis Function
# =============================================================================
def analyze_jobs(resume_text, jobs_df, embedding_weight=0.7, batch_size=2, cooldown=5):
    logger.info(f"Analyzing {len(jobs_df)} jobs for similarity")
    
    if jobs_df.empty:
        logger.warning("No jobs found, skipping analysis")
        return jobs_df

    # 1) Compute similarity for ALL rows (if JobDescription is NaN, score=0.0)
    logger.info("Computing hybrid similarity scores")
    jobs_df['SimilarityScore'] = jobs_df['JobDescription'].apply(
        lambda desc: hybrid_similarity(resume_text, desc, embedding_weight=embedding_weight)
        if pd.notna(desc) else 0.0
    )

    # 2) Sort by similarity so we know which are top 50
    jobs_df_sorted = jobs_df.sort_values(by='SimilarityScore', ascending=False).reset_index(drop=False)
    top_50 = jobs_df_sorted.head(50).copy()  # Make a copy for the top 50

    # 3) Missing keywords & suggestions for top 50 only
    logger.info("Starting batch analysis (only on top 50) for missing keywords / suggestions / tips")

    # Pull out their JobDescription text
    top_50_desc = top_50['JobDescription'].fillna("").tolist()

    batch_results = batch_analyze_missing_keywords(
        cv_text=resume_text,
        job_descriptions=top_50_desc,
        batch_size=batch_size,
        cooldown=cooldown
    )

    # Extract the results for top 50
    mk_list = []
    rs_list = []
    at_list = []
    for item in batch_results:
        if item is None:
            mk_list.append([])
            rs_list.append("")
            at_list.append("")
        else:
            mk_list.append(item.get("missingKeywords", []))
            rs_list.append(item.get("resumeSuggestions", ""))
            at_list.append(item.get("applicationTips", ""))

    # 4) Place the AI results back into `top_50` DataFrame
    top_50['MissingKeywords'] = mk_list
    top_50['ResumeSuggestions'] = rs_list
    top_50['ApplicationTips'] = at_list

    # 5) For the other rows outside top 50, set empty placeholders
    # First ensure these columns exist in the overall sorted frame
    jobs_df_sorted['MissingKeywords'] = [[] for _ in range(len(jobs_df_sorted))]
    jobs_df_sorted['ResumeSuggestions'] = ["" for _ in range(len(jobs_df_sorted))]
    jobs_df_sorted['ApplicationTips'] = ["" for _ in range(len(jobs_df_sorted))]

    # Now merge top_50 results back into jobs_df_sorted by matching the index
    for i in top_50.index:
        jobs_df_sorted.at[i, 'MissingKeywords'] = top_50.at[i, 'MissingKeywords']
        jobs_df_sorted.at[i, 'ResumeSuggestions'] = top_50.at[i, 'ResumeSuggestions']
        jobs_df_sorted.at[i, 'ApplicationTips'] = top_50.at[i, 'ApplicationTips']

    # If you want to keep it sorted descending by similarity, just drop the old index column:
    final_df = jobs_df_sorted.drop(columns='index')

    logger.info(f"Analysis complete. Only top 50 have AI suggestions. Total rows: {len(final_df)}")

    return final_df


# =============================================================================
# 5) Full Processing Orchestration
# =============================================================================
def full_processing_flow(resume_path, roles=None, skills=None, filters=None):
    """Orchestrates the entire pipeline."""
    logger.info("\n=== STARTING PROCESSING FLOW ===")
    logger.info(f"Resume: {resume_path}")
    logger.info(f"Roles: {roles}")
    logger.info(f"Skills: {skills}")
    logger.info(f"Filters: {filters}")

    # Process resume
    try:
        logger.info("Processing resume...")
        resume_data = process_resume(resume_path)
        resume_text = resume_data['text']
        logger.info(f"Resume processed. Skills found: {resume_data['skills']}")
    except Exception as e:
        logger.error("Resume processing failed", exc_info=True)
        raise

    # Combine queries
    combined_search_terms = (roles or []) + (skills or [])
    cleaned_queries = [term.replace(" ", "-") for term in combined_search_terms if term.strip()]
    logger.info(f"Combined search queries: {cleaned_queries}")

    if not cleaned_queries:
        logger.error("No valid search terms provided")
        return {
            'top_matches': [],
            'csv_path': 'all_jobs.csv',
            'total_jobs_found': 0
        }

    # Scrape jobs
    try:
        logger.info("Starting job scraping...")
        jobs_df = asyncio.run(scrape_reed_jobs_today_playwright(cleaned_queries, filters or {}))
        logger.info(f"Initial job count: {len(jobs_df)}")
    except Exception as e:
        logger.error("Job scraping failed", exc_info=True)
        raise

    # Deduplicate
    initial_count = len(jobs_df)
    jobs_df.drop_duplicates(
        subset=['Title','Company','Salary','Location'],
        keep='first',
        inplace=True
    )
    logger.info(f"After deduplication: {len(jobs_df)} jobs (removed {initial_count - len(jobs_df)})")

    # Scrape descriptions
    try:
        logger.info("Starting description scraping...")
        desc_list = asyncio.run(scrape_job_descriptions_playwright(jobs_df['Link'].tolist()))
        jobs_df['JobDescription'] = desc_list
        logger.info(f"Jobs with descriptions: {jobs_df['JobDescription'].notna().sum()}")
    except Exception as e:
        logger.error("Description scraping failed", exc_info=True)
        raise

    # Analyze jobs
    try:
        logger.info("Starting job analysis...")
        jobs_df = analyze_jobs(resume_text, jobs_df)
        jobs_df.to_csv('all_jobs.csv', index=False)
        logger.info("Saved results to all_jobs.csv")
    except Exception as e:
        logger.error("Job analysis failed", exc_info=True)
        raise

    # Final results: now show top 50 instead of 30
    jobs_df_sorted = jobs_df.sort_values(by='SimilarityScore', ascending=False)
    top_matches = jobs_df_sorted.head(50).to_dict(orient='records')

    logger.info(f"Processing complete. Top {len(top_matches)} matches identified")

    return {
        'top_matches': top_matches,
        'csv_path': 'all_jobs.csv',
        'total_jobs_found': len(jobs_df)
    }

