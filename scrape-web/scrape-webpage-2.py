"""
This script is used to scrape a webpage and extract the text content.

user will provide the url of the webpage and the script will scrape the webpage and extract the text content.
use HuggingFace's pipeline to extract the text content.

# Cell 1: Install required dependencies
!pip install playwright beautifulsoup4 huggingface_hub -q
!playwright install chromium
!sudo playwright install-deps chromium

------------------------------------------------------------------------------------------------
Output:
------------------------------------------------------------------------------------------------
Starting scraper process...
[+] Loading URL: https://jpmc.fa.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1001/job/210681626

=== EXTRACTED JOB DETAILS (JSON) ===
{
  "url": "https://jpmc.fa.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1001/job/210681626",
  "jobid": "210681626",
  "jobtitle": "Software Engineer III - React",
  "location": "1 Cabot Square, London, E14 4QJ, GB",
  "posting_date": "09/14/2026, 09:24 AM",
  "jobdescription": "Out of the successful launch of Chase in 2021, we\u2019re a new team, with a new mission. We\u2019re creating products that solve real world problems and put customers at the center - all in an environment that nurtures skills and helps you realize your potential. Our team is key to our success. We\u2019re people-first. We value collaboration, curiosity and commitment.\n\nAs a Software Engineer at JPMorgan Chase within the accelerator, you are the heart of this venture, focused on getting smart ideas into the hands of our customers. You have a curious mindset, thrive in collaborative squads, and are passionate about new technology. By your nature, you are also solution-oriented, commercially savvy and have a head for fintech. You thrive in working in tribes and squads that focus on specific products and projects \u2013 and depending on your strengths and interests, you'll have the opportunity to move between them.\n\nWhile we\u2019re looking for professional skills, culture is just as important to us. We understand that everyone's unique \u2013 and that diversity of thought, experience and background is what makes a good team, great. By bringing people with different points of view together, we can represent everyone and truly reflect the communities we serve. This way, there's scope for you to make a huge difference \u2013 on us as a company, and on our clients and business partners around the world.",
  "required_skills": [
    "Commercial experience with React with TypeScript, with a strong engineering-first mindset, an ability to solve problems elegantly with an emphasis on scalability and performance.",
    "Thrive in an environment of uncertainty and rapid changes in direction.",
    "A team player who thrives in working in cross-functional teams, driving things forward.",
    "Demonstrate a knack for rapid learning and possess an ambitious, results-driven personality.",
    "Excellent communication and organizational skills to work effectively within a fast-paced, high-performing team.",
    "Ability to excel as part of a cohesive team in a high-speed environment.",
    "Hands-on experience using enterprise-authorized AI-assisted software development tools within the work environment (e.g., for coding, test creation, troubleshooting, or documentation) with demonstrated ability to critically evaluate, validate, and refine AI-generated outputs for correctness, performance, and security.",
    "Understanding of responsible AI use in engineering workflows, including data sensitivity considerations, secure handling of inputs/outputs, and adherence to resiliency and security expectations; ability to guide peers on safe and effective usage within team practices."
  ],
  "roles_and_responsibilities": [
    "Significant contributions to designing and building scalable and performant front-end solutions written in TypeScript using React",
    "Working across B2C and B2B features",
    "Maintaining high code quality and participating in a strong engineering culture.",
    "Writing testable code, built for scalability, reliability and resilience.",
    "Working as part of a cross-functional squad, taking ownership of issues and working together with necessary stakeholders to drive delivery.",
    "An active participant in technical discussions, contributing with insightful and informed opinions to the decision making process.",
    "Leverages enterprise-authorized AI coding assist tools within the work environment to improve code quality, delivery speed, and productivity across complex deliverables (e.g., code generation/refactoring, unit test creation, documentation), while validating outputs through peer review, automated testing, and secure coding standards; contributes learnings and reusable patterns to improve broader team effectiveness.",
    "Applies knowledge of tools within the Software Development Life Cycle toolchain, including enterprise-authorized AI-assisted development and automation capabilities, to improve the value realized by automation"
  ]
}

"""
#------------------------------------------------------------------------------------------------
# Input: url of the webpage -> scrape the webpage and extract the text content. -> return in JSON format
#------------------------------------------------------------------------------------------------

import asyncio
import json
import os
import re
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from huggingface_hub import InferenceClient

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Set HF_TOKEN in your environment (or a local .env file that is not committed)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Free open-source instruction-tuned model hosted on Hugging Face Inference API
HF_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# ------------------------------------------------------------------
# 1. SCRAPE WEBPAGE HTML (Playwright)
# ------------------------------------------------------------------
async def fetch_webpage_content(url: str) -> str:
    """Renders the webpage using Playwright and extracts body text."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Use a realistic browser user-agent to bypass basic bot blockers
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            print(f"[+] Loading URL: {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(3000)  # Wait for dynamic JS content to render
            content = await page.content()
        except Exception as e:
            print(f"[-] Error fetching page: {e}")
            content = ""
        finally:
            await browser.close()

        return content

# ------------------------------------------------------------------
# 2. CLEAN HTML FOR LLM CONTEXT
# ------------------------------------------------------------------
def clean_html(html_content: str) -> str:
    """Strips tags, CSS, scripts, and navigation elements to compress context size."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Prioritize extracting content from the job-details-wrapper if it exists
    job_details_wrapper = soup.find('job-details-wrapper')

    if job_details_wrapper:
        # Remove non-content tags only within the job details wrapper
        for element in job_details_wrapper(["script", "style", "nav", "footer", "header", "svg", "button", "iframe", "link", "meta", "img"]):
            element.decompose()
        text = job_details_wrapper.get_text(separator="\n")
    else:
        # Fallback to general cleaning if the specific wrapper isn't found
        for element in soup(["script", "style", "nav", "footer", "header", "svg", "button", "iframe", "link", "meta", "img"]):
            element.decompose()
        text = soup.get_text(separator="\n")

    # Clean whitespace and empty lines
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = "\n".join(chunk for chunk in chunks if chunk)

    # No arbitrary truncation here, let the LLM's max_tokens handle the length
    return cleaned_text

# ------------------------------------------------------------------
# 3. LLM EXTRACTION VIA HUGGINGFACE
# ------------------------------------------------------------------
def extract_job_details_ai(url: str, webpage_text: str, token: str) -> dict:
    """Uses Hugging Face Free Inference API to convert text into structured JSON."""
    client = InferenceClient(api_key=token)

    system_prompt = """You are an expert AI recruiter and data extractor.
Your job is to analyze raw job posting text and extract specific job details into valid, strictly formatted JSON.
Do not include markdown formatting or extra text outside the JSON object."""

    user_prompt = f"""
Target Webpage URL: {url}

Extracted Webpage Text:
{webpage_text}

Extract the job details according to these strict rules:
1. "url": Use the target webpage URL provided above.
2. "jobtitle": Extract the official job title.
3. "jobid": Extract the official job/requisition ID if present. IF NO JOBID IS FOUND, set "jobid" EQUAL TO the "jobtitle".
4. "jobdescription": Extract the complete and verbatim job description section from the text, without summarizing.
5. "required_skills": Extract ALL individual required skills as they are explicitly listed or presented in distinct points on the page. Each skill must be a separate, distinct string in the array, retaining its exact wording. DO NOT combine multiple skills into a single string within the array, even if they appear close together.
6. "roles_and_responsibilities": Extract ALL individual roles and responsibilities as they are explicitly listed or presented in distinct points on the page. Each responsibility must be a separate, distinct string in the array, retaining its exact wording. DO NOT combine multiple responsibilities into a single string within the array, even if they appear close together.

Return ONLY a valid JSON object matching this schema:
{{
  "url": "{url}",
  "jobid": "...",
  "jobtitle": "...",
  "location": "...",
  "posting_date": "...",
  "jobdescription": "...",
  "required_skills": ["..."],
  "roles_and_responsibilities": ["..."]
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat_completion(
            model=HF_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=4000 # Increased max_tokens to allow for full verbatim description
        )

        raw_output = response.choices[0].message.content.strip()

        # Clean potential markdown fences ```json ... ``` from output
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)
            return json.loads(clean_json_str)
        else:
            return json.loads(raw_output)

    except Exception as e:
        print(f"[-] AI Parsing Error: {e}")
        return {"error": str(e)}

# ------------------------------------------------------------------
# 4. MAIN PIPELINE EXECUTION
# ------------------------------------------------------------------
async def scrape_job_page(url: str, hf_token: str) -> str:
    """Executes full pipeline: Fetch -> Clean -> Parse -> JSON Output."""
    # Step 1: Scrape HTML
    raw_html = await fetch_webpage_content(url)
    if not raw_html:
        return json.dumps({"error": "Failed to fetch webpage content"}, indent=2)

    # Step 2: Clean content
    clean_text = clean_html(raw_html)

    # Step 3: Extract JSON via Hugging Face LLM
    job_json = extract_job_details_ai(url, clean_text, hf_token)

    # Enforce jobid fallback if LLM missed rule
    if not job_json.get("jobid") or job_json.get("jobid").lower() == "none":
        job_json["jobid"] = job_json.get("jobtitle", "N/A")

    return json.dumps(job_json, indent=2)

# ------------------------------------------------------------------
# DEMONSTRATION / RUNNER
# ------------------------------------------------------------------
# Replace with a real job URL to test
sample_job_url = "https://jpmc.fa.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1001/job/210681626"

print("Starting scraper process...")
# Run async pipeline in Colab environment
output_json = await scrape_job_page(sample_job_url, HF_TOKEN)

print("\n=== EXTRACTED JOB DETAILS (JSON) ===")
print(output_json)
