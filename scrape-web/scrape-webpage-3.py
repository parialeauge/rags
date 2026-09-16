"""
this script is using praidapi jsearch api to fetch the jobs and then use hugging face ai 
to enrich the jobs and return the jobs in JSON format.

!pip install playwright beautifulsoup4 langchain-openai python-dotenv
!playwright install-deps
!playwright install

import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Cell 1: Install required dependencies
!pip install playwright beautifulsoup4 huggingface_hub -q
!playwright install chromium
!sudo playwright install-deps chromium

# Cell 1: Install required dependencies
!pip install requests huggingface_hub -q

------------------------------------------------------------------------------------------------
Output:
------------------------------------------------------------------------------------------------
{
  "top_10_jobs": [
    {
      "rank": 1,
      "job_title": "Software Engineering, Team Lead - U.S. Based Hybrid Opportunity",
      "company": "Common App",
      "location": "Arlington, Virginia, US",
      "apply_link": "https://www.linkedin.com/jobs/view/software-engineering-team-lead-u-s-based-hybrid-opportunity-at-common-app-4467213918",
      "posted_date": "2026-09-14T17:00:00.000Z",
      "jobdescription": "About Us\n\nDo you have a passion for higher education? Do you want to make a positive impact on the college admissions process? Our staff help to remove barriers and encourage students to forge their path to a better future. Common App is a national not-for-profit organization dedicated to the pursuit of access, equity, and integrity in the college admission process. Each year we support more than 1 million students, one-third of whom are first-generation, as they apply to our more than 1100 diverse member colleges & universities using the Common App's free online application.\n\nIf you are an experienced full-stack professional with leadership experience and want to be part of a mission-driven non-profit that uses innovative technology to advance the college admission process, Common App may be a great match for you. Common App is currently searching for a Software Engineering, Team Lead.\n\nResponsibilities\n\nOur Software Engineering, Team Lead, reports to our Senior Director, Software Engineering, and is part of a team of engineers designing and developing both web-based and mobile software products for Common App. This role will provide technical guidance and mentor a team of Software Developers while implementing high quality software products that meet the needs of The Common Application community.\n\nRequirements\n\nQUALIFICATIONS\n\nThis role requires:\n\n\u2022 Candidates must live in the United States\n\u2022 Hybrid Work & Travel: Employees can live anywhere in the US or its territories, with the willingness and ability to travel for periodic in-person events\u2014including semi-annual Common App retreats, department retreats, and strategic leadership sessions\n\u2022 Bachelor's degree or higher in Computer Science or Engineering or relevant field; or equivalent combination of education and experience.\n\u2022 8-10 years of successful, progressive experience in software development\n\u2022 6-8 years of experience as people manager.\n\nWeb & Frontend Development\n\n\u2022 Demonstrates advanced proficiency in Web Application Development using JavaScript, HTML, CSS, and modern SPA frameworks such as Angular, TypeScript, with the ability to define architecture, guide team implementation, and influence frontend standards across the organization\n\nBackend & API Development\n\n\u2022 Expert in backend and API development using REST using C#, .NET Core, ORMs, and applying Domain Driven Design and Dependency Injection\n\u2022 Providing guidance on architecture and best practices across teams.\n\u2022 Expert in relational and non-relational databases, including MySQL and NoSQL data stores(AWS DynamoDB), with the ability to advise on database strategy and enterprise-level implementations\n\nCloud & DevOps\n\n\u2022 Demonstrates advanced proficiency in Continuous Integration / Continuous Delivery (CI/CD) using tools such as AWS CodeBuild, AWS Lambda, CloudFormation, GitHub Actions, Jenkins, and source control systems (GitHub), with guidance on organizational CI/CD strategy\n\u2022 Demonstrates advanced proficiency with AWS services, including Fargate/ECS, S3, SWF, Step Functions, SQS, SES, ElasticCache, CloudFront, EC2, Serverless Lambda, VPC, and DynamoDB, with the ability to provide strategic guidance on cloud architecture\n\nAgile & Software Practices\n\n\u2022 Expert in Agile development environments, with the ability to shape practices across teams and guide the adoption of best practices\n\u2022 An expert across the software development lifecycle, including:\n\u2022 Code repository and branching strategies\n\u2022 Code reviews and coding standards\n\u2022 Code deployment processes\n\u2022 Quality assurance practices\n\u2022 Expert in debugging and troubleshooting production environments, providing oversight and guidance to teams on resolution approaches\n\u2022 Expert in evaluating and recommending solutions based on complex POC, with guidance on adoption and organizing impacts\n\nSecurity & Compliance\n\n\u2022 Demonstrates advanced proficiency in understanding OWASP Top 10 security vulnerabilities and secure coding practices, providing guidance to teams on secure architecture and compliance\n\nThe ideal candidate will possess:\n\n\u2022 AWS Solution Architect certification\n\nPAY RANGE\n\n\u2022 $135,000-$149,034\n\nBenefits\n\nCommon App is a virtual first environment. We value our employees' time and efforts. Our commitment to your success is enhanced by our competitive salary and an extensive benefits package including:\n\n\u2022 Work-Life balance\n\u2022 Virtual-first office\n\u2022 Paid Time Off (PTO)\n\u2022 Seven company-wide holidays\n\u2022 Nine floating holidays*\n\u2022 Sick leave\n\u2022 Monthly mental health day\n\u2022 floating holidays prorated depending on start date\n\u2022 Virtual-first support\n\u2022 Choice of PC of MAC laptop\n\u2022 May choose an external monitor, keyboard, mouse, and/or headset\n\u2022 One-time office set-up stipend\n\u2022 Monthly remote work stipend\n\u2022 Monthly mobile stipend\n\u2022 Financial security\n\u2022 Market-based salaries\n\u2022 Performance-based bonus\n\u2022 403(b) retirement plan\n\u2022 5% company contribution\n\u2022 additional 5% company match\n\u2022 3-year vesting schedule\n\u2022 Participation may begin immediately\n\u2022 Health & wellness\n\u2022 Choice of two health insurance plans\n\u2022 Health Savings Account, depending on health plan selection\n\u2022 Medical Flexible Savings Account, depending on health plan selection\n\u2022 Vision insurance\n\u2022 Dental insurance\n\u2022 Insurance coverage begins on the date of hire\n\u2022 Dependent Care Flexible Spending Account\n\u2022 Maven virtual clinic for women's and family health\n\u2022 Company provided life and ad&d insurance\n\u2022 Opportunity to purchase additional life insurance for self, spouse, and dependents\n\u2022 Company provided short and long-term disability insurance\n\u2022 Career development\n\u2022 Budgeted annual funds for professional development\n\u2022 Growth opportunities within the company\n\u2022 Additional perks\n\u2022 Mutual of Omaha Employee Assistance Program\n\u2022 Mutual of Omaha will preparation services\n\u2022 Mutual of Omaha travel assistance\n\u2022 Payroll dedication pet insurance through PinPaws\n\u2022 1Password family account\n\nWe work to maintain the best possible environment for our staff, where people can learn and grow. We strive to provide a diverse, collaborative, team-oriented, creative environment where each person feels encouraged to contribute to our processes, decisions, planning, and culture.\n\nHow Do i Apply\n\nTo apply for this opportunity, send your resume and cover letter with salary expectations.\n\nPROTECTING YOUR PERSONAL INFORMATION:\n\nDuring the recruiting process, please note that Common App will never:\n\n\u2022 Provide a job offer without an interview\n\u2022 Ask for payment to process documents, purchase equipment or for any other reason\n\u2022 Request banking or credit card information\n\u2022 Direct you to third-party services to obtain visas or other documentation\n\nAs we work alongside you through our recruitment process, please remain alert and never provide financial information or payment to anyone claiming to offer a job opportunity.\n\nIf you believe you're a victim of a job scam, report it to the Federal Trade Commission (FTC) or your state attorney general. To learn more about job scams, read the FBI's public service announcement or visit the FTC site.",
      "key_skills": []
    },
    {
      "rank": 2,
      "job_title": "Software Engineer I",
      "company": "Reflexive Concepts",
      "location": "Annapolis Junction, Maryland, US",
      "apply_link": "https://www.linkedin.com/jobs/view/software-engineer-i-at-reflexive-concepts-4467257033",
      "posted_date": "2026-09-14T19:00:00.000Z",
      "jobdescription": "Reflexive Concepts is seeking a skilled Software Engineer to join our team!\n\nSpecifically, we are looking for a candidate with strong Java development experience to support the Digital Network Analytics (DNA) mission.\n\nQualifications\n\n\u2022 Seven (7) years experience as a SWE in programs and contracts of similar scope, type, and complexity\n\u2022 Bachelor\u2019s degree in Computer Science or related discipline from an accredited college or university\n\u2022 Four (4) years of additional SWE experience on projects with similar software processes may be substituted for a bachelor\u2019s degree\nRequired:\n\n\u2022 Experience with Java\n\u2022 Knowledge of Confluence, JIRA or similar\n\u2022 Requirements gathering, documentation and work-flow\n\u2022 Product result driven / problem solving ability\n\nDesired:\n\n\u2022 Experience with AWS\n\u2022 Experience with Maven\n\u2022 Experience with Git, GitLab\n\u2022 Experience with MapReduce\n\u2022 Experience with JUnit testing and code coverage\n\u2022 Experience with Python and streaming capabilities\n\u2022 Experience with Linux",
      "key_skills": []
    },
    {
      "rank": 3,
      "job_title": "Remote Software Developer",
      "company": "Turing",
      "location": "N/A",
      "apply_link": "https://www.linkedin.com/jobs/view/remote-software-developer-at-turing-4466189161",
      "posted_date": "2026-09-15T08:00:00.000Z",
      "jobdescription": "About Us:\n\nBased in San Francisco, California, Turing is the world\u2019s leading research accelerator for frontier AI labs and a trusted partner for global enterprises deploying advanced AI systems. Turing supports customers in two ways: first, by accelerating frontier research with high-quality data, advanced training pipelines, plus top AI researchers who specialize in software engineering, logical reasoning, STEM, multilinguality, multimodality, and agents; and second, by applying that expertise to help enterprises transform AI from proof of concept into proprietary intelligence with systems that perform reliably, deliver measurable impact, and drive lasting results on the P&L.\n\nIdeal Background:\n\nThis role is ideal for engineers who have built production systems at companies like Google, Microsoft, Apple, Amazon, Meta, or similar high-scale engineering organizations. We especially welcome graduates from top computer science programs such as Stanford, MIT, Carnegie Mellon, UC Berkeley, Georgia Tech, and comparable institutions \u2014 though exceptional experience and skill always take precedence over pedigree.\n\nProject Overview:\n\nAs a Software Engineering evaluator, you will create cutting-edge datasets for training, benchmarking, and advancing large language models, collaborating closely with researchers. This includes curating code examples, providing precise solutions, and making corrections \u2014 with a primary focus on Python across backend services, data pipelines, and ML infrastructure, alongside JavaScript (including ReactJS), C/C++, Java, Rust, and Go. You will evaluate and refine AI-generated code for efficiency, scalability, and reliability, and work with cross-functional teams to enhance enterprise-level AI-driven coding solutions.\n\nWhat Does a Typical Day Look Like?\n\u2022 Work on AI model training initiatives by curating code examples, building solutions, and correcting code \u2014 primarily in Python, with additional work in JavaScript (including ReactJS), C/C++, Java, Rust, and Go.\n\u2022 Evaluate and refine AI-generated code to ensure that it is efficient, scalable, and reliable.\n\u2022 Collaborate with cross-functional teams to enhance AI-driven coding solutions against industry performance benchmarks.\n\u2022 Build agents and automated verification tools in Python that can verify the quality of code and identify error patterns.\n\u2022 Hypothesize on steps in the software engineering cycle (prototyping, architecture design, API design, production implementation, launch, experiments, monitoring, operational maintenance) and evaluate model capabilities on them.\n\u2022 Design verification mechanisms that can automatically verify a solution to a software engineering task.\n\nRequired Skills:\n\u2022 Several years of software engineering experience (3 years or more)\n\u2022 Strong expertise in Python with deep knowledge of frameworks, tooling, and best practices for building production-grade software.\n\u2022 Experience building full-stack applications and deploying scalable software using modern languages and tools.\n\u2022 Deep understanding of software architecture, design, development, debugging, and code quality/review assessment.\n\u2022 Excellent oral and written communication skills for clear, structured evaluation rationales.\n\nEngagement Details:\n\u2022 Commitment: flexible engagement, minimum 10 hrs/week, up to 40 hrs/week\n\u2022 Type: Contractor (no medical/paid leave)\n\u2022 Duration: 1 month (potential extensions based on performance and fit)\n\u2022 Location: Candidates must be based in the United States\n\nEvaluation Process:\n\u2022 The application process takes 15\u201330 minutes.\n\u2022 Completion of an AI video interview is required.",
      "key_skills": []
    }]
    

"""
#------------------------------------------------------------------------------------------------
# Simple function to fetch the HTML content of a webpage
#------------------------------------------------------------------------------------------------
# Cell 2: Complete Script Returning JSON Output
import requests
import json
import re
from huggingface_hub import InferenceClient

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
RAPIDAPI_KEY = "YOUR_RAPIDAPI_KEY_HERE"  # Replace with your RapidAPI Key
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"  # Replace with your Hugging Face Token

HF_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# ------------------------------------------------------------------
# 1. FETCH JOBS VIA JSEARCH API
# ------------------------------------------------------------------
def fetch_jobs_jsearch(query: str = "Software Engineer in USA", num_jobs: int = 10) -> list:
    url = "https://jsearch.p.rapidapi.com/search-v2"

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    params = {
        "query": str(query),
        "page": "1",
        "num_pages": "1",
        "date_posted": "today"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            return []

        data = response.json()
        if data.get("status") == "OK":
            raw_data = data.get("data")

            jobs_list = []
            if isinstance(raw_data, list):
                jobs_list = raw_data
            elif isinstance(raw_data, dict):
                jobs_list = raw_data.get("jobs", raw_data.get("data", []))

            if isinstance(jobs_list, list):
                return jobs_list[:num_jobs]
            return []
        return []

    except Exception:
        return []

# ------------------------------------------------------------------
# 2. PROCESS AND ENRICH JOBS USING HUGGING FACE AI
# ------------------------------------------------------------------
def process_and_rank_jobs_ai(jobs_raw: list, hf_token: str) -> dict:
    if not jobs_raw:
        return {"top_10_jobs": []}

    condensed_jobs = []
    for idx, job in enumerate(jobs_raw, start=1):
        condensed_jobs.append({
            "index": idx,
            "title": job.get("job_title", "N/A"),
            "company": job.get("employer_name", "N/A"),
            "location": f"{job.get('job_city', '')}, {job.get('job_state', '')}, {job.get('job_country', '')}",
            "apply_link": job.get("job_apply_link", "N/A"),
            "description_snippet": job.get("job_description", "")[:400]
        })

    client = InferenceClient(api_key=hf_token)

    system_prompt = """You are an AI career advisor.
Analyze the provided list of software engineering jobs posted in the last 24 hours.
Extract and return a clean JSON object containing the job listings with direct application links and key skills."""

    user_prompt = f"""
Raw Jobs Data:
{json.dumps(condensed_jobs, indent=2)}

Return ONLY a valid JSON object matching this structure:
{{
  "top_10_jobs": [
    {{
      "rank": 1,
      "job_title": "...",
      "company": "...",
      "location": "...",
      "apply_link": "...",
      "key_skills": ["skill1", "skill2"]
    }}
  ]
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
            max_tokens=2000
        )

        raw_output = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return json.loads(raw_output)

    except Exception:
        fallback_jobs = []
        for idx, item in enumerate(condensed_jobs, start=1):
            fallback_jobs.append({
                "rank": idx,
                "job_title": item["title"],
                "company": item["company"],
                "location": item["location"],
                "apply_link": item["apply_link"],
                "key_skills": []
            })
        return {"top_10_jobs": fallback_jobs}

# ------------------------------------------------------------------
# 3. MAIN PIPELINE EXECUTION (RETURNS JSON STRING)
# ------------------------------------------------------------------
def get_jobs_json(query: str = "Software Engineer in USA", num_jobs: int = 10) -> str:
    """
    Executes the job search and AI enrichment pipeline.
    Returns a valid JSON string formatted with 2-space indentation.
    """
    raw_jobs = fetch_jobs_jsearch(query=query, num_jobs=num_jobs)

    if not raw_jobs:
        return json.dumps({
            "status": "error",
            "message": "No jobs retrieved. Check RapidAPI Key, query, or JSearch subscription.",
            "top_10_jobs": []
        }, indent=2)

    result = process_and_rank_jobs_ai(raw_jobs, HF_TOKEN)
    result["status"] = "success"

    return json.dumps(result, indent=2)

# Execute and capture JSON output
json_output = get_jobs_json(query="Software Engineer in USA", num_jobs=10)

# Display JSON output
print(json_output)

# Cell 2: Complete Script Returning Updated JSON Output
import requests
import json
import re
from huggingface_hub import InferenceClient

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
#RAPIDAPI_KEY = "YOUR_RAPIDAPI_KEY_HERE"  # Replace with your RapidAPI Key
#HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"  # Replace with your Hugging Face Token

HF_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# ------------------------------------------------------------------
# 1. FETCH JOBS VIA JSEARCH API
# ------------------------------------------------------------------
def fetch_jobs_jsearch(query: str = "Software Engineer in USA", num_jobs: int = 10) -> list:
    url = "https://jsearch.p.rapidapi.com/search-v2"

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    params = {
        "query": str(query),
        "page": "1",
        "num_pages": "1",
        "date_posted": "today"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            return []

        data = response.json()
        if data.get("status") == "OK":
            raw_data = data.get("data")

            jobs_list = []
            if isinstance(raw_data, list):
                jobs_list = raw_data
            elif isinstance(raw_data, dict):
                jobs_list = raw_data.get("jobs", raw_data.get("data", []))

            if isinstance(jobs_list, list):
                return jobs_list[:num_jobs]
            return []
        return []

    except Exception:
        return []

# ------------------------------------------------------------------
# 2. PROCESS AND ENRICH JOBS USING HUGGING FACE AI
# ------------------------------------------------------------------
def process_and_rank_jobs_ai(jobs_raw: list, hf_token: str) -> dict:
    if not jobs_raw:
        return {"top_10_jobs": []}

    condensed_jobs = []
    for idx, job in enumerate(jobs_raw, start=1):
        condensed_jobs.append({
            "index": idx,
            "title": job.get("job_title", "N/A"),
            "company": job.get("employer_name", "N/A"),
            "location": f"{job.get('job_city', '')}, {job.get('job_state', '')}, {job.get('job_country', '')}",
            "apply_link": job.get("job_apply_link", "N/A"),
            "posted_date": job.get("job_posted_at_datetime_utc", job.get("job_posted_at_timestamp", "N/A")),
            "description_raw": job.get("job_description", "")[:1000]  # Expanded context for AI summary
        })

    client = InferenceClient(api_key=hf_token)

    system_prompt = """You are an AI career advisor.
Analyze the provided list of software engineering jobs posted in the last 24 hours.
Extract and return a clean JSON object containing job details including job description and posted date."""

    user_prompt = f"""
Raw Jobs Data:
{json.dumps(condensed_jobs, indent=2)}

Return ONLY a valid JSON object matching this structure:
{{
  "top_10_jobs": [
    {{
      "rank": 1,
      "job_title": "...",
      "company": "...",
      "location": "...",
      "apply_link": "...",
      "posted_date": "...",
      "jobdescription": "A concise summary of the position and core focus.",
      "key_skills": ["skill1", "skill2"]
    }}
  ]
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
            max_tokens=2500
        )

        raw_output = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return json.loads(raw_output)

    except Exception:
        fallback_jobs = []
        for idx, item in enumerate(condensed_jobs, start=1):
            fallback_jobs.append({
                "rank": idx,
                "job_title": item["title"],
                "company": item["company"],
                "location": item["location"],
                "apply_link": item["apply_link"],
                "posted_date": item["posted_date"],
                "jobdescription": item["description_raw"][:250] + "...",
                "key_skills": []
            })
        return {"top_10_jobs": fallback_jobs}

# ------------------------------------------------------------------
# 3. MAIN PIPELINE EXECUTION (RETURNS JSON STRING)
# ------------------------------------------------------------------
def get_jobs_json(query: str = "Software Engineer in USA", num_jobs: int = 10) -> str:
    """
    Executes the job search and AI enrichment pipeline.
    Returns a valid JSON string with jobdescription and posted_date included.
    """
    raw_jobs = fetch_jobs_jsearch(query=query, num_jobs=num_jobs)

    if not raw_jobs:
        return json.dumps({
            "status": "error",
            "message": "No jobs retrieved. Check RapidAPI Key, query, or JSearch subscription.",
            "top_10_jobs": []
        }, indent=2)

    result = process_and_rank_jobs_ai(raw_jobs, HF_TOKEN)
    result["status"] = "success"

    return json.dumps(result, indent=2)

# Execute and capture JSON output
json_output = get_jobs_json(query="Software Engineer in USA", num_jobs=10)

# Display JSON output
print(json_output)

# Cell 2: Complete Script Returning Full, Unaltered Job Descriptions
import requests
import json
import re
from huggingface_hub import InferenceClient

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
#RAPIDAPI_KEY = "YOUR_RAPIDAPI_KEY_HERE"  # Replace with your RapidAPI Key
#HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"  # Replace with your Hugging Face Token

HF_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# ------------------------------------------------------------------
# 1. FETCH JOBS VIA JSEARCH API
# ------------------------------------------------------------------
def fetch_jobs_jsearch(query: str = "Software Engineer in USA", num_jobs: int = 10) -> list:
    url = "https://jsearch.p.rapidapi.com/search-v2"

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    params = {
        "query": str(query),
        "page": "1",
        "num_pages": "1",
        "date_posted": "today"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            return []

        data = response.json()
        if data.get("status") == "OK":
            raw_data = data.get("data")

            jobs_list = []
            if isinstance(raw_data, list):
                jobs_list = raw_data
            elif isinstance(raw_data, dict):
                jobs_list = raw_data.get("jobs", raw_data.get("data", []))

            if isinstance(jobs_list, list):
                return jobs_list[:num_jobs]
            return []
        return []

    except Exception:
        return []

# ------------------------------------------------------------------
# 2. EXTRACT SKILLS VIA AI & PRESERVE EXACT JOB DESCRIPTIONS
# ------------------------------------------------------------------
def process_and_rank_jobs(jobs_raw: list, hf_token: str) -> dict:
    if not jobs_raw:
        return {"top_10_jobs": []}

    # Extract raw job descriptions without truncation or rephrasing
    structured_jobs = []
    for idx, job in enumerate(jobs_raw, start=1):
        # Format location
        city = job.get('job_city', '')
        state = job.get('job_state', '')
        country = job.get('job_country', '')
        location_parts = [p for p in [city, state, country] if p]
        location_str = ", ".join(location_parts) if location_parts else "N/A"

        # Format posted date
        posted_date = job.get("job_posted_at_datetime_utc") or job.get("job_posted_at_timestamp") or "N/A"

        structured_jobs.append({
            "rank": idx,
            "job_title": job.get("job_title", "N/A"),
            "company": job.get("employer_name", "N/A"),
            "location": location_str,
            "apply_link": job.get("job_apply_link", "N/A"),
            "posted_date": str(posted_date),
            "jobdescription": job.get("job_description", ""),  # Unaltered full text
            "description_snippet_for_ai": job.get("job_description", "")#[:800]  # Used only for AI skill extraction
        })

    # Optional: Use Hugging Face AI strictly to extract key skills keywords
    try:
        client = InferenceClient(api_key=hf_token)

        skills_payload = [
            {"index": item["rank"], "text": item["description_snippet_for_ai"]}
            for item in structured_jobs
        ]

        system_prompt = "You are an AI assistant. Extract technical and professional key skills array for each job index."
        user_prompt = f"""
Given these job listings:
{json.dumps(skills_payload, indent=2)}

Return ONLY a JSON object mapping each index to its extracted key skills array:
{{
  "1": ["skill1", "skill2"],
  "2": ["skill1", "skill2"]
}}
"""

        response = client.chat_completion(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )

        raw_output = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        skills_map = json.loads(json_match.group(0)) if json_match else json.loads(raw_output)

        # Merge extracted skills into output payload
        for item in structured_jobs:
            idx_str = str(item["rank"])
            item["key_skills"] = skills_map.get(idx_str, [])
            del item["description_snippet_for_ai"]  # Clean temporary key

    except Exception:
        # Fallback if skill extraction fails
        for item in structured_jobs:
            item["key_skills"] = []
            if "description_snippet_for_ai" in item:
                del item["description_snippet_for_ai"]

    return {"top_10_jobs": structured_jobs}

# ------------------------------------------------------------------
# 3. MAIN PIPELINE EXECUTION (RETURNS JSON STRING)
# ------------------------------------------------------------------
def get_jobs_json(query: str = "Software Engineer in USA", num_jobs: int = 10) -> str:
    """
    Executes the job search and AI enrichment pipeline.
    Returns valid JSON string with exact, unaltered job descriptions.
    """
    raw_jobs = fetch_jobs_jsearch(query=query, num_jobs=num_jobs)

    if not raw_jobs:
        return json.dumps({
            "status": "error",
            "message": "No jobs retrieved. Check RapidAPI Key, query, or JSearch subscription.",
            "top_10_jobs": []
        }, indent=2)

    result = process_and_rank_jobs(raw_jobs, HF_TOKEN)
    result["status"] = "success"

    return json.dumps(result, indent=2)

# Execute and display JSON output
json_output = get_jobs_json(query="Software Engineer in USA", num_jobs=10)
print(json_output)

