"""
This script is used to scrape a webpage and extract the text content.

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

"""
#------------------------------------------------------------------------------------------------
# Simple function to fetch the HTML content of a webpage
#------------------------------------------------------------------------------------------------
async def fetch_html(page_url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(page_url, wait_until='networkidle')
        content = await page.content()
        await browser.close()
        return content

#page_url = "https://jpmc.fa.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1001/jobs"
page_url = "https://jpmc.fa.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1001/job/210790530"
value = await fetch_html(page_url)
print(value)

