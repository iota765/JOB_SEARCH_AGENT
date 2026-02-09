# ================= IMPORTS =================

import requests
import json
from bs4 import BeautifulSoup as Soup
from urllib.parse import quote

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from prompts import RESUME_ANALYSIS_PROMPT, RELATABLE_JOB_ROLES_PROMPT


# ================= CONFIG =================

MODEL = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

TOP_N = 3


# ================= HELPERS =================

def fetch_page(url):

    try:
        return requests.get(url, headers=HEADERS, timeout=15).text
    except:
        return ""


def load_resume(path):

    loader = PyPDFLoader(path)

    return "\n".join([p.page_content for p in loader.load()])


# ================= JOB SCRAPING =================

def get_google_jobs(keyword, location, level):

    level_map = {
        "ENTRY": "EARLY",
        "MID": "MID",
        "SENIOR": "ADVANCED"
    }

    g_level = level_map.get(level, "MID")

    url = (
        "https://www.google.com/about/careers/applications/jobs/results"
        f"?q={quote(keyword)}"
        f"&location={quote(location)}"
        f"&target_level={g_level}"
    )

    html = fetch_page(url)

    soup = Soup(html, "html.parser")

    jobs = []

    for a in soup.find_all("a", href=True):

        if "jobs/results/" in a["href"]:

            jobs.append({
                "title": a.get_text().strip() or "Google Role",
                "link": "https://www.google.com/about/careers/applications/" + a["href"]
            })

    return jobs


def get_other_company_jobs(company, keyword, location):

    links = {
        "Microsoft": f"https://jobs.careers.microsoft.com/global/en/search?q={quote(keyword)}&lc={quote(location)}",
        "Amazon": f"https://www.amazon.jobs/en/search?base_query={quote(keyword)}&loc_query={quote(location)}",
        "Walmart": f"https://careers.walmart.com/results?q={quote(keyword)}&l={quote(location)}"
    }

    return [{
        "title": f"{company} {keyword}",
        "link": links.get(company)
    }]


def extract_jobs(company, keyword, location, level):

    if company == "Google":
        return get_google_jobs(keyword, location, level)

    return get_other_company_jobs(company, keyword, location)


# ================= ROLE MATCHING =================

def match_roles(jobs, resume):

    job_titles = ", ".join([j["title"] for j in jobs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", RELATABLE_JOB_ROLES_PROMPT),
        ("human", "Job Roles:\n{jobs}\n\nResume:\n{resume}")
    ])

    chain = prompt | MODEL | StrOutputParser()

    response = chain.invoke({
        "jobs": job_titles,
        "resume": resume[:6000]
    })

    ai_lines = [
        l.strip()
        for l in response.split("\n")
        if l.strip()
    ]

    matched = []

    for job in jobs:
        if job["title"] in ai_lines:
            matched.append(job)

    if not matched:
        return jobs[:TOP_N]

    return matched[:TOP_N]


# ================= JOB CONTENT =================

def extract_job_content(url):

    try:
        return requests.get(
            f"https://r.jina.ai/{url}",
            timeout=15
        ).text
    except:
        return ""


# ================= REQUIREMENT EXTRACTION =================

def extract_requirements(job_content):

    prompt = ChatPromptTemplate.from_template("""
Extract ONLY from the job description.
Do NOT invent anything.

Return valid JSON:

{{
 "minimum": "...",
 "preferred": "...",
 "responsibilities": "..."
}}

If missing, write "Not specified".

JD:
{jd}
""")

    chain = prompt | MODEL | StrOutputParser()

    response = chain.invoke({
        "jd": job_content[:7000]
    })

    try:
        return json.loads(response)
    except:
        return {
            "minimum": "Not specified",
            "preferred": "Not specified",
            "responsibilities": "Not specified"
        }


# ================= ANALYSIS =================

def analyze_job(resume, requirements):

    prompt = ChatPromptTemplate.from_messages([
        ("system", RESUME_ANALYSIS_PROMPT),
        ("human", """
Minimum Qualification:
{minq}

Preferred Qualification:
{prefq}

Responsibilities:
{resp}

Resume:
{resume}
""")
    ])

    chain = prompt | MODEL | StrOutputParser()

    return chain.invoke({
        "minq": requirements["minimum"],
        "prefq": requirements["preferred"],
        "resp": requirements["responsibilities"],
        "resume": resume[:6000]
    })


# ================= MAIN PIPELINE =================

def run_pipeline(resume_path, company, keyword, location, level):

    # Load resume
    resume = load_resume(resume_path)

    # Get jobs
    jobs = extract_jobs(company, keyword, location, level)

    if not jobs:
        return []

    # Match roles
    selected = match_roles(jobs, resume)

    results = []

    # Analyze
    for job in selected:

        content = extract_job_content(job["link"])

        if not content:
            continue

        requirements = extract_requirements(content)

        report = analyze_job(resume, requirements)

        results.append({
            "title": job["title"],
            "link": job["link"],
            "report": report
        })

    return results
