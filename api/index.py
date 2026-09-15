import os
import json
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Skill-to-Career Mapping Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class AnalyzeRequest(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=500
    )

    location: str = Field(
        default="India",
        min_length=2,
        max_length=100
    )


# ---------------------------------------------------------
# Environment variable helper
# ---------------------------------------------------------

def env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise HTTPException(
            status_code=500,
            detail=f"Missing Vercel environment variable: {name}"
        )

    return value


# ---------------------------------------------------------
# Job search using JSearch / RapidAPI
# ---------------------------------------------------------

def search_jobs_impl(skill: str, location: str):

    rapid_api_key = env("RAPID_API_KEY")

    url = "https://jsearch.p.rapidapi.com/search-v2"

    headers = {
        "x-rapidapi-key": rapid_api_key,
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
    }

    # Start with the user's skill and fall back
    # to related job titles.
    queries = [
        f"{skill} in {location}",
        f"{skill} AI engineer in {location}",
        f"{skill} machine learning in {location}",
    ]

    collected = []
    seen = set()

    for query in queries:

        params = {
            "query": query,
            "page": "1",
            "num_pages": "1",
            "country": "in",
            "employment_types": "INTERN,FULLTIME",
        }

        try:

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=8,
            )

            response.raise_for_status()

            data = response.json()

        except requests.RequestException:
            # JSearch can occasionally be slow or unavailable.
            # Do not stop the complete AI analysis because
            # the job-search API failed.
            continue

        for job in data.get("data", []) or []:

            job_id = (
                job.get("job_id")
                or job.get("job_apply_link")
                or job.get("job_title")
            )

            if job_id in seen:
                continue

            seen.add(job_id)

            collected.append(
                {
                    "title": (
                        job.get("job_title")
                        or "Untitled role"
                    ),

                    "company": (
                        job.get("employer_name")
                        or "Company not listed"
                    ),

                    "location": ", ".join(
                        x
                        for x in [
                            job.get("job_city"),
                            job.get("job_state"),
                            job.get("job_country"),
                        ]
                        if x
                    ) or location,

                    "type": (
                        job.get("job_employment_type")
                        or "Not specified"
                    ),

                    "posted": (
                        job.get(
                            "job_posted_at_datetime_utc"
                        )
                        or job.get(
                            "job_posted_at"
                        )
                        or ""
                    ),

                    "apply_link": (
                        job.get("job_apply_link")
                        or job.get("job_google_link")
                        or ""
                    ),

                    "remote": bool(
                        job.get("job_is_remote")
                    ),
                }
            )

        # Stop once we have enough jobs.
        if len(collected) >= 8:
            break

    return collected[:8]


# ---------------------------------------------------------
# LangChain job-search tool
# ---------------------------------------------------------

@tool
def search_jobs(skill: str, location: str) -> list:
    """
    Search current job listings requiring a skill
    in a location using JSearch on RapidAPI.
    """

    return search_jobs_impl(
        skill,
        location
    )


# ---------------------------------------------------------
# Build AI agent
# ---------------------------------------------------------

def build_agent():

    gemini_key = env("GEMINI_API_KEY")

    tavily_key = env("TAVILY_API_KEY")

    # Gemini model
    model = init_chat_model(
        model="google_genai:gemini-2.5-flash",
        api_key=gemini_key,
    )

    # Tavily research tool
    skill_demand_tool = TavilySearch(
        max_results=5,
        topic="general",
        search_depth="advanced",
        tavily_api_key=tavily_key,
    )

    # Agent instructions
    system_prompt = """
You are a Skill-to-Career Mapping assistant for students
and early-career professionals.

Your job is to research a user's career or job query,
explain current skill demand and career trends, and find
relevant job opportunities.

You have two tools:

1. skill_demand_tool:
   Research industry demand, career trends, salary
   information and relevant skills.

2. search_jobs:
   Search current job listings for a skill and location.

For every user request:

- Use the research tool to understand the career.
- Use the job search tool when relevant.
- If the query is broad, identify the most relevant
  skill terms before searching jobs.
- Prefer concrete and useful information over generic advice.

Return a concise, structured answer.

Use plain text headings and bullet points.

Include:

- Career overview
- Current industry demand
- Important technical skills
- Important soft skills
- Suggested learning roadmap
- Relevant job opportunities
- Job title
- Company
- Location
- Employment type
- Application link when available

Do not invent jobs, salaries, sources, or links.

If no jobs are found, clearly say so and suggest
narrower search terms.

If the job-search service is temporarily unavailable,
still provide the career research and skill roadmap.
"""


    return create_agent(
        model=model,
        tools=[
            skill_demand_tool,
            search_jobs
        ],
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/api")
def health():

    return {
        "status": "ok",
        "service": "Skill-to-Career Mapping Agent"
    }


# ---------------------------------------------------------
# Main analysis endpoint
# ---------------------------------------------------------

@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):

    try:

        agent = build_agent()

        prompt = (
            f"User query: {payload.query}\n"
            f"Target location for job openings: "
            f"{payload.location}\n\n"
            "Research the skill/career demand and return "
            "relevant current job opportunities."
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        messages = result.get(
            "messages",
            []
        )

        final = (
            messages[-1].content
            if messages
            else "No response generated."
        )

        # Gemini/LangChain may sometimes return
        # structured content instead of a simple string.
        if isinstance(final, list):

            final = "\n".join(
                item.get("text", "")
                for item in final
                if (
                    isinstance(item, dict)
                    and item.get("text")
                )
            )

        return {
            "answer": str(final),
            "query": payload.query,
            "location": payload.location
        }

    except HTTPException:
        raise

        except Exception as exc:
        print(f"ANALYSIS ERROR: {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}"
        ) from exc
