import os
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
# JSearch job search
# ---------------------------------------------------------

def search_jobs_impl(skill: str, location: str):

    rapid_api_key = env("RAPID_API_KEY")

    url = "https://jsearch.p.rapidapi.com/search-v2"

    headers = {
        "x-rapidapi-key": rapid_api_key,
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
    }

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

        except requests.RequestException as exc:

            print(
                f"JSEARCH REQUEST ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            continue

        except ValueError as exc:

            print(
                f"JSEARCH JSON ERROR: {exc}"
            )

            continue


        # -------------------------------------------------
        # Make sure the API response is a dictionary
        # -------------------------------------------------

        if not isinstance(data, dict):

            print(
                "JSEARCH RESPONSE WAS NOT A DICTIONARY: "
                f"{type(data).__name__}"
            )

            continue


        jobs = data.get("data", [])


        # -------------------------------------------------
        # Make sure jobs is a list
        # -------------------------------------------------

        if not isinstance(jobs, list):

            print(
                "JSEARCH DATA WAS NOT A LIST: "
                f"{type(jobs).__name__}"
            )

            continue


        # -------------------------------------------------
        # Process individual jobs
        # -------------------------------------------------

        for job in jobs:

            # Ignore malformed/string entries
            if not isinstance(job, dict):
                continue

            job_id = (
                job.get("job_id")
                or job.get("job_apply_link")
                or job.get("job_title")
            )

            if not job_id:
                continue

            if job_id in seen:
                continue

            seen.add(job_id)


            job_location_parts = [
                job.get("job_city"),
                job.get("job_state"),
                job.get("job_country"),
            ]

            job_location_parts = [
                str(x)
                for x in job_location_parts
                if x
            ]

            job_location = (
                ", ".join(job_location_parts)
                if job_location_parts
                else location
            )


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

                    "location": job_location,

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


    # Gemini
    model = init_chat_model(
        model="google_genai:gemini-3.5-flash",
        api_key=gemini_key,
    )


    # Tavily
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

- Research the career and current industry demand.
- Identify important technical skills.
- Identify important soft skills.
- Provide a practical learning roadmap.
- Search for relevant job opportunities.
- Use the user's requested location for job searches.

Return a concise, structured answer.

Use plain text headings and bullet points.

Include:

Career Overview

Industry Demand

Technical Skills

Soft Skills

Learning Roadmap

Job Opportunities

For every available job include:

- Job title
- Company
- Location
- Employment type
- Application link

Do not invent jobs, salaries, sources, or links.

If no jobs are found, clearly say so.

If the job-search service is temporarily unavailable,
still provide the career research, industry demand,
skills and learning roadmap.
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
            "Research the career demand, identify the "
            "important skills and provide relevant "
            "current job opportunities."
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


        # -------------------------------------------------
        # Safely extract messages
        # -------------------------------------------------

        if not isinstance(result, dict):

            raise RuntimeError(
                "Agent returned an unexpected response."
            )


        messages = result.get(
            "messages",
            []
        )


        if not isinstance(messages, list):

            raise RuntimeError(
                "Agent messages had an unexpected format."
            )


        if not messages:

            final = "No response generated."

        else:

            last_message = messages[-1]


            # ---------------------------------------------
            # Extract content safely
            # ---------------------------------------------

            if hasattr(last_message, "content"):

                final = last_message.content

            elif isinstance(last_message, dict):

                final = last_message.get(
                    "content",
                    ""
                )

            else:

                final = str(last_message)


        # -------------------------------------------------
        # Handle different content formats
        # -------------------------------------------------

        if isinstance(final, str):

            final_text = final


        elif isinstance(final, list):

            parts = []

            for item in final:

                if isinstance(item, str):

                    parts.append(item)

                elif isinstance(item, dict):

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:
                        parts.append(str(text))

                else:

                    parts.append(str(item))


            final_text = "\n".join(parts)


        elif isinstance(final, dict):

            if final.get("text"):

                final_text = str(
                    final.get("text")
                )

            elif final.get("content"):

                final_text = str(
                    final.get("content")
                )

            else:

                final_text = str(final)


        else:

            final_text = str(final)


        if not final_text.strip():

            final_text = "No response generated."


        return {
            "answer": final_text,
            "query": payload.query,
            "location": payload.location
        }


    except HTTPException:

        raise


    except Exception as exc:

        print(
            "ANALYSIS ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(exc).__name__}: {exc}"
            )
        ) from exc
