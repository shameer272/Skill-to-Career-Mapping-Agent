# Skill-to-Career Mapping Agent

A Vercel-ready web version of the Skill-to-Career Mapping Agent from the original Google Colab notebook.

## What it does

- Uses Gemini 2.5 Flash through LangChain
- Researches skill demand and career trends with Tavily
- Searches current job listings through JSearch on RapidAPI
- Presents the result through a responsive web interface

## Project structure

```text
Skill-to-Career-Mapping-Agent/
├── api/
│   └── index.py
├── public/
│   └── index.html
├── Skill_Map_Agent.ipynb
├── requirements.txt
├── vercel.json
├── .env.example
├── .gitignore
└── README.md
```

## Vercel environment variables

Add these in Vercel Project Settings → Environment Variables:

- `GEMINI_API_KEY`
- `TAVILY_API_KEY`
- `RAPID_API_KEY`

Do not commit real API keys to GitHub.

## Deploy

Import the GitHub repository into Vercel and deploy. The static frontend is served from `public/index.html`, while the Python API is exposed at `/api/analyze`.
