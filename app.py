from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Reuse the agent API already implemented in api/index.py.
from api.index import app

# Serve the complete frontend from /public on the same Vercel deployment.
# API routes are registered before this mount, so /api/* continues to work.
app.mount("/", StaticFiles(directory="public", html=True), name="frontend")
