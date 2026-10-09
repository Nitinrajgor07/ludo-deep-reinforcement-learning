"""Main FastAPI web application for PhD Ludo DRL Research Platform.

Replaces Streamlit with a clean, high-performance ASGI application.
Strict Constraint: 100% Light Theme Only. Zero Dark Elements.
Zero Streamlit Dependency.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from web.routes.pages import router as pages_router
from web.routes.api import router as api_router
from web.services.data_service import get_project_root

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = get_project_root()
STATIC_DIR = BASE_DIR / "static"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Ensure required directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "assets").mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Create FastAPI application instance
app = FastAPI(
    title="PhD Ludo DRL Platform",
    description="Deep Reinforcement Learning in Stochastic Multi-Agent Ludo (Light Glassmorphic Web App)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount static asset directories
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Include page & REST API routes
app.include_router(pages_router)
app.include_router(api_router)


# Alias routes for convenience
@app.get("/visualization", include_in_schema=False)
def visualization_alias():
    """Redirect singular /visualization to plural /visualizations."""
    return RedirectResponse(url="/visualizations", status_code=302)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve SVG logo as favicon."""
    from fastapi.responses import FileResponse
    logo_path = STATIC_DIR / "assets" / "ludo_logo.svg"
    if logo_path.exists():
        return FileResponse(logo_path, media_type="image/svg+xml")
    return RedirectResponse(url="/static/assets/ludo_logo.svg")


def main():
    """Run localhost web server."""
    uvicorn.run(
        "web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
