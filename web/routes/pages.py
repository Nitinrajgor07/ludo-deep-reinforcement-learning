"""HTML page routing for PhD Ludo DRL demonstration web application.

Renders modern Light Theme Glassmorphism pages with Jinja2 templates.
Zero Streamlit Dependency.
"""

from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from web.services.data_service import data_service, get_system_telemetry

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(include_in_schema=False)


import sys

@router.get("/")
def index_redirect():
    """Redirect root path to /dashboard."""
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/dashboard")
def dashboard_page(request: Request):
    """Render executive research dashboard."""
    specs = data_service.get_model_specs()
    checkpoints = data_service.scan_checkpoints()
    telemetry = get_system_telemetry()
    pilot_df = data_service.load_table_dataframe("pilot_summary")
    training_df = data_service.load_table_dataframe("table_training_results")
    eval_df = data_service.load_table_dataframe("table_evaluation_results")
    figures = data_service.get_figures_manifest()
    tables = data_service.get_tables_manifest()
    test_status = data_service.get_test_status()
    activities = data_service.get_activity_feed()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "active_page": "dashboard",
            "page_title": "Executive Dashboard",
            "specs": specs,
            "checkpoints": checkpoints,
            "telemetry": telemetry,
            "pilot_df": pilot_df,
            "training_df": training_df,
            "eval_df": eval_df,
            "figures_count": len(figures),
            "tables_count": len(tables),
            "test_status": test_status,
            "activities": activities,
            "python_version": sys.version.split()[0],
        },
    )


@router.get("/results")
def results_page(request: Request):
    """Render consolidated compact Results Hub aggregating all model comparison results."""
    specs = data_service.get_model_specs()
    checkpoints = data_service.scan_checkpoints()
    pilot_df = data_service.load_table_dataframe("pilot_summary")
    training_df = data_service.load_table_dataframe("table_training_results")
    eval_df = data_service.load_table_dataframe("table_evaluation_results")
    robustness_df = data_service.load_table_dataframe("robustness_summary")
    figures = data_service.get_figures_manifest()
    tables = data_service.get_tables_manifest()

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "request": request,
            "active_page": "results",
            "page_title": "Unified Results Hub",
            "specs": specs,
            "checkpoints": checkpoints,
            "pilot_df": pilot_df,
            "training_df": training_df,
            "eval_df": eval_df,
            "robustness_df": robustness_df,
            "figures": figures,
            "tables": tables,
        },
    )



@router.get("/models")
def models_page(request: Request):
    """Render 4 Deep Reinforcement Learning models overview and checkpoints."""
    specs = data_service.get_model_specs()
    checkpoints = data_service.scan_checkpoints()
    figures = data_service.get_figures_manifest()

    return templates.TemplateResponse(
        request=request,
        name="models.html",
        context={
            "request": request,
            "active_page": "models",
            "page_title": "RL Models & Architectures",
            "specs": specs,
            "checkpoints": checkpoints,
            "figures_count": len(figures),
        },
    )


@router.get("/training")
@router.get("/evaluation")
@router.get("/statistics")
@router.get("/visualizations")
def legacy_results_redirect():
    """Redirect legacy redundant result routes cleanly to the Unified Results Hub."""
    return RedirectResponse(url="/results", status_code=302)


@router.get("/about")
def legacy_about_redirect():
    """Redirect legacy about route cleanly to executive Dashboard."""
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/ludo")
def ludo_page(request: Request):
    """Render interactive Ludo AI demonstration match arena."""
    checkpoints = data_service.scan_checkpoints()

    return templates.TemplateResponse(
        request=request,
        name="ludo.html",
        context={
            "request": request,
            "active_page": "ludo",
            "page_title": "Interactive Ludo Arena",
            "checkpoints": checkpoints,
        },
    )
