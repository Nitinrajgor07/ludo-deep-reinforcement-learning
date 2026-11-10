"""HTML page routing for PhD Ludo DRL demonstration web application.

Renders modern Light Theme Glassmorphism pages with Jinja2 templates.
Zero Streamlit Dependency.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional
import urllib.parse
from fastapi import APIRouter, Form, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from web.services.data_service import data_service, get_system_telemetry
from web.utils.auth import (
    authenticate_user,
    check_user_exists,
    create_session,
    delete_session,
    get_user_by_identifier,
    get_user_by_session,
    get_user_match_history,
    get_user_profile,
    register_user,
)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(include_in_schema=False)


def _get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Retrieve user dictionary from cookie session if present."""
    token = request.cookies.get("session_token")
    return get_user_by_session(token)


@router.get("/")
def index_redirect():
    """Redirect root path to /dashboard (which gates unauthenticated users to /login)."""
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/dashboard")
def dashboard_page(request: Request):
    """Render executive research dashboard for authenticated users."""
    current_user = _get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login?next=/dashboard", status_code=302)

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
            "current_user": current_user,
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
    current_user = _get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login?next=/results", status_code=302)

    specs = data_service.get_model_specs()
    checkpoints = data_service.scan_checkpoints()
    pilot_df = data_service.load_table_dataframe("pilot_summary")
    training_df = data_service.load_table_dataframe("table_training_results")
    eval_df = data_service.load_table_dataframe("table_evaluation_results")
    multimode_df = data_service.load_table_dataframe("table_multimode_comparison")
    robustness_df = data_service.load_table_dataframe("robustness_summary")
    figures = data_service.get_figures_manifest()
    tables = data_service.get_tables_manifest()

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "request": request,
            "current_user": current_user,
            "active_page": "results",
            "page_title": "Unified Results Hub",
            "specs": specs,
            "checkpoints": checkpoints,
            "pilot_df": pilot_df,
            "training_df": training_df,
            "eval_df": eval_df,
            "multimode_df": multimode_df,
            "robustness_df": robustness_df,
            "figures": figures,
            "tables": tables,
        },
    )


@router.get("/models")
def models_page(request: Request):
    """Render 4 Deep Reinforcement Learning models overview and checkpoints."""
    current_user = _get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login?next=/models", status_code=302)

    specs = data_service.get_model_specs()
    checkpoints = data_service.scan_checkpoints()
    figures = data_service.get_figures_manifest()

    return templates.TemplateResponse(
        request=request,
        name="models.html",
        context={
            "request": request,
            "current_user": current_user,
            "active_page": "models",
            "page_title": "RL Models & Architectures",
            "specs": specs,
            "checkpoints": checkpoints,
            "figures_count": len(figures),
        },
    )


@router.get("/ludo")
def ludo_page(request: Request):
    """Render interactive Ludo AI demonstration match arena."""
    current_user = _get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login?next=/ludo", status_code=302)

    checkpoints = data_service.scan_checkpoints()

    return templates.TemplateResponse(
        request=request,
        name="ludo.html",
        context={
            "request": request,
            "current_user": current_user,
            "active_page": "ludo",
            "page_title": "Interactive Ludo Arena",
            "checkpoints": checkpoints,
        },
    )


@router.get("/login")
def login_page(
    request: Request,
    username: Optional[str] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    next: Optional[str] = None,
):
    """Render login form or redirect if already authenticated."""
    current_user = _get_current_user(request)
    if current_user:
        target = next if (next and next.startswith("/") and not next.startswith("//")) else "/dashboard"
        return RedirectResponse(url=target, status_code=302)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "request": request,
            "current_user": None,
            "active_page": "login",
            "page_title": "Sign In",
            "username": username or "",
            "message": message,
            "error": error,
            "next": next or "",
        },
    )


@router.post("/login")
def handle_login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    next: Optional[str] = Form(None),
):
    """Authenticate credentials based on database existence."""
    clean_username = username.strip().lower()

    # If the user does not exist in database, inform them to register
    if not check_user_exists(clean_username):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "request": request,
                "current_user": None,
                "active_page": "login",
                "page_title": "Sign In",
                "username": username,
                "error": f"No account found for '{username}' in the database. Please register to create an account.",
                "next": next or "",
            },
            status_code=400,
        )

    user = authenticate_user(clean_username, password)
    if not user:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "request": request,
                "current_user": None,
                "active_page": "login",
                "page_title": "Sign In",
                "username": username,
                "error": "Invalid username or password. Please try again.",
                "next": next or "",
            },
            status_code=400,
        )

    token = create_session(user["id"])
    target = next if (next and next.startswith("/") and not next.startswith("//")) else "/dashboard"
    redirect_resp = RedirectResponse(url=target, status_code=303)
    redirect_resp.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=86400 * 7,  # 7 days aligned with session max age
        samesite="lax",
    )
    return redirect_resp


@router.get("/register")
def register_page(
    request: Request,
    username: Optional[str] = None,
    error: Optional[str] = None,
    message: Optional[str] = None,
):
    """Render registration form."""
    if _get_current_user(request):
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={
            "request": request,
            "current_user": None,
            "active_page": "register",
            "page_title": "Create Account",
            "username": username or "",
            "error": error,
            "message": message,
        },
    )


@router.post("/register")
def handle_register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    email: Optional[str] = Form(None),
    full_name: Optional[str] = Form(None),
):
    """Process user registration into SQLite, checking if already existing."""
    clean_username = username.strip().lower()

    # If the user already exists in database, reject with clear message and status 400
    if check_user_exists(clean_username):
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "current_user": None,
                "active_page": "register",
                "page_title": "Create Account",
                "username": username,
                "email": email or "",
                "full_name": full_name or "",
                "error": f"Username '{username}' is already taken. An account already exists in the database. Please sign in.",
            },
            status_code=400,
        )

    try:
        register_user(
            username=clean_username,
            password=password,
            email=email if email and email.strip() else None,
            full_name=full_name if full_name and full_name.strip() else None,
        )
        msg = urllib.parse.quote_plus("Account created successfully! Please sign in with your password.")
        return RedirectResponse(
            url=f"/login?username={urllib.parse.quote_plus(clean_username)}&message={msg}",
            status_code=303,
        )
    except ValueError as exc:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "current_user": None,
                "active_page": "register",
                "page_title": "Create Account",
                "username": username,
                "email": email or "",
                "full_name": full_name or "",
                "error": str(exc),
            },
            status_code=400,
        )


@router.get("/profile")
def profile_page(request: Request):
    """Render player profile, statistics, and match history for authenticated user."""
    current_user = _get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login?next=/profile", status_code=302)

    profile = get_user_profile(current_user["id"])
    history = get_user_match_history(current_user["id"], limit=30)

    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "request": request,
            "current_user": current_user,
            "active_page": "profile",
            "page_title": "Player Profile",
            "profile": profile,
            "history": history,
        },
    )


@router.get("/logout")
def logout_page(request: Request):
    """Clear session cookie and invalidate SQLite session."""
    token = request.cookies.get("session_token")
    if token:
        delete_session(token)
    resp = RedirectResponse(url="/dashboard", status_code=302)
    resp.delete_cookie("session_token")
    return resp


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
