"""Final functional verification test suite for the Authentication Portal.

Tests all 9 items requested:
1. Auto-Detect for existing and new usernames/emails via /api/auth/check.
2. Sign In with valid and invalid credentials.
3. Register with valid and invalid inputs.
4. Password visibility toggle structure and contract.
5. Understandable validation error messages.
6. Individual account registration and login (team decoupled from user profile).
7. Safe redirection after authentication and open-redirect prevention.
8. Session handling, SQLite session persistence, and logout invalidation.
9. Desktop and mobile responsiveness (meta viewport, responsive cards, fluid controls).
"""

from pathlib import Path
import re
import pytest
from starlette.testclient import TestClient

from web.app import app
from web.utils.auth import (
    create_session,
    delete_session,
    get_db,
    get_user_by_session,
    register_user,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_1_auto_detect_existing_and_new_usernames(client):
    """Verify Auto-Detect API for existing accounts, non-existing accounts, and emails."""
    # 1. Register a test user
    test_user = "autodetect_user"
    test_email = "autodetect@phd.edu"
    register_user(test_user, "Password123!", email=test_email, full_name="Auto Detect User")

    # 2. Existing user by username (GET)
    res_exist = client.get(f"/api/auth/check?identifier={test_user}")
    assert res_exist.status_code == 200
    data_exist = res_exist.json()
    assert data_exist["exists"] is True
    assert data_exist["username"] == test_user
    assert data_exist["full_name"] == "Auto Detect User"

    # 3. Existing user by email (GET)
    res_email = client.get(f"/api/auth/check?identifier={test_email}")
    assert res_email.status_code == 200
    data_email = res_email.json()
    assert data_email["exists"] is True
    assert data_email["username"] == test_user

    # 4. Non-existing user (GET)
    res_none = client.get("/api/auth/check?identifier=brand_new_researcher_xyz")
    assert res_none.status_code == 200
    data_none = res_none.json()
    assert data_none["exists"] is False
    assert data_none["username"] == "brand_new_researcher_xyz"

    # 5. POST check via JSON body
    res_post = client.post("/api/auth/check", json={"identifier": test_user})
    assert res_post.status_code == 200
    assert res_post.json()["exists"] is True


def test_2_sign_in_valid_and_invalid_credentials(client):
    """Verify Sign In with valid and invalid credentials."""
    user = "signin_test_user"
    pwd = "ValidPassword123!"
    register_user(user, pwd, full_name="Sign In User")

    # 1. Valid credentials -> 303 Redirect with session cookie
    good_login = client.post(
        "/login",
        data={"username": user, "password": pwd},
        follow_redirects=False,
    )
    assert good_login.status_code == 303
    assert "session_token" in good_login.cookies
    cookie_val = good_login.cookies["session_token"]
    assert len(cookie_val) > 20

    # 2. Invalid password -> 400 with clear message
    bad_login = client.post(
        "/login",
        data={"username": user, "password": "WrongPassword999!"},
    )
    assert bad_login.status_code == 400
    assert "Invalid username or password" in bad_login.text

    # 3. Non-existent username -> 400 with guidance to register
    ghost_login = client.post(
        "/login",
        data={"username": "ghost_nonexistent_user", "password": "SomePassword123!"},
    )
    assert ghost_login.status_code == 400
    assert "No account found" in ghost_login.text
    assert "register" in ghost_login.text.lower()


def test_3_register_valid_and_invalid_inputs(client):
    """Verify registration with valid inputs and various invalid validation edge cases."""
    # 1. Valid registration -> 303 Redirect to login with success message
    valid_user = "valid_reg_user_1"
    res_valid = client.post(
        "/register",
        data={
            "username": valid_user,
            "password": "SecretPassword123!",
            "email": "valid1@university.edu",
            "full_name": "Valid Researcher",
        },
        follow_redirects=False,
    )
    assert res_valid.status_code == 303
    assert "/login" in res_valid.headers["location"]
    assert valid_user in res_valid.headers["location"]

    # 2. Duplicate username -> 400
    res_dup_u = client.post(
        "/register",
        data={"username": valid_user, "password": "AnotherPassword123!"},
    )
    assert res_dup_u.status_code == 400
    assert "already taken" in res_dup_u.text or "already exists" in res_dup_u.text

    # 3. Duplicate email -> 400
    res_dup_e = client.post(
        "/register",
        data={
            "username": "different_user_name",
            "password": "Password123!",
            "email": "valid1@university.edu",
        },
    )
    assert res_dup_e.status_code == 400
    assert "already registered" in res_dup_e.text

    # 4. Short username (<3 chars) -> 400
    res_short_u = client.post(
        "/register",
        data={"username": "ab", "password": "Password123!"},
    )
    assert res_short_u.status_code == 400
    assert "between 3 and 50 characters" in res_short_u.text

    # 5. Invalid characters in username -> 400
    res_bad_chars = client.post(
        "/register",
        data={"username": "invalid user with spaces", "password": "Password123!"},
    )
    assert res_bad_chars.status_code == 400
    assert "invalid characters" in res_bad_chars.text.lower()

    # 6. Short password (<4 chars) -> 400
    res_short_pwd = client.post(
        "/register",
        data={"username": "valid_user_2", "password": "123"},
    )
    assert res_short_pwd.status_code == 400
    assert "between 4 and 128 characters" in res_short_pwd.text

    # 7. Invalid email format -> 400
    res_bad_email = client.post(
        "/register",
        data={"username": "valid_user_3", "password": "Password123!", "email": "not-an-email"},
    )
    assert res_bad_email.status_code == 400
    assert "Invalid email address format" in res_bad_email.text


def test_4_password_visibility_toggle_contracts(client):
    """Verify password visibility toggle markup and JS functions in login and register templates."""
    res_login = client.get("/login")
    assert res_login.status_code == 200
    html_login = res_login.text

    # Check function exists
    assert "togglePasswordVisibility" in html_login

    # Check eye and eye-off icon elements exist
    assert "eye-icon" in html_login
    assert "eye-off-icon" in html_login

    # Check password input fields have companion toggle buttons
    assert 'onclick="togglePasswordVisibility(\'auto-login-password\', this)"' in html_login
    assert 'onclick="togglePasswordVisibility(\'auto-reg-password\', this)"' in html_login
    assert 'onclick="togglePasswordVisibility(\'direct-login-password\', this)"' in html_login
    assert 'onclick="togglePasswordVisibility(\'direct-reg-password\', this)"' in html_login

    res_reg = client.get("/register")
    assert res_reg.status_code == 200
    html_reg = res_reg.text
    assert 'onclick="togglePasswordVisibility(\'password\', this)"' in html_reg


def test_5_validation_messages_clarity(client):
    """Verify validation messages provide clear guidance to the user."""
    # Test registration failure messages
    res_short = client.post("/register", data={"username": "x", "password": "foo"})
    assert "Username must be between 3 and 50 characters" in res_short.text

    res_chars = client.post("/register", data={"username": "bad!name@#", "password": "foobar"})
    assert "Use letters, numbers, hyphens, underscores, or periods" in res_chars.text


def test_6_individual_account_registration_and_login(client):
    """Verify registration creates an individual player account without hardcoding team membership."""
    user = "individual_player_1"
    reg_res = client.post(
        "/register",
        data={
            "username": user,
            "password": "Password123!",
            "full_name": "Solo Player",
            "email": "solo@ludo.ai",
        },
        follow_redirects=False,
    )
    assert reg_res.status_code == 303

    # Check in SQLite database that user table contains no team_id column
    conn = get_db()
    c = conn.cursor()
    c.execute("PRAGMA table_info(users)")
    columns = [col["name"] for col in c.fetchall()]
    assert "team_id" not in columns
    assert "username" in columns
    assert "password_hash" in columns
    assert "salt" in columns

    # Verify user record
    c.execute("SELECT username, full_name, rating FROM users WHERE username = ?", (user,))
    row = c.fetchone()
    assert row is not None
    assert row["username"] == user
    assert row["rating"] == 1000
    conn.close()


def test_7_safe_redirection_and_open_redirect_guards(client):
    """Verify redirect to ?next path on login, and open-redirect protection."""
    user = "redirect_user"
    pwd = "Password123!"
    register_user(user, pwd, full_name="Redirect Tester")

    # 1. Safe relative next: /results
    res_results = client.post(
        "/login",
        data={"username": user, "password": pwd, "next": "/results"},
        follow_redirects=False,
    )
    assert res_results.status_code == 303
    assert res_results.headers["location"] == "/results"

    # 2. Safe relative next: /ludo
    res_ludo = client.post(
        "/login",
        data={"username": user, "password": pwd, "next": "/ludo"},
        follow_redirects=False,
    )
    assert res_ludo.status_code == 303
    assert res_ludo.headers["location"] == "/ludo"

    # 3. Unsafe external / protocol-relative URL: //evil.com -> Defaults to /dashboard
    res_unsafe = client.post(
        "/login",
        data={"username": user, "password": pwd, "next": "//evil.com"},
        follow_redirects=False,
    )
    assert res_unsafe.status_code == 303
    assert res_unsafe.headers["location"] == "/dashboard"

    # 4. Default without next -> /dashboard
    res_default = client.post(
        "/login",
        data={"username": user, "password": pwd},
        follow_redirects=False,
    )
    assert res_default.status_code == 303
    assert res_default.headers["location"] == "/dashboard"


def test_8_session_handling_and_logout(client):
    """Verify session creation in SQLite, cookie parameters, and logout invalidation."""
    user = "session_user"
    pwd = "Password123!"
    reg = register_user(user, pwd)
    user_id = reg["id"]

    # 1. Login sets session cookie
    login_res = client.post(
        "/login",
        data={"username": user, "password": pwd},
        follow_redirects=False,
    )
    token = login_res.cookies.get("session_token")
    assert token is not None

    # 2. Session exists in SQLite database
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT user_id, created_at FROM sessions WHERE token = ?", (token,))
    sess_row = c.fetchone()
    assert sess_row is not None
    assert sess_row["user_id"] == user_id
    conn.close()

    # 3. Access protected dashboard with session cookie
    dash_res = client.get("/dashboard", cookies={"session_token": token})
    assert dash_res.status_code == 200
    assert "session_user" in dash_res.text or "Executive Dashboard" in dash_res.text

    # 4. Logout via GET /logout
    logout_res = client.get("/logout", cookies={"session_token": token}, follow_redirects=False)
    assert logout_res.status_code == 302

    # 5. Session token is removed from SQLite
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT 1 FROM sessions WHERE token = ?", (token,))
    assert c.fetchone() is None
    conn.close()

    # 6. Re-using deleted token fails authentication
    after_res = client.get("/profile", cookies={"session_token": token}, follow_redirects=False)
    assert after_res.status_code == 302
    assert "/login" in after_res.headers["location"]


def test_9_desktop_and_mobile_responsiveness_contract(client):
    """Verify viewport meta tag, responsive container constraints, and fluid layout."""
    res = client.get("/login")
    assert res.status_code == 200
    html = res.text

    # 1. Viewport meta tag for mobile scaling
    assert '<meta name="viewport" content="width=device-width, initial-scale=1.0">' in html

    # 2. Responsive card constraints (fluid width up to max-width: 480px)
    assert "max-width: 480px" in html
    assert "width: 100%" in html

    # 3. Form input responsiveness
    assert "width: 100%" in html
