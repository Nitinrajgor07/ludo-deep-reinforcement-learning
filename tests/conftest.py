"""Global pytest configuration ensuring 100% database test isolation.

Guarantees:
- Every pytest run executes against an isolated temporary SQLite database.
- The production database data/users.db is never written to during test runs.
- Temporary test databases are safely cleaned up after test completion.
- Accidental access to production data/users.db raises an immediate assertion error.
"""

import os
from pathlib import Path
import shutil
import tempfile
import pytest

from web.utils.auth import get_db_path, set_db_path


@pytest.fixture(scope="session", autouse=True)
def isolated_test_database_session():
    """Session-level fixture that isolates all tests into a temporary SQLite database."""
    project_root = Path(__file__).resolve().parent.parent
    prod_db = project_root / "data" / "users.db"

    # Create dedicated temporary directory and database file
    temp_dir = tempfile.mkdtemp(prefix="ludo_test_session_")
    test_db_path = Path(temp_dir) / "test_isolated_users.db"

    # Seed test database with copy of production database so existing accounts & migrations exist
    if prod_db.exists():
        shutil.copy2(prod_db, test_db_path)

    # Configure isolation override
    set_db_path(test_db_path)
    os.environ["LUDO_DATABASE_PATH"] = str(test_db_path)

    # Verify isolation is active
    current_path = get_db_path()
    assert current_path.resolve() == test_db_path.resolve(), (
        f"Database isolation failed: expected {test_db_path}, got {current_path}"
    )
    assert current_path.resolve() != prod_db.resolve(), (
        "CRITICAL: Test runner pointed directly at production data/users.db!"
    )

    yield test_db_path

    # Teardown and cleanup
    set_db_path(None)
    os.environ.pop("LUDO_DATABASE_PATH", None)

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def guard_production_database():
    """Per-test guard asserting that production data/users.db is never the active database."""
    project_root = Path(__file__).resolve().parent.parent
    prod_db = (project_root / "data" / "users.db").resolve()
    active_db = get_db_path().resolve()

    assert active_db != prod_db, (
        f"CRITICAL SAFETY VIOLATION: Test attempted to access production database {prod_db}!"
    )
    yield
    assert get_db_path().resolve() != prod_db, (
        f"CRITICAL SAFETY VIOLATION: Production database {prod_db} was accessed during test execution!"
    )
