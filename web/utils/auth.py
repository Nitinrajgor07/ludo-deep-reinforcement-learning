"""SQLite user database and authentication management for PhD Ludo DRL Platform.

Provides lightweight user registration, password hashing (PBKDF2-HMAC-SHA256 with salt),
session tokens, and role-based access control.
"""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


# Configurable/overridable database path for reliable test isolation
_OVERRIDE_DB_PATH: Optional[Path] = None


def get_db_path() -> Path:
    """Return SQLite database file path, honoring test overrides and environment variables."""
    global _OVERRIDE_DB_PATH
    if _OVERRIDE_DB_PATH is not None:
        return _OVERRIDE_DB_PATH

    env_path = os.environ.get("LUDO_DATABASE_PATH")
    if env_path:
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "users.db"


def set_db_path(path: Optional[Path]) -> None:
    """Explicitly override database path for tests or reset to default."""
    global _OVERRIDE_DB_PATH
    _OVERRIDE_DB_PATH = Path(path) if path is not None else None


def get_db() -> sqlite3.Connection:
    """Return a configured sqlite3 connection with Row factory."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize SQLite database tables and apply safe repeatable migrations."""
    target_path = Path(db_path) if db_path is not None else get_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT,
                avatar_color TEXT DEFAULT 'rose',
                bio TEXT DEFAULT '',
                rating INTEGER DEFAULT 1000,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS match_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                match_mode TEXT NOT NULL,
                winner_id INTEGER,
                winner_name TEXT,
                winner_controller TEXT,
                total_turns INTEGER NOT NULL,
                p0_type TEXT,
                p1_type TEXT,
                p2_type TEXT,
                p3_type TEXT,
                p0_score REAL,
                p1_score REAL,
                p2_score REAL,
                p3_score REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS match_participants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                user_id INTEGER,
                username TEXT NOT NULL,
                player_seat INTEGER NOT NULL,
                team_id INTEGER,
                controller_type TEXT NOT NULL,
                is_winner BOOLEAN NOT NULL DEFAULT 0,
                tokens_finished INTEGER NOT NULL DEFAULT 0,
                total_captures INTEGER NOT NULL DEFAULT 0,
                final_score REAL NOT NULL DEFAULT 0.0,
                FOREIGN KEY (match_id) REFERENCES match_records(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )

        # Safe, repeatable schema migrations using PRAGMA inspection
        cursor.execute("PRAGMA table_info(users)")
        user_cols = {row[1] for row in cursor.fetchall()}
        if "avatar_color" not in user_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN avatar_color TEXT DEFAULT 'rose'")
        if "bio" not in user_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN bio TEXT DEFAULT ''")
        if "rating" not in user_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN rating INTEGER DEFAULT 1000")

        cursor.execute("PRAGMA table_info(match_records)")
        match_cols = {row[1] for row in cursor.fetchall()}
        if "user_id" not in match_cols:
            cursor.execute("ALTER TABLE match_records ADD COLUMN user_id INTEGER REFERENCES users(id)")
        if "game_mode" not in match_cols:
            cursor.execute("ALTER TABLE match_records ADD COLUMN game_mode TEXT DEFAULT ''")
        if "room_code" not in match_cols:
            cursor.execute("ALTER TABLE match_records ADD COLUMN room_code TEXT DEFAULT ''")
        if "is_online" not in match_cols:
            cursor.execute("ALTER TABLE match_records ADD COLUMN is_online BOOLEAN DEFAULT 0")
        if "team_winner" not in match_cols:
            cursor.execute("ALTER TABLE match_records ADD COLUMN team_winner INTEGER")

        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_match_records_room_code
            ON match_records(room_code)
            WHERE room_code IS NOT NULL AND room_code != '' AND room_code != 'LOCAL-1V1'
            """
        )

        conn.commit()


# Initialize database on module load
init_db()



def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash password using PBKDF2-HMAC-SHA256 with a unique salt."""
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100000,
    )
    return key.hex(), salt


def register_user(
    username: str,
    password: str,
    email: Optional[str] = None,
    full_name: Optional[str] = None,
    avatar_color: Optional[str] = "rose",
    bio: Optional[str] = "",
) -> Dict[str, Any]:
    """Register a new user in the SQLite database with rigorous validation."""
    if not username or not isinstance(username, str):
        raise ValueError("Username cannot be empty.")
    clean_username = username.strip().lower()
    if not (3 <= len(clean_username) <= 50):
        raise ValueError("Username must be between 3 and 50 characters long.")
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", clean_username):
        raise ValueError("Username contains invalid characters. Use letters, numbers, hyphens, underscores, or periods.")

    if not password or not isinstance(password, str):
        raise ValueError("Password cannot be empty.")
    if not (4 <= len(password) <= 128):
        raise ValueError("Password must be between 4 and 128 characters long.")

    clean_email = email.strip().lower() if (email and isinstance(email, str) and email.strip()) else None
    if clean_email:
        if len(clean_email) > 100 or "@" not in clean_email or "." not in clean_email.split("@")[-1]:
            raise ValueError("Invalid email address format.")

    clean_name = (full_name.strip()[:100] if (full_name and full_name.strip()) else clean_username.title())
    avatar_col = (avatar_color or "rose").strip().lower()
    valid_avatars = {"rose", "emerald", "amber", "blue", "indigo", "purple"}
    if avatar_col not in valid_avatars:
        avatar_col = "rose"
    user_bio = (bio or "").strip()[:300]

    pwd_hash, salt = _hash_password(password)
    created_at = datetime.now(timezone.utc).isoformat()
    db_path = get_db_path()

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Explicit duplicate pre-check
            cursor.execute("SELECT 1 FROM users WHERE username = ?", (clean_username,))
            if cursor.fetchone():
                raise ValueError(f"Username '{clean_username}' is already taken.")
            if clean_email:
                cursor.execute("SELECT 1 FROM users WHERE email = ?", (clean_email,))
                if cursor.fetchone():
                    raise ValueError(f"Email '{clean_email}' is already registered.")

            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, salt, full_name, avatar_color, bio, rating, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1000, ?)
                """,
                (clean_username, clean_email, pwd_hash, salt, clean_name, avatar_col, user_bio, created_at),
            )
            user_id = cursor.lastrowid
            conn.commit()
            return {
                "id": user_id,
                "username": clean_username,
                "email": clean_email,
                "full_name": clean_name,
                "avatar_color": avatar_col,
                "bio": user_bio,
                "rating": 1000,
                "created_at": created_at,
            }
    except sqlite3.IntegrityError as e:
        if "users.username" in str(e) or "UNIQUE constraint failed: users.username" in str(e):
            raise ValueError(f"Username '{clean_username}' is already taken.")
        elif "users.email" in str(e) or "UNIQUE constraint failed: users.email" in str(e):
            raise ValueError(f"Email '{clean_email}' is already registered.")
        raise ValueError("User registration failed: User already exists.")


def check_user_exists(identifier: str) -> bool:
    """Check whether a user account exists in the database by username or email."""
    if not identifier:
        return False
    ident = identifier.strip().lower()
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM users WHERE username = ? OR email = ?",
            (ident, ident),
        )
        return cursor.fetchone() is not None


def get_user_by_identifier(identifier: str) -> Optional[Dict[str, Any]]:
    """Retrieve basic user record by username or email."""
    if not identifier:
        return None
    ident = identifier.strip().lower()
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, email, full_name, avatar_color, bio, rating, created_at
            FROM users WHERE username = ? OR email = ?
            """,
            (ident, ident),
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "avatar_color": row["avatar_color"] or "rose",
                "bio": row["bio"] or "",
                "rating": row["rating"] or 1000,
                "created_at": row["created_at"],
            }
        return None


def authenticate_user(username_or_email: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify credentials against SQLite by username or email."""
    ident = username_or_email.strip().lower()
    db_path = get_db_path()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (ident, ident),
        )
        row = cursor.fetchone()
        if not row:
            return None

        expected_hash, _ = _hash_password(password, row["salt"])
        if secrets.compare_digest(expected_hash, row["password_hash"]):
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "avatar_color": row["avatar_color"] if "avatar_color" in row.keys() else "rose",
                "bio": row["bio"] if "bio" in row.keys() else "",
                "rating": row["rating"] if "rating" in row.keys() else 1000,
                "created_at": row["created_at"],
            }
        return None


SESSION_MAX_AGE_SECONDS: int = int(os.environ.get("SESSION_MAX_AGE_SECONDS", 86400 * 7))  # 7 days default


def is_session_expired(created_at_str: Optional[str], max_age_seconds: Optional[int] = None) -> bool:
    """Check if session creation timestamp exceeds maximum allowed age."""
    if not created_at_str:
        return True
    try:
        dt = datetime.fromisoformat(created_at_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age = (now - dt).total_seconds()
        limit = max_age_seconds if max_age_seconds is not None else SESSION_MAX_AGE_SECONDS
        return age > limit
    except Exception:
        return False


def create_session(user_id: Any) -> str:
    """Create a persistent session token for an authenticated user."""
    uid = user_id["id"] if isinstance(user_id, dict) else int(user_id)
    token = secrets.token_hex(32)
    created_at = datetime.now(timezone.utc).isoformat()
    db_path = get_db_path()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, uid, created_at),
        )
        conn.commit()
    return token


def rotate_session(old_token: str) -> Optional[str]:
    """Atomically rotate a session token for session fixation prevention."""
    if not old_token:
        return None
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM sessions WHERE token = ?", (old_token,))
        row = cursor.fetchone()
        if not row:
            return None
        user_id = row[0]
        new_token = secrets.token_hex(32)
        now_ts = datetime.now(timezone.utc).isoformat()
        cursor.execute("DELETE FROM sessions WHERE token = ?", (old_token,))
        cursor.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)", (new_token, user_id, now_ts))
        conn.commit()
        return new_token


def get_user_by_session(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Retrieve user details for a given session token, invalidating expired sessions."""
    if not token:
        return None
    db_path = get_db_path()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.id, u.username, u.email, u.full_name, u.avatar_color, u.bio, u.rating, u.created_at,
                   s.created_at AS session_created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
            """,
            (token,),
        )
        row = cursor.fetchone()
        if row:
            sess_time = row["session_created_at"]
            if is_session_expired(sess_time):
                cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
                conn.commit()
                return None
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "avatar_color": row["avatar_color"] or "rose",
                "bio": row["bio"] or "",
                "rating": row["rating"] or 1000,
                "created_at": row["created_at"],
            }
        return None


def delete_session(token: Optional[str]) -> None:
    """Remove a session token from the database."""
    if not token:
        return
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


# ==============================================================================
# Player Profile & User Statistics Service
# ==============================================================================

def get_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve full user profile along with aggregate match statistics."""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, email, full_name, avatar_color, bio, rating, created_at
            FROM users WHERE id = ?
            """,
            (user_id,),
        )
        user_row = cursor.fetchone()
        if not user_row:
            return None

        user_dict = dict(user_row)
        user_dict["avatar_color"] = user_dict.get("avatar_color") or "rose"
        user_dict["bio"] = user_dict.get("bio") or ""
        user_dict["rating"] = user_dict.get("rating") or 1000

        # Calculate statistics from match_participants and match_records
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_matches,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as total_wins,
                SUM(tokens_finished) as total_tokens_finished,
                SUM(total_captures) as total_captures
            FROM match_participants
            WHERE user_id = ?
            """,
            (user_id,),
        )
        part_stats = cursor.fetchone()
        p_matches = (part_stats["total_matches"] or 0) if part_stats else 0
        p_wins = (part_stats["total_wins"] or 0) if part_stats else 0

        # Also count direct user_id matches from match_records if not in participants
        cursor.execute(
            """
            SELECT COUNT(*) as legacy_matches,
                   SUM(CASE WHEN winner_controller = 'HUMAN' OR winner_name = 'Red' THEN 1 ELSE 0 END) as legacy_wins
            FROM match_records
            WHERE user_id = ? AND id NOT IN (SELECT match_id FROM match_participants WHERE user_id = ?)
            """,
            (user_id, user_id),
        )
        leg_stats = cursor.fetchone()
        leg_matches = (leg_stats["legacy_matches"] or 0) if leg_stats else 0
        leg_wins = (leg_stats["legacy_wins"] or 0) if leg_stats else 0

        total_matches = p_matches + leg_matches
        total_wins = p_wins + leg_wins
        win_rate = round((total_wins / total_matches) * 100, 1) if total_matches > 0 else 0.0

        # Performance by game mode
        mode_counts = {
            "1v1_ai": {"name": "Human vs AI (1v1)", "matches": 0, "wins": 0, "win_rate": 0.0},
            "1v1_human": {"name": "Human vs Human (1v1)", "matches": 0, "wins": 0, "win_rate": 0.0},
            "2v2_team": {"name": "Team vs Team (2v2)", "matches": 0, "wins": 0, "win_rate": 0.0},
            "4p_ffa": {"name": "4-Player Arena (FFA)", "matches": 0, "wins": 0, "win_rate": 0.0},
        }

        cursor.execute(
            """
            SELECT m.game_mode, m.match_mode, p.is_winner
            FROM match_participants p
            JOIN match_records m ON p.match_id = m.id
            WHERE p.user_id = ?
            """,
            (user_id,),
        )
        for row in cursor.fetchall():
            raw_mode = (row["game_mode"] or row["match_mode"] or "").lower()
            if "ai" in raw_mode or "1v1_ai" in raw_mode or "1 human" in raw_mode:
                mkey = "1v1_ai"
            elif "human" in raw_mode and "1v1" in raw_mode:
                mkey = "1v1_human"
            elif "team" in raw_mode or "2v2" in raw_mode:
                mkey = "2v2_team"
            else:
                mkey = "4p_ffa"
            mode_counts[mkey]["matches"] += 1
            if row["is_winner"]:
                mode_counts[mkey]["wins"] += 1

        for mkey, mdata in mode_counts.items():
            m_total = mdata["matches"]
            m_win = mdata["wins"]
            mdata["win_rate"] = round((m_win / m_total) * 100, 1) if m_total > 0 else 0.0

        user_dict["stats"] = {
            "total_matches": total_matches,
            "total_wins": total_wins,
            "total_losses": max(0, total_matches - total_wins),
            "win_rate": win_rate,
            "rating": user_dict["rating"],
            "total_captures": (part_stats["total_captures"] or 0) if part_stats else 0,
            "total_tokens_finished": (part_stats["total_tokens_finished"] or 0) if part_stats else 0,
            "mode_breakdown": mode_counts,
        }
        # Flatten key statistics for template, API, and client convenience
        user_dict["total_matches"] = total_matches
        user_dict["total_wins"] = total_wins
        user_dict["total_losses"] = max(0, total_matches - total_wins)
        user_dict["win_rate"] = win_rate
        user_dict["total_captures"] = user_dict["stats"]["total_captures"]
        user_dict["total_tokens_finished"] = user_dict["stats"]["total_tokens_finished"]
        user_dict["by_mode"] = mode_counts
        user_dict["mode_breakdown"] = mode_counts
        return user_dict


def update_user_profile(
    user_id: int,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    avatar_color: Optional[str] = None,
    bio: Optional[str] = None,
) -> Dict[str, Any]:
    """Update user profile fields with integrity validation and strict constraints."""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
            existing = cursor.fetchone()
            if not existing:
                raise ValueError(f"User with ID {user_id} not found.")

            # Check email uniqueness if email provided and changed
            clean_email = None
            if email is not None:
                if email.strip():
                    clean_email = email.strip().lower()
                    if len(clean_email) > 100 or "@" not in clean_email or "." not in clean_email.split("@")[-1]:
                        raise ValueError("Invalid email address format.")
                    cursor.execute("SELECT id FROM users WHERE email = ? AND id != ?", (clean_email, user_id))
                    if cursor.fetchone():
                        raise ValueError(f"Email '{clean_email}' is already registered to another account.")

            fields = []
            params = []
            if full_name is not None:
                clean_name = full_name.strip()[:100]
                fields.append("full_name = ?")
                params.append(clean_name if clean_name else None)
            if email is not None:
                fields.append("email = ?")
                params.append(clean_email)
            if avatar_color is not None:
                valid_colors = {"rose", "emerald", "amber", "blue", "indigo", "purple"}
                col = avatar_color.strip().lower()
                if col not in valid_colors:
                    raise ValueError(f"Invalid avatar color '{avatar_color}'. Allowed colors: {', '.join(sorted(valid_colors))}")
                fields.append("avatar_color = ?")
                params.append(col)
            if bio is not None:
                fields.append("bio = ?")
                params.append(bio.strip()[:300])

            if fields:
                params.append(user_id)
                query = f"UPDATE users SET {', '.join(fields)} WHERE id = ?"
                cursor.execute(query, tuple(params))

    return get_user_profile(user_id) or {}


def get_user_match_history(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve detailed match history for a specific user."""
    db_path = get_db_path()
    matches: List[Dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                m.id,
                m.id as match_id,
                m.session_id,
                COALESCE(NULLIF(m.game_mode, ''), m.match_mode) as game_mode,
                m.room_code,
                m.is_online,
                m.team_winner,
                m.winner_id,
                m.winner_id as winner,
                m.winner_name,
                m.winner_controller,
                m.total_turns,
                m.created_at,
                m.created_at as timestamp,
                p.player_seat,
                p.team_id,
                p.controller_type,
                p.is_winner,
                p.tokens_finished,
                p.total_captures,
                p.final_score
            FROM match_records m
            LEFT JOIN match_participants p ON m.id = p.match_id AND p.user_id = ?
            WHERE p.user_id = ? OR m.user_id = ?
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (user_id, user_id, user_id, limit),
        )
        for row in cursor.fetchall():
            m = dict(row)
            is_win = bool(m.get("is_winner"))
            if not is_win and m.get("player_seat") is not None and m.get("winner_id") is not None:
                is_win = (m["player_seat"] == m["winner_id"])
            m["outcome"] = "Won" if is_win else "Lost"
            matches.append(m)
    return matches


# ==============================================================================
# Match Recording & History
# ==============================================================================

def save_match_record(
    session_id: str = "default",
    match_mode: str = "1v1",
    winner_id: Optional[int] = None,
    winner_name: Optional[str] = None,
    winner_controller: Optional[str] = None,
    total_turns: int = 0,
    player_types: Optional[Dict[int, str]] = None,
    player_scores: Optional[Dict[int, float]] = None,
    user_id: Optional[int] = None,
    game_mode: Optional[str] = None,
    room_code: Optional[str] = None,
    is_online: bool = False,
    team_winner: Optional[int] = None,
    participants: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> int:
    """Persist a completed match and its participants into SQLite database."""
    if winner_id is None and "winner" in kwargs:
        winner_id = kwargs["winner"]
    if player_types is None:
        player_types = {}
    if player_scores is None:
        player_scores = {}
    db_path = get_db_path()
    created_at = datetime.now(timezone.utc).isoformat()
    clean_game_mode = game_mode or match_mode

    try:
        with sqlite3.connect(db_path, timeout=15.0) as conn:
            with conn:
                cursor = conn.cursor()

                # Prevent duplicate match records for the same authoritative room
                clean_code = room_code.strip() if room_code else ""
                if clean_code and clean_code.upper() != "LOCAL-1V1":
                    cursor.execute("SELECT id FROM match_records WHERE room_code = ?", (clean_code,))
                    existing = cursor.fetchone()
                    if existing:
                        return existing[0]

                if session_id and session_id != "default" and session_id.startswith("room_"):
                    cursor.execute("SELECT id FROM match_records WHERE session_id = ?", (session_id.strip(),))
                    existing = cursor.fetchone()
                    if existing:
                        return existing[0]

                cursor.execute(
                    """
                    INSERT INTO match_records (
                        session_id, match_mode, winner_id, winner_name, winner_controller,
                        total_turns, p0_type, p1_type, p2_type, p3_type,
                        p0_score, p1_score, p2_score, p3_score, created_at,
                        user_id, game_mode, room_code, is_online, team_winner
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        match_mode,
                        winner_id,
                        winner_name,
                        winner_controller,
                        total_turns,
                        player_types.get(0, ""),
                        player_types.get(1, ""),
                        player_types.get(2, ""),
                        player_types.get(3, ""),
                        player_scores.get(0, 0.0),
                        player_scores.get(1, 0.0),
                        player_scores.get(2, 0.0),
                        player_scores.get(3, 0.0),
                        created_at,
                        user_id,
                        clean_game_mode,
                        room_code or "",
                        1 if is_online else 0,
                        team_winner,
                    ),
                )
                match_id = cursor.lastrowid

                # Insert participant records and update ratings atomically
                if participants and match_id:
                    for p in participants:
                        p_uid = p.get("user_id")
                        p_win = 1 if p.get("is_winner") else 0
                        cursor.execute(
                            """
                            INSERT INTO match_participants (
                                match_id, user_id, username, player_seat, team_id,
                                controller_type, is_winner, tokens_finished, total_captures, final_score
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                match_id,
                                p_uid,
                                p.get("username", f"Player {p.get('player_seat', 0)}"),
                                p.get("player_seat", 0),
                                p.get("team_id"),
                                p.get("controller_type", "human"),
                                p_win,
                                p.get("tokens_finished", 0),
                                p.get("total_captures", 0),
                                p.get("final_score", 0.0),
                            ),
                        )
                        # Adjust rating for human users (+25 for win, -15 for loss, minimum 100)
                        if p_uid:
                            rating_delta = 25 if p_win else -15
                            cursor.execute(
                                "UPDATE users SET rating = MAX(100, rating + ?) WHERE id = ?",
                                (rating_delta, p_uid),
                            )

                return match_id
    except sqlite3.IntegrityError:
        # Cross-process race condition: another process persisted this match record first
        with sqlite3.connect(db_path, timeout=15.0) as fetch_conn:
            fetch_cursor = fetch_conn.cursor()
            if clean_code and clean_code.upper() != "LOCAL-1V1":
                fetch_cursor.execute("SELECT id FROM match_records WHERE room_code = ?", (clean_code,))
                existing = fetch_cursor.fetchone()
                if existing:
                    return existing[0]
            if session_id and session_id != "default" and session_id.startswith("room_"):
                fetch_cursor.execute("SELECT id FROM match_records WHERE session_id = ?", (session_id.strip(),))
                existing = fetch_cursor.fetchone()
                if existing:
                    return existing[0]
            raise


def get_recent_matches(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent match records from SQLite."""
    db_path = get_db_path()
    matches: List[Dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM match_records ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        for row in cursor.fetchall():
            matches.append(dict(row))
    return matches


def get_match_statistics_summary() -> Dict[str, Any]:
    """Calculate live win rates and counts from stored SQLite matches."""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total_matches FROM match_records")
        total_matches = cursor.fetchone()["total_matches"]

        cursor.execute(
            """
            SELECT winner_controller, COUNT(*) as wins
            FROM match_records
            WHERE winner_controller IS NOT NULL AND winner_controller != ''
            GROUP BY winner_controller
            ORDER BY wins DESC
            """
        )
        controller_wins = {row["winner_controller"]: row["wins"] for row in cursor.fetchall()}

        cursor.execute(
            """
            SELECT match_mode, COUNT(*) as count
            FROM match_records
            GROUP BY match_mode
            """
        )
        mode_counts = {row["match_mode"]: row["count"] for row in cursor.fetchall()}

    return {
        "total_matches": total_matches,
        "controller_wins": controller_wins,
        "mode_counts": mode_counts,
    }


# Backward-compatible helper functions
def get_configured_credentials() -> Tuple[str, str]:
    """Return local development role."""
    return ("researcher", "")


def is_authenticated() -> bool:
    """Convenience helper."""
    return True


def login(input_user: str = "", input_pwd: str = "") -> bool:
    """Authenticate helper."""
    user = authenticate_user(input_user, input_pwd)
    return user is not None


def logout() -> None:
    """No-op logout."""
    pass


def render_login_page() -> bool:
    """No-op login renderer."""
    return True
