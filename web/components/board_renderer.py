"""Interactive SVG Board Renderer for Standard 4-Player Ludo Environment.

Redesigned for the Soft Light Minimal Card visual system.
Supports 4 Active Players:
- Player 0: Red (Top-Left Yard, Start Sq 0, Horizontal Home Row 7 Cols 1..5)
- Player 1: Green (Top-Right Yard, Start Sq 13, Vertical Home Col 7 Rows 1..5)
- Player 2: Yellow (Bottom-Right Yard, Start Sq 26, Horizontal Home Row 7 Cols 13..9)
- Player 3: Blue (Bottom-Left Yard, Start Sq 39, Vertical Home Col 7 Rows 13..9)
"""

from typing import Any, Dict, List, Optional, Tuple
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.constants import TokenState, SAFE_SQUARES

# 15x15 standard Ludo grid coordinates (row, col) for common track squares 0..51
TRACK_COORDINATES: Dict[int, Tuple[int, int]] = {
    0: (6, 1), 1: (6, 2), 2: (6, 3), 3: (6, 4), 4: (6, 5),
    5: (5, 6), 6: (4, 6), 7: (3, 6), 8: (2, 6), 9: (1, 6), 10: (0, 6),
    11: (0, 7), 12: (0, 8),
    13: (1, 8), 14: (2, 8), 15: (3, 8), 16: (4, 8), 17: (5, 8),
    18: (6, 9), 19: (6, 10), 20: (6, 11), 21: (6, 12), 22: (6, 13), 23: (6, 14),
    24: (7, 14), 25: (8, 14),
    26: (8, 13), 27: (8, 12), 28: (8, 11), 29: (8, 10), 30: (8, 9),
    31: (9, 8), 32: (10, 8), 33: (11, 8), 34: (12, 8), 35: (13, 8), 36: (14, 8),
    37: (14, 7), 38: (14, 6),
    39: (13, 6), 40: (12, 6), 41: (11, 6), 42: (10, 6), 43: (9, 6),
    44: (8, 5), 45: (8, 4), 46: (8, 3), 47: (8, 2), 48: (8, 1), 49: (8, 0),
    50: (7, 0), 51: (6, 0),
}

# Dedicated 5-square Home Columns (steps 51..55) for all 4 players
HOME_COLUMNS: Dict[int, Dict[int, Tuple[int, int]]] = {
    # P0 (Red): moving right towards center
    0: {51: (7, 1), 52: (7, 2), 53: (7, 3), 54: (7, 4), 55: (7, 5)},
    # P1 (Green): moving down towards center
    1: {51: (1, 7), 52: (2, 7), 53: (3, 7), 54: (4, 7), 55: (5, 7)},
    # P2 (Yellow): moving left towards center
    2: {51: (7, 13), 52: (7, 12), 53: (7, 11), 54: (7, 10), 55: (7, 9)},
    # P3 (Blue): moving up towards center
    3: {51: (13, 7), 52: (12, 7), 53: (11, 7), 54: (10, 7), 55: (9, 7)},
}

# Yard coordinates (r, c) for 4 tokens per player
YARD_SLOTS: Dict[int, List[Tuple[float, float]]] = {
    # Red (Top-Left)
    0: [(1.8, 1.8), (1.8, 4.2), (4.2, 1.8), (4.2, 4.2)],
    # Green (Top-Right)
    1: [(1.8, 10.8), (1.8, 13.2), (4.2, 10.8), (4.2, 13.2)],
    # Yellow (Bottom-Right)
    2: [(10.8, 10.8), (10.8, 13.2), (13.2, 10.8), (13.2, 13.2)],
    # Blue (Bottom-Left)
    3: [(10.8, 1.8), (10.8, 4.2), (13.2, 1.8), (13.2, 4.2)],
}

# Soft Light Minimal Player Metadata
PLAYER_COLORS: Dict[int, Dict[str, str]] = {
    0: {"name": "Red", "hex": "#E11D48", "dark": "#9F1239", "light": "#FFF1F2", "token": "R"},
    1: {"name": "Green", "hex": "#059669", "dark": "#065F46", "light": "#ECFDF5", "token": "G"},
    2: {"name": "Yellow", "hex": "#D97706", "dark": "#92400E", "light": "#FFFBEB", "token": "Y"},
    3: {"name": "Blue", "hex": "#0284C7", "dark": "#1E40AF", "light": "#F0F9FF", "token": "B"},
}

START_SQUARES_MAP: Dict[int, int] = {
    0: 0,   # Red
    13: 1,  # Green
    26: 2,  # Yellow
    39: 3,  # Blue
}


def render_ludo_board_svg(
    tokens: List[Token],
    current_player: int,
    dice_value: int,
    legal_actions: List[int],
    selected_action: Optional[int] = None,
    winner: Optional[int] = None,
    size: int = 600,
) -> str:
    """Generate an authoritative SVG visualization of the 4-Player Ludo board in Soft Light Card theme."""
    cell = size / 15.0
    safe_set = set(SAFE_SQUARES)

    svg: List[str] = [
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background: #FAFAFC; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); font-family: Plus Jakarta Sans, system-ui, sans-serif;">',
        '<defs>',
        '  <filter id="shadow" x="-30%" y="-30%" width="160%" height="160%">',
        '    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.15"/>',
        '  </filter>',
        '  <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">',
        '    <feDropShadow dx="0" dy="0" stdDeviation="5" flood-color="#4F46E5" flood-opacity="0.6"/>',
        '  </filter>',
        '</defs>',
    ]

    # 1. Base Grid (15x15 Track & Home Cells)
    for r in range(15):
        for c in range(15):
            x = c * cell
            y = r * cell

            # Yard Regions (rendered separately below)
            if (r < 6 and c < 6) or (r < 6 and c >= 9) or (r >= 9 and c >= 9) or (r >= 9 and c < 6):
                continue
            # Center Goal Region (rendered separately below)
            elif 6 <= r <= 8 and 6 <= c <= 8:
                continue
            else:
                bg = "#FFFFFF"
                border = "#E2E8F0"

                # Check for start squares or safe squares
                for sq, coord in TRACK_COORDINATES.items():
                    if coord == (r, c):
                        if sq == 0:
                            bg = "#FFF1F2"  # Red start
                            border = "#E11D48"
                        elif sq == 13:
                            bg = "#ECFDF5"  # Green start
                            border = "#059669"
                        elif sq == 26:
                            bg = "#FFFBEB"  # Yellow start
                            border = "#D97706"
                        elif sq == 39:
                            bg = "#F0F9FF"  # Blue start
                            border = "#0284C7"
                        elif sq in safe_set:
                            bg = "#FEF3C7"  # Star safe square
                            border = "#F59E0B"
                        break

                # Check for Home Columns
                if r == 7 and 1 <= c <= 5:
                    bg = "#FFF1F2"  # Red Home
                    border = "#E11D48"
                elif c == 7 and 1 <= r <= 5:
                    bg = "#ECFDF5"  # Green Home
                    border = "#059669"
                elif r == 7 and 9 <= c <= 13:
                    bg = "#FFFBEB"  # Yellow Home
                    border = "#D97706"
                elif c == 7 and 9 <= r <= 13:
                    bg = "#F0F9FF"  # Blue Home
                    border = "#0284C7"

                svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{bg}" stroke="{border}" stroke-width="1.2"/>')

    # 2. Four Active Colored Yard Bases
    yards_config = [
        # (x, y, fill, border, label, p_id)
        (0, 0, "#FFF1F2", "#E11D48", "RED YARD (P0)", 0),
        (9 * cell, 0, "#ECFDF5", "#059669", "GREEN YARD (P1)", 1),
        (9 * cell, 9 * cell, "#FFFBEB", "#D97706", "YELLOW YARD (P2)", 2),
        (0, 9 * cell, "#F0F9FF", "#0284C7", "BLUE YARD (P3)", 3),
    ]

    for yx, yy, fill, stroke, label, pid in yards_config:
        # Yard base container
        svg.append(f'<rect x="{yx}" y="{yy}" width="{6*cell}" height="{6*cell}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        # Inner white card
        svg.append(f'<rect x="{yx + 0.8*cell}" y="{yy + 0.8*cell}" width="{4.4*cell}" height="{4.4*cell}" fill="#FFFFFF" rx="12" stroke="{stroke}" stroke-width="1.2"/>')
        # Yard title
        svg.append(f'<text x="{yx + 3*cell}" y="{yy + 1.4*cell}" font-size="{cell*0.36}px" font-weight="800" fill="{stroke}" text-anchor="middle" letter-spacing="0.5px">{label}</text>')
        # Yard token slots
        for sr, sc in YARD_SLOTS[pid]:
            slot_fill = "#FAFAFC"
            slot_border = stroke
            svg.append(f'<circle cx="{sc*cell}" cy="{sr*cell}" r="{cell*0.44}" fill="{slot_fill}" stroke="{slot_border}" stroke-width="1.2" stroke-dasharray="3,2"/>')

    # 3. Center Goal Triangles
    cx, cy = 7.5 * cell, 7.5 * cell
    # Left: Red
    svg.append(f'<polygon points="{6*cell},{6*cell} {cx},{cy} {6*cell},{9*cell}" fill="#FFF1F2" stroke="#E11D48" stroke-width="1.5"/>')
    # Top: Green
    svg.append(f'<polygon points="{6*cell},{6*cell} {cx},{cy} {9*cell},{6*cell}" fill="#ECFDF5" stroke="#059669" stroke-width="1.5"/>')
    # Right: Yellow
    svg.append(f'<polygon points="{9*cell},{6*cell} {cx},{cy} {9*cell},{9*cell}" fill="#FFFBEB" stroke="#D97706" stroke-width="1.5"/>')
    # Bottom: Blue
    svg.append(f'<polygon points="{6*cell},{9*cell} {cx},{cy} {9*cell},{9*cell}" fill="#F0F9FF" stroke="#0284C7" stroke-width="1.5"/>')

    # Center Goal Emblem
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{cell*0.72}" fill="#FFFFFF" stroke="#4F46E5" stroke-width="2" filter="url(#shadow)"/>')
    svg.append(f'<text x="{cx}" y="{cy + cell*0.14}" font-size="{cell*0.42}px" font-weight="900" fill="#19191F" text-anchor="middle" letter-spacing="0.5px">GOAL</text>')

    # 4. Safe Square Indicators (Glowing Sparkle Stars)
    for sq in safe_set:
        r, c = TRACK_COORDINATES[sq]
        sx = (c + 0.5) * cell
        sy = (r + 0.65) * cell
        star_color = "#D97706"
        svg.append(f'<text x="{sx}" y="{sy}" font-size="{cell*0.65}px" text-anchor="middle" fill="{star_color}">★</text>')

    # 5. Render Tokens
    token_positions: Dict[Tuple[float, float], List[Tuple[Token, str, str, bool, bool]]] = {}

    for t in tokens:
        pid = t.player_id % 4
        tid = t.token_id
        color_info = PLAYER_COLORS.get(pid, PLAYER_COLORS[0])
        hex_color = color_info["hex"]
        label = f"{color_info['token']}{tid}"

        is_legal = (t.player_id == current_player) and (tid in legal_actions)
        is_selected = (t.player_id == current_player) and (tid == selected_action)

        if t.is_in_yard:
            slots = YARD_SLOTS.get(pid, YARD_SLOTS[0])
            pos = slots[tid % 4]
        elif t.is_on_track and t.track_square is not None:
            r, c = TRACK_COORDINATES.get(t.track_square, (0, 0))
            pos = (r + 0.5, c + 0.5)
        elif t.is_in_home and t.step_count is not None:
            hcol = HOME_COLUMNS.get(pid, HOME_COLUMNS[0])
            r, c = hcol.get(t.step_count, (7, 7))
            pos = (r + 0.5, c + 0.5)
        else:
            # Finished in Goal
            goal_positions = {
                0: (7.5, 6.7),
                1: (6.7, 7.5),
                2: (7.5, 8.3),
                3: (8.3, 7.5),
            }
            pos = goal_positions.get(pid, (7.5, 7.5))

        token_positions.setdefault(pos, []).append((t, hex_color, label, is_legal, is_selected))

    # Render tokens with visual offset if multiple tokens share the same square
    for (r, c), group in token_positions.items():
        base_x = c * cell
        base_y = r * cell
        count = len(group)

        for i, (t, color, label, is_legal, is_selected) in enumerate(group):
            if count == 1:
                tx = base_x
                ty = base_y
            else:
                offset = cell * 0.22
                tx = base_x + offset * (1 if i % 2 == 0 else -1)
                ty = base_y + offset * (1 if i < 2 else -1)

            t_radius = cell * 0.38
            filter_attr = 'filter="url(#glow)"' if is_legal else 'filter="url(#shadow)"'
            stroke_color = "#4F46E5" if is_selected else ("#ffffff" if is_legal else "rgba(255,255,255,0.8)")
            stroke_width = "3.5" if (is_legal or is_selected) else "1.5"

            svg.append(
                f'<circle cx="{tx}" cy="{ty}" r="{t_radius}" fill="{color}" '
                f'stroke="{stroke_color}" stroke-width="{stroke_width}" {filter_attr}/>'
            )
            svg.append(
                f'<text x="{tx}" y="{ty + cell*0.12}" font-size="{cell*0.28}px" font-weight="800" '
                f'fill="#ffffff" text-anchor="middle">{label}</text>'
            )

    svg.append('</svg>')
    return "".join(svg)
