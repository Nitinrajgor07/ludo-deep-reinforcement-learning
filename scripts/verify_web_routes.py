"""Verify all web routes, static assets, and figures against the live FastAPI server."""
import urllib.request
import re
import sys

routes = [
    '/',
    '/dashboard',
    '/models',
    '/training',
    '/evaluation',
    '/statistics',
    '/visualizations',
    '/ludo',
    '/about',
    '/static/css/theme.css',
    '/static/js/main.js',
    '/static/js/ludo.js',
    '/static/assets/ludo_logo.svg',
    '/api/status',
    '/api/models',
    '/api/checkpoints',
    '/api/visualizations'
]

base = 'http://127.0.0.1:8000'
all_ok = True

print("=== TESTING ALL ROUTES AND ASSETS ON 127.0.0.1:8000 ===")
for r in routes:
    url = base + r
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            code = resp.getcode()
            content = resp.read()
            length = len(content)
            mime = resp.headers.get('Content-Type', '')
            print(f"[PASS] {r:30} -> {code} ({mime.split(';')[0]}) [{length} bytes]")
    except Exception as e:
        print(f"[FAIL] {r:30} -> ERROR: {e}")
        all_ok = False

# Inspect all figure URLs on /visualizations
print("\n=== TESTING FIGURE ASSETS ON /visualizations ===")
try:
    with urllib.request.urlopen(base + '/visualizations') as resp:
        html = resp.read().decode('utf-8')
        figs = re.findall(r'src="(/figures/[^"]+)"', html)
        print(f"Discovered {len(figs)} figure images on /visualizations:")
        for fig in figs:
            fig_url = base + fig
            try:
                with urllib.request.urlopen(fig_url) as fresp:
                    size = len(fresp.read())
                    print(f"  [PASS] {fig:35} -> {fresp.getcode()} [{size} bytes]")
            except Exception as fe:
                print(f"  [FAIL] {fig:35} -> ERROR: {fe}")
                all_ok = False
except Exception as e:
    print(f"Error reading /visualizations: {e}")
    all_ok = False

# Test interactive Ludo API lifecycle
print("\n=== TESTING 4-PLAYER LUDO API LIFECYCLE ===")
import json
try:
    # 1. New 4-player game
    req_data = json.dumps({
        "session_id": "route_check_session",
        "p0_type": "dqn",
        "p1_type": "expert",
        "p2_type": "random",
        "p3_type": "expert",
        "seed": 42
    }).encode('utf-8')
    req = urllib.request.Request(base + '/api/ludo/new', data=req_data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f"  [PASS] POST /api/ludo/new          -> 200 (Turn: {data['state']['turn_count']}, Player: {data['state']['current_player']})")

    # 2. Step 1 turn
    step_data = json.dumps({"session_id": "route_check_session"}).encode('utf-8')
    req = urllib.request.Request(base + '/api/ludo/step', data=step_data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f"  [PASS] POST /api/ludo/step         -> 200 (Action: {data.get('action')}, Terminal: {data['state']['is_terminal']})")

except Exception as le:
    print(f"  [FAIL] 4-Player Ludo API error: {le}")
    all_ok = False

if all_ok:
    print("\n==================================================================")
    print("ALL ROUTES, STATIC ASSETS, FIGURES, AND API CALLS PASSED (0 ERRORS)")
    print("==================================================================")
    sys.exit(0)
else:
    print("\nSome tests failed!")
    sys.exit(1)
