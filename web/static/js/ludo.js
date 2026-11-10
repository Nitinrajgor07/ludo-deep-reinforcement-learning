/**
 * PhD Ludo DRL Platform — Authoritative Arena & Multiplayer Controller
 * Supports:
 * - Tab 1: Playable Arena
 *   - Mode 1: Human vs AI (1v1) with DRL models (DQN, PPO, SAC, C51, Expert, Random)
 *   - Mode 2: Human vs Human (1v1) — Local Pass-and-Play & Online WebSocket Rooms
 *   - Mode 3: Team vs Team (2v2) — Local Pass-and-Play & Online WebSocket Rooms
 * - Tab 2: Research AI Tournament (Preserved 4-Player AI Tournament with Telemetry)
 *
 * 100% Light Theme AI Research Standard. Zero Streamlit.
 */

// =============================================================================
// GLOBAL STATE & METADATA
// =============================================================================

const PLAYER_COLORS_META = {
    0: { name: 'Red', hex: '#E11D48', badgeClass: 'badge-rose' },
    1: { name: 'Green', hex: '#059669', badgeClass: 'badge-emerald' },
    2: { name: 'Yellow', hex: '#D97706', badgeClass: 'badge-amber' },
    3: { name: 'Blue', hex: '#0284C7', badgeClass: 'badge-blue' }
};

// Playable Modes State
let playableState = {
    mode: null,              // '1v1_ai' | '1v1_human_local' | '1v1_human_online' | '2v2_team_local' | '2v2_team_online'
    isOnline: false,
    roomCode: null,
    currentRoom: null,       // Authoritative Room state from server
    aiSessionId: null,       // Session ID for Mode 1 AI
    aiModel: 'dqn',
    ws: null,
    wsHeartbeat: null,
    mySeatId: null,
    actionSeq: 0,
    isRolling: false,
    isAIStepping: false,     // Prevent duplicate AI requests
    matchStartTime: null
};

// Research AI Tournament State (Preserved)
let researchState = {
    sessionId: 'ludo4p_' + Math.random().toString(36).substring(2, 9),
    isBusy: false,
    timerLimit: 15,
    timerSeconds: 15,
    timerInterval: null
};

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', function () {
    // Check URL query param for active tab
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('tab') === 'research') {
        switchArenaTab('research');
    } else {
        switchArenaTab('playable');
    }

    // Initialize Research Tournament in background
    initResearchTournament();

    // Auto-uppercase room code input
    const joinInput = document.getElementById('joinRoomCodeInput');
    if (joinInput) {
        joinInput.addEventListener('input', function (e) {
            this.value = this.value.toUpperCase().replace(/[^A-Z0-9]/g, '');
        });
    }
});

// =============================================================================
// TAB NAVIGATION: PLAYABLE ARENA vs RESEARCH AI TOURNAMENT
// =============================================================================

function switchArenaTab(tabName) {
    const btnPlayable = document.getElementById('tabBtnPlayable');
    const btnResearch = document.getElementById('tabBtnResearch');
    const tabPlayable = document.getElementById('tabPlayableArena');
    const tabResearch = document.getElementById('tabResearchTournament');

    if (tabName === 'playable') {
        if (btnPlayable) btnPlayable.classList.add('active');
        if (btnResearch) btnResearch.classList.remove('active');
        if (tabPlayable) tabPlayable.style.display = 'block';
        if (tabResearch) tabResearch.style.display = 'none';
    } else {
        if (btnResearch) btnResearch.classList.add('active');
        if (btnPlayable) btnPlayable.classList.remove('active');
        if (tabResearch) tabResearch.style.display = 'block';
        if (tabPlayable) tabPlayable.style.display = 'none';
        fetchMatchesHistory();
    }
}

// =============================================================================
// PLAYABLE MODES SELECTION (LOCAL vs ONLINE TOGGLES)
// =============================================================================

function setModeOption(modeKey, optionType) {
    const pillLocal = document.getElementById(`pill${modeKey}Local`);
    const pillOnline = document.getElementById(`pill${modeKey}Online`);
    const groupLocal = document.getElementById(`btnGroup${modeKey}Local`);
    const groupOnline = document.getElementById(`btnGroup${modeKey}Online`);

    if (optionType === 'local') {
        if (pillLocal) pillLocal.classList.add('active');
        if (pillOnline) pillOnline.classList.remove('active');
        if (groupLocal) groupLocal.style.display = 'flex';
        if (groupOnline) groupOnline.style.display = 'none';
    } else {
        if (pillOnline) pillOnline.classList.add('active');
        if (pillLocal) pillLocal.classList.remove('active');
        if (groupOnline) groupOnline.style.display = 'grid';
        if (groupLocal) groupLocal.style.display = 'none';
    }
}

// =============================================================================
// MODE 1: HUMAN VS AI (1v1)
// =============================================================================

async function startMode1HumanVsAI() {
    const select = document.getElementById('aiOpponentSelect');
    const opponent = select ? select.value : 'dqn';

    playableState.mode = '1v1_ai';
    playableState.isOnline = false;
    playableState.aiModel = opponent;
    playableState.aiSessionId = '1v1ai_' + Math.random().toString(36).substring(2, 9);
    playableState.mySeatId = 0; // Human is always Player 0 (Red)
    playableState.isAIStepping = false;
    playableState.matchStartTime = Date.now();

    try {
        const resp = await fetch('/api/ludo/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: playableState.aiSessionId,
                num_players: 2,
                p0_type: 'human',
                p1_type: opponent,
                seed: 42
            })
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || 'Failed to start 1v1 AI match.');
        }
        const data = await resp.json();

        // Switch to Active Match View
        document.getElementById('playableModeSelectSection').style.display = 'none';
        document.getElementById('multiplayerLobbySection').style.display = 'none';
        document.getElementById('playableArenaSection').style.display = 'block';
        document.getElementById('teamBannerContainer').style.display = 'none';
        document.getElementById('localPassNotice').style.display = 'none';

        const arenaTitle = document.getElementById('playableArenaTitle');
        if (arenaTitle) arenaTitle.textContent = `Human (Red) vs ${opponent.toUpperCase()} AI (Yellow) — 1v1 Arena`;

        renderPlayableGameStateFromSummary(data.state, data.svg);

        // Auto-step AI if match begins on AI turn
        if (!data.state.is_terminal && !data.state.is_human_turn) {
            setTimeout(step1v1AI, 600);
        }

    } catch (err) {
        alert('Error starting 1v1 AI Match: ' + err.message);
    }
}

// =============================================================================
// MODES 2 & 3: LOCAL PASS-AND-PLAY MATCHES
// =============================================================================

async function startLocalMatch(gameMode) {
    // gameMode: '1v1_human' or '2v2_team'
    playableState.mode = gameMode === '1v1_human' ? '1v1_human_local' : '2v2_team_local';
    playableState.isOnline = false;

    try {
        // 1. Create Local Room (is_online: false auto-fills local seats)
        const createResp = await fetch('/api/rooms/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game_mode: gameMode,
                is_online: false
            })
        });

        if (!createResp.ok) {
            const errData = await createResp.json().catch(() => ({}));
            throw new Error(errData.detail || 'Failed to create local match.');
        }

        const createData = await createResp.json();
        const code = createData.room_code;
        playableState.roomCode = code;
        playableState.currentRoom = createData.room;
        playableState.mySeatId = 0; // Host is Seat 0

        // 2. Open Dedicated Local Demonstration Lobby
        enterLobbyView(createData.room);

    } catch (err) {
        alert('Error starting Local Match: ' + err.message);
    }
}

// =============================================================================
// MODES 2 & 3: ONLINE MULTIPLAYER (ROOM CREATION & JOINING)
// =============================================================================

function openCreateRoomModal(defaultMode) {
    const sel = document.getElementById('createRoomModeSelect');
    if (sel && defaultMode) sel.value = defaultMode;
    const err = document.getElementById('createRoomError');
    if (err) err.style.display = 'none';
    const modal = document.getElementById('modalCreateRoom');
    if (modal) modal.style.display = 'flex';
}

function closeCreateRoomModal(e) {
    if (e && e.target !== e.currentTarget) return;
    const modal = document.getElementById('modalCreateRoom');
    if (modal) modal.style.display = 'none';
}

async function submitCreateRoom(e) {
    e.preventDefault();
    const sel = document.getElementById('createRoomModeSelect');
    const mode = sel ? sel.value : '1v1_human';
    const errBox = document.getElementById('createRoomError');
    const submitBtn = document.getElementById('btnSubmitCreateRoom');

    if (submitBtn) submitBtn.disabled = true;
    if (errBox) errBox.style.display = 'none';

    try {
        const resp = await fetch('/api/rooms/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game_mode: mode,
                is_online: true
            })
        });

        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Failed to create room.');

        playableState.roomCode = data.room_code;
        playableState.isOnline = true;
        playableState.mode = mode === '1v1_human' ? '1v1_human_online' : '2v2_team_online';
        playableState.currentRoom = data.room;
        playableState.mySeatId = 0; // Host is Seat 0

        closeCreateRoomModal();
        enterLobbyView(data.room);
        connectRoomWebSocket(data.room_code);

    } catch (err) {
        if (errBox) {
            errBox.textContent = err.message;
            errBox.style.display = 'block';
        }
    } finally {
        if (submitBtn) submitBtn.disabled = false;
    }
}

function openJoinRoomModal() {
    const input = document.getElementById('joinRoomCodeInput');
    if (input) input.value = '';
    const err = document.getElementById('joinRoomError');
    if (err) err.style.display = 'none';
    const modal = document.getElementById('modalJoinRoom');
    if (modal) modal.style.display = 'flex';
    setTimeout(() => { if (input) input.focus(); }, 100);
}

function closeJoinRoomModal(e) {
    if (e && e.target !== e.currentTarget) return;
    const modal = document.getElementById('modalJoinRoom');
    if (modal) modal.style.display = 'none';
}

async function submitJoinRoom(e) {
    e.preventDefault();
    const input = document.getElementById('joinRoomCodeInput');
    const code = (input ? input.value : '').trim().toUpperCase();
    const errBox = document.getElementById('joinRoomError');
    const submitBtn = document.getElementById('btnSubmitJoinRoom');

    if (code.length !== 6) {
        if (errBox) {
            errBox.textContent = 'Please enter a valid 6-character room code.';
            errBox.style.display = 'block';
        }
        return;
    }

    if (submitBtn) submitBtn.disabled = true;
    if (errBox) errBox.style.display = 'none';

    try {
        const resp = await fetch('/api/rooms/join', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ room_code: code })
        });

        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Failed to join room.');

        playableState.roomCode = code;
        playableState.isOnline = true;
        playableState.currentRoom = data.room;
        playableState.mode = data.room.game_mode === '1v1_human' ? '1v1_human_online' : '2v2_team_online';
        playableState.mySeatId = data.seat.seat_id;

        closeJoinRoomModal();
        enterLobbyView(data.room);
        connectRoomWebSocket(code);

    } catch (err) {
        if (errBox) {
            errBox.textContent = err.message;
            errBox.style.display = 'block';
        }
    } finally {
        if (submitBtn) submitBtn.disabled = false;
    }
}

// =============================================================================
// LOBBY VIEW & READY / START CONTROLS
// =============================================================================

function enterLobbyView(room) {
    document.getElementById('playableModeSelectSection').style.display = 'none';
    document.getElementById('playableArenaSection').style.display = 'none';
    document.getElementById('multiplayerLobbySection').style.display = 'block';

    renderLobby(room);
}

function renderLobby(room) {
    if (!room) return;
    playableState.currentRoom = room;

    const codeDisplay = document.getElementById('lobbyRoomCodeDisplay');
    if (codeDisplay) {
        codeDisplay.textContent = room.room_code || 'DEMO2V2';
    }

    const modeBadge = document.getElementById('lobbyModeBadge');
    if (modeBadge) {
        if (!room.is_online) {
            modeBadge.className = 'badge badge-emerald';
            modeBadge.textContent = room.game_mode === '1v1_human' ? 'Local 1v1 Demonstration' : 'Local 2v2 Team Demonstration';
        } else {
            modeBadge.className = 'badge badge-indigo';
            modeBadge.textContent = room.game_mode === '1v1_human' ? '1v1 Online Match' : '2v2 Team Co-op';
        }
    }

    const connBadge = document.getElementById('lobbyConnectionBadge');
    if (connBadge) {
        if (!room.is_online) {
            connBadge.className = 'badge badge-zinc';
            connBadge.textContent = 'Local Pass-and-Play (Offline Demo)';
        } else {
            connBadge.className = 'badge badge-emerald';
            connBadge.textContent = 'Online WebSocket';
        }
    }

    const statusBadge = document.getElementById('lobbyStatusBadge');
    if (statusBadge) {
        if (!room.is_online) {
            statusBadge.className = 'badge badge-emerald';
            statusBadge.textContent = 'All 4 Local Players Ready — Ready to Start!';
        } else if (room.status === 'ready') {
            statusBadge.className = 'badge badge-emerald';
            statusBadge.textContent = 'All Players Ready — Ready to Start!';
        } else {
            statusBadge.className = 'badge badge-amber';
            statusBadge.textContent = `Waiting for Players (${room.player_count}/${room.max_players})...`;
        }
    }

    // Render Seat Cards
    const seatsContainer = document.getElementById('lobbySeatsContainer');
    if (seatsContainer) {
        seatsContainer.innerHTML = '';
        const myUserId = window.CURRENT_USER ? window.CURRENT_USER.id : null;

        for (let i = 0; i < room.max_players; i++) {
            const player = room.players.find(p => p.seat_id === i);
            const isFilled = !!player;
            const isMe = (player && player.user_id === myUserId);
            const isHost = (player && player.user_id === room.host_user_id) || (!room.is_online && i === 0);

            // Determine Seat Color & Team
            let seatColor = 'Red';
            let teamLabel = '';
            let teamBadgeClass = 'badge-indigo';
            let playerDemoTitle = '';

            if (room.game_mode === '1v1_human') {
                seatColor = (i === 0) ? 'Red' : 'Yellow';
                playerDemoTitle = (i === 0) ? 'Player 1 — Red' : 'Player 2 — Yellow';
            } else {
                seatColor = ['Red', 'Green', 'Yellow', 'Blue'][i];
                if (i === 0) {
                    playerDemoTitle = 'Player 1 — Red — Team 0';
                    teamLabel = 'Team 0: Red+Yellow';
                    teamBadgeClass = 'badge-rose';
                } else if (i === 1) {
                    playerDemoTitle = 'Player 2 — Green — Team 1';
                    teamLabel = 'Team 1: Green+Blue';
                    teamBadgeClass = 'badge-emerald';
                } else if (i === 2) {
                    playerDemoTitle = 'Player 3 — Yellow — Team 0';
                    teamLabel = 'Team 0: Red+Yellow';
                    teamBadgeClass = 'badge-rose';
                } else {
                    playerDemoTitle = 'Player 4 — Blue — Team 1';
                    teamLabel = 'Team 1: Green+Blue';
                    teamBadgeClass = 'badge-emerald';
                }
            }

            const card = document.createElement('div');
            card.className = `lobby-seat-card ${isMe ? 'active-seat' : ''}`;

            if (isFilled) {
                const avatarLetter = (player.username || 'U').substring(0, 1).toUpperCase();
                const avatarColor = player.avatar_color || (i === 0 ? 'rose' : (i === 1 ? 'emerald' : (i === 2 ? 'amber' : 'blue')));
                const readyText = player.is_ready ? '✓ Ready' : '⏳ Not Ready';
                const readyClass = player.is_ready ? 'badge-emerald' : 'badge-amber';
                const connText = (!room.is_online)
                    ? '🖥️ Pass-and-Play Seat'
                    : (player.is_connected ? '🟢 Connected' : '⚪ Disconnected');

                let displayUsername = `@${player.username}`;
                let roleSubtitle = `Rating: ${player.rating || 1000}`;
                if (!room.is_online) {
                    if (i === 0) {
                        roleSubtitle = `Host Player (@${player.username})`;
                    } else {
                        roleSubtitle = `Local Demonstration Player (@${player.username})`;
                    }
                }

                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 11px; font-weight: 800; color: var(--text-secondary);">SEAT ${i} — ${seatColor.toUpperCase()}</span>
                        <div style="display: flex; gap: 4px;">
                            ${teamLabel ? `<span class="badge ${teamBadgeClass}" style="font-size: 9.5px; font-weight: 800;">${teamLabel}</span>` : ''}
                            ${isHost ? `<span class="badge badge-indigo" style="font-size: 9.5px; font-weight: 800;">HOST</span>` : ''}
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px; margin: 4px 0;">
                        <div class="avatar-circle avatar-${avatarColor}" style="width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; font-weight: 900; background: var(--surface-secondary); border: 2px solid #fff;">
                            ${avatarLetter}
                        </div>
                        <div>
                            <div style="font-weight: 800; font-size: 13.5px; color: var(--text-primary);">
                                ${!room.is_online ? playerDemoTitle : (displayUsername + (isMe ? ' <span style="font-size: 10px; color: var(--accent);">(You)</span>' : ''))}
                            </div>
                            <div style="font-size: 11px; color: var(--text-secondary);">${roleSubtitle}</div>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; border-top: 1px solid var(--border-subtle); padding-top: 8px;">
                        <span style="color: var(--text-secondary); font-size: 11px;">${connText}</span>
                        <span class="badge ${readyClass}" style="font-size: 10px;">${!room.is_online ? '✓ Ready (Local)' : readyText}</span>
                    </div>
                `;
            } else {
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 11px; font-weight: 800; color: var(--text-secondary);">SEAT ${i} — ${seatColor.toUpperCase()}</span>
                        ${teamLabel ? `<span class="badge ${teamBadgeClass}" style="font-size: 9.5px; font-weight: 800;">${teamLabel}</span>` : ''}
                    </div>
                    <div style="display: flex; align-items: center; justify-content: center; height: 50px; color: var(--text-muted); font-size: 12px; font-style: italic;">
                        Waiting for player to join...
                    </div>
                    <div style="font-size: 11px; color: var(--text-faint); text-align: center; border-top: 1px dashed var(--border-subtle); padding-top: 8px;">
                        Open Slot
                    </div>
                `;
            }
            seatsContainer.appendChild(card);
        }
    }

    // Update Action Buttons
    const myUserId = window.CURRENT_USER ? window.CURRENT_USER.id : null;
    const isHost = (room.host_user_id === myUserId) || (!room.is_online);
    const myPlayer = room.players.find(p => p.user_id === myUserId);

    const readyBtn = document.getElementById('btnLobbyReady');
    const startBtn = document.getElementById('btnLobbyStart');

    if (!room.is_online) {
        if (readyBtn) readyBtn.style.display = 'none';
        if (startBtn) {
            startBtn.style.display = 'inline-flex';
            startBtn.disabled = false;
            startBtn.style.opacity = '1';
            startBtn.textContent = '▶ Start Match';
        }
    } else {
        if (readyBtn) {
            readyBtn.style.display = 'inline-flex';
            if (myPlayer && myPlayer.is_ready) {
                readyBtn.textContent = '✗ Cancel Ready';
                readyBtn.className = 'btn btn-secondary';
            } else {
                readyBtn.textContent = '✓ I am Ready';
                readyBtn.className = 'btn btn-primary';
            }
        }
        if (startBtn) {
            if (isHost) {
                startBtn.style.display = 'inline-flex';
                startBtn.disabled = (room.status !== 'ready');
                startBtn.style.opacity = (room.status === 'ready') ? '1' : '0.5';
                startBtn.textContent = '▶ Start Match';
            } else {
                startBtn.style.display = 'none';
            }
        }
    }
}

function copyRoomCode() {
    if (!playableState.roomCode) return;
    navigator.clipboard.writeText(playableState.roomCode).then(() => {
        const btnText = document.getElementById('copyBtnText');
        if (btnText) {
            btnText.textContent = 'Copied!';
            setTimeout(() => { btnText.textContent = 'Copy Code'; }, 2000);
        }
    }).catch(() => {
        prompt('Copy your room code:', playableState.roomCode);
    });
}

async function togglePlayerReady() {
    if (!playableState.currentRoom) return;
    const myUserId = window.CURRENT_USER ? window.CURRENT_USER.id : null;
    const myPlayer = playableState.currentRoom.players.find(p => p.user_id === myUserId);
    const nextReady = myPlayer ? !myPlayer.is_ready : true;

    if (playableState.ws && playableState.ws.readyState === WebSocket.OPEN) {
        playableState.ws.send(JSON.stringify({
            type: 'PLAYER_READY',
            is_ready: nextReady
        }));
    } else {
        // Fallback REST
        try {
            const resp = await fetch('/api/rooms/ready', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    room_code: playableState.roomCode,
                    is_ready: nextReady
                })
            });
            const data = await resp.json();
            if (resp.ok) renderLobby(data.room);
        } catch (err) {
            console.error('Ready toggle error:', err);
        }
    }
}

async function startMultiplayerMatch() {
    if (!playableState.roomCode) return;

    if (playableState.ws && playableState.ws.readyState === WebSocket.OPEN) {
        playableState.ws.send(JSON.stringify({
            type: 'START_GAME'
        }));
    } else {
        // Fallback REST
        try {
            const resp = await fetch('/api/rooms/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ room_code: playableState.roomCode })
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Could not start match.');
            onMatchStarted(data.room);
        } catch (err) {
            alert('Cannot start game: ' + err.message);
        }
    }
}

async function leaveMultiplayerRoom() {
    if (playableState.ws) {
        playableState.ws.close();
        playableState.ws = null;
    }
    if (playableState.roomCode) {
        fetch('/api/rooms/leave', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ room_code: playableState.roomCode })
        }).catch(() => {});
    }
    exitPlayableMatch();
}

function exitPlayableMatch() {
    if (playableState.ws) {
        playableState.ws.close();
        playableState.ws = null;
    }
    clearInterval(playableState.wsHeartbeat);
    playableState.currentRoom = null;
    playableState.roomCode = null;
    playableState.aiSessionId = null;

    closeVictoryModal();

    document.getElementById('playableModeSelectSection').style.display = 'block';
    document.getElementById('multiplayerLobbySection').style.display = 'none';
    document.getElementById('playableArenaSection').style.display = 'none';
}

// =============================================================================
// WEBSOCKET CLIENT INTEGRATION (/ws/rooms/{room_code})
// =============================================================================

function connectRoomWebSocket(roomCode) {
    if (playableState.ws) {
        playableState.ws.close();
        playableState.ws = null;
    }
    clearInterval(playableState.wsHeartbeat);

    // Build dynamic address without hardcoded IP
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/rooms/${roomCode}`;

    const ws = new WebSocket(wsUrl);
    playableState.ws = ws;

    ws.onopen = function () {
        // Send JOIN_ROOM handshake event
        ws.send(JSON.stringify({ type: 'JOIN_ROOM' }));

        // Start 25s PING heartbeat
        playableState.wsHeartbeat = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'PING' }));
            }
        }, 25000);
    };

    ws.onmessage = function (event) {
        try {
            const msg = JSON.parse(event.data);
            handleWebSocketMessage(msg);
        } catch (e) {
            console.error('WebSocket parse error:', e);
        }
    };

    ws.onerror = function (err) {
        console.warn('WebSocket connection notice:', err);
    };

    ws.onclose = function (evt) {
        clearInterval(playableState.wsHeartbeat);
        if (evt.code === 1008) {
            alert('WebSocket access denied: ' + (evt.reason || 'Unauthorized session.'));
            exitPlayableMatch();
        }
    };
}

function handleWebSocketMessage(msg) {
    const type = msg.type;

    if (type === 'PONG') {
        return;
    }

    if (type === 'PLAYER_READY' || (type === 'GAME_STATE_UPDATE' && msg.event === 'PLAYER_READY')) {
        renderLobby(msg.state);
        return;
    }

    if (type === 'GAME_STATE_UPDATE' && msg.event === 'GAME_STARTED') {
        onMatchStarted(msg.state);
        return;
    }

    if (type === 'DICE_ROLLED') {
        animatePlayableDice(msg.dice);
        renderPlayableRoomState(msg.state);
        return;
    }

    if (type === 'GAME_STATE_UPDATE') {
        renderPlayableRoomState(msg.state);
        return;
    }

    if (type === 'GAME_FINISHED') {
        renderPlayableRoomState(msg.state);
        onPlayableGameFinished(msg.winner, msg.team_winner, msg.state);
        return;
    }

    if (type === 'ERROR') {
        alert('Action Notice: ' + msg.message);
        return;
    }
}

function onMatchStarted(room) {
    playableState.currentRoom = room;
    playableState.matchStartTime = Date.now();

    document.getElementById('playableModeSelectSection').style.display = 'none';
    document.getElementById('multiplayerLobbySection').style.display = 'none';
    document.getElementById('playableArenaSection').style.display = 'block';

    const isTeam = (room.game_mode === '2v2_team');
    document.getElementById('teamBannerContainer').style.display = isTeam ? 'grid' : 'none';
    document.getElementById('localPassNotice').style.display = (!room.is_online) ? 'flex' : 'none';

    const arenaTitle = document.getElementById('playableArenaTitle');
    if (arenaTitle) {
        if (!room.is_online) {
            arenaTitle.textContent = isTeam ? 'Local 2v2 Team vs Team Demonstration Arena' : 'Local Human vs Human (1v1) Arena';
        } else {
            arenaTitle.textContent = isTeam ? 'Online Team vs Team (2v2) Arena' : 'Online Human vs Human (1v1) Arena';
        }
    }

    renderPlayableRoomState(room);
}

// =============================================================================
// AUTHORITATIVE PLAYABLE BOARD & ACTIONS RENDERING
// =============================================================================

function renderPlayableRoomState(room) {
    if (!room) return;
    playableState.currentRoom = room;
    playableState.actionSeq = room.action_seq || 0;

    // 1. Render SVG Board
    const boardContainer = document.getElementById('playableBoardContainer');
    if (boardContainer && room.svg) {
        boardContainer.innerHTML = room.svg;
    }

    // 2. Active Turn & Player
    const currSeatId = room.current_player;
    const activePlayer = room.players.find(p => p.seat_id === currSeatId);
    const currUsername = activePlayer ? activePlayer.username : `Player ${currSeatId}`;
    const myUserId = window.CURRENT_USER ? window.CURRENT_USER.id : null;
    const isMyTurn = playableState.isOnline ? (activePlayer && activePlayer.user_id === myUserId) : true;

    // Color Name
    let colorName = 'Red';
    let badgeClass = 'badge-rose';
    if (room.game_mode === '1v1_human') {
        colorName = currSeatId === 0 ? 'Red' : 'Yellow';
        badgeClass = currSeatId === 0 ? 'badge-rose' : 'badge-amber';
    } else {
        const c = PLAYER_COLORS_META[currSeatId] || PLAYER_COLORS_META[0];
        colorName = c.name;
        badgeClass = c.badgeClass;
    }

    let displayLabel = currUsername.includes('(') ? currUsername : `${colorName} (${currUsername})`;
    if (!room.is_online && room.game_mode === '2v2_team') {
        const currTeam = (currSeatId === 0 || currSeatId === 2) ? 0 : 1;
        displayLabel = `Player ${currSeatId + 1} — ${colorName} (Team ${currTeam})`;
    }

    const turnBadge = document.getElementById('playableTurnBadge');
    if (turnBadge) {
        turnBadge.className = `badge ${badgeClass}`;
        turnBadge.textContent = `${displayLabel} — Active Turn`;
    }

    // 3. Local Pass-Device Banner & Next Player Indicator
    const passNotice = document.getElementById('localPassNotice');
    const passNoticeText = document.getElementById('localPassNoticeText');
    if (!room.is_online) {
        if (passNotice) passNotice.style.display = 'flex';
        if (passNoticeText) {
            const nextSeatId = (currSeatId + 1) % room.max_players;
            const nextColors = (room.game_mode === '1v1_human') ? ['Red', 'Yellow'] : ['Red', 'Green', 'Yellow', 'Blue'];
            const nextColor = nextColors[nextSeatId];
            const nextTeam = (room.game_mode === '2v2_team') ? ((nextSeatId === 0 || nextSeatId === 2) ? 'Team 0' : 'Team 1') : '';
            passNoticeText.textContent = `🎮 Pass Device to ${displayLabel} • Next Up: Player ${nextSeatId + 1} (${nextColor}${nextTeam ? ', ' + nextTeam : ''})`;
        }
    } else {
        if (passNotice) passNotice.style.display = 'none';
    }

    // 4. Action Phase & Dice Box
    const phaseBadge = document.getElementById('playableActionPhaseBadge');
    const rollBtn = document.getElementById('btnPlayableRoll');
    const diceBox = document.getElementById('playableDiceBox3D');

    if (room.dice_value && diceBox) {
        diceBox.textContent = room.dice_value;
    }

    if (room.turn_phase === 'NEED_ROLL') {
        if (phaseBadge) {
            phaseBadge.className = 'badge badge-indigo';
            phaseBadge.textContent = 'Phase: Roll Dice';
        }
        if (rollBtn) {
            rollBtn.disabled = !isMyTurn;
            rollBtn.style.opacity = isMyTurn ? '1' : '0.5';
            if (!room.is_online && room.game_mode === '2v2_team') {
                rollBtn.textContent = isMyTurn ? `🎲 Roll Authoritative Dice (Player ${currSeatId + 1} — ${colorName})` : `Awaiting Roll...`;
            } else {
                rollBtn.textContent = isMyTurn ? '🎲 Roll Authoritative Dice' : `Awaiting ${currUsername} to Roll...`;
            }
        }
    } else {
        if (phaseBadge) {
            phaseBadge.className = 'badge badge-amber';
            phaseBadge.textContent = 'Phase: Select Token Move';
        }
        if (rollBtn) {
            rollBtn.disabled = true;
            rollBtn.style.opacity = '0.5';
            rollBtn.textContent = `Rolled ${room.dice_value} — Awaiting Move`;
        }
    }

    // 5. Legal Action Buttons
    const movesContainer = document.getElementById('playableLegalActionsContainer');
    const moveNotice = document.getElementById('playableMoveNotice');

    if (movesContainer) {
        movesContainer.innerHTML = '';
        if (room.turn_phase === 'NEED_MOVE' && room.legal_actions && room.legal_actions.length > 0) {
            if (moveNotice) moveNotice.style.display = isMyTurn ? 'inline-block' : 'none';

            room.legal_actions.forEach(tid => {
                const btn = document.createElement('button');
                btn.className = 'btn btn-secondary';
                btn.style.fontWeight = '700';
                btn.style.fontSize = '12px';
                btn.style.padding = '6px 14px';
                btn.disabled = !isMyTurn;
                btn.textContent = `Move Token ${tid}`;
                btn.onclick = () => onPlayableMoveToken(tid);
                movesContainer.appendChild(btn);
            });
        } else if (room.turn_phase === 'NEED_MOVE') {
            movesContainer.innerHTML = '<span style="font-size: 12px; color: var(--text-secondary);">No legal moves possible with this roll. Passing turn...</span>';
        } else {
            if (moveNotice) moveNotice.style.display = 'none';
            movesContainer.innerHTML = '<span style="font-size: 12px; color: var(--text-secondary);">Roll the dice first to reveal valid moves.</span>';
        }
    }

    // 6. 2v2 Team Progress Indicators
    if (room.game_mode === '2v2_team') {
        const team0Fin = room.players.filter(p => p.team_id === 0).reduce((acc, p) => acc + (p.tokens_finished || 0), 0);
        const team1Fin = room.players.filter(p => p.team_id === 1).reduce((acc, p) => acc + (p.tokens_finished || 0), 0);

        const pill0 = document.getElementById('team0FinishedPill');
        const pill1 = document.getElementById('team1FinishedPill');
        if (pill0) pill0.textContent = `Team 0: ${team0Fin} / 8 Finished (${8 - team0Fin} Remaining)`;
        if (pill1) pill1.textContent = `Team 1: ${team1Fin} / 8 Finished (${8 - team1Fin} Remaining)`;
    }

    // 7. Move History Log Table
    const historyBody = document.getElementById('playableHistoryBody');
    const turnCountBadge = document.getElementById('playableTurnCountBadge');
    if (turnCountBadge) turnCountBadge.textContent = `Turn: ${room.turn_count || 0}`;

    if (historyBody && room.recent_history && room.recent_history.length > 0) {
        historyBody.innerHTML = '';
        room.recent_history.slice().reverse().forEach(entry => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>#${entry.turn || '-'}</strong></td>
                <td>P${entry.player_seat ?? '-'}</td>
                <td>@${entry.username || '-'}</td>
                <td><span class="badge badge-indigo">${entry.dice || '-'}</span></td>
                <td>${entry.action || '-'}</td>
                <td style="font-size: 11.5px; color: var(--text-secondary);">${entry.status || '-'}</td>
            `;
            historyBody.appendChild(tr);
        });
    }

    // 8. Check Victory Terminal
    if (room.status === 'finished') {
        onPlayableGameFinished(room.winner, room.team_winner, room);
    }
}

function renderPlayableGameStateFromSummary(summary, svg) {
    if (!summary) return;

    // Render SVG
    const boardContainer = document.getElementById('playableBoardContainer');
    if (boardContainer && svg) boardContainer.innerHTML = svg;

    // Active Turn
    const turnBadge = document.getElementById('playableTurnBadge');
    if (turnBadge) {
        turnBadge.className = summary.current_player === 0 ? 'badge badge-rose' : 'badge badge-amber';
        turnBadge.textContent = `${summary.current_player_color} (${summary.current_player_controller.toUpperCase()}) ${summary.is_human_turn ? '— YOUR TURN' : '— AI TURN'}`;
    }

    const diceBox = document.getElementById('playableDiceBox3D');
    if (diceBox) diceBox.textContent = summary.dice_value || '-';

    const rollBtn = document.getElementById('btnPlayableRoll');
    const movesContainer = document.getElementById('playableLegalActionsContainer');

    if (rollBtn) {
        const canRoll = summary.is_human_turn && (!summary.legal_actions || summary.legal_actions.length === 0);
        rollBtn.disabled = !canRoll;
        rollBtn.style.opacity = canRoll ? '1' : '0.5';
        if (!summary.is_human_turn) {
            rollBtn.textContent = `Awaiting ${summary.current_player_color} (${summary.current_player_controller.toUpperCase()})...`;
        } else if (canRoll) {
            rollBtn.textContent = '🎲 Roll Authoritative Dice';
        } else {
            rollBtn.textContent = `Rolled ${summary.dice_value} — Awaiting Token Move`;
        }
    }

    // Action phase badge
    const phaseBadge = document.getElementById('playableActionPhaseBadge');
    if (phaseBadge) {
        if (!summary.is_human_turn) {
            phaseBadge.className = 'badge badge-amber';
            phaseBadge.textContent = '🤖 AI Thinking...';
        } else if (summary.legal_actions && summary.legal_actions.length > 0) {
            phaseBadge.className = 'badge badge-emerald';
            phaseBadge.textContent = 'Phase: Select Token Move';
        } else {
            phaseBadge.className = 'badge badge-indigo';
            phaseBadge.textContent = 'Phase: Roll Dice';
        }
    }

    // Turn count badge
    const turnCountBadge = document.getElementById('playableTurnCountBadge');
    if (turnCountBadge) turnCountBadge.textContent = `Turn: ${summary.turn_count || 0}`;

    // History body
    const historyBody = document.getElementById('playableHistoryBody');
    if (historyBody && summary.recent_history && summary.recent_history.length > 0) {
        historyBody.innerHTML = '';
        summary.recent_history.slice().reverse().forEach(entry => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>#${entry.turn || '-'}</strong></td>
                <td>${entry.player || '-'}</td>
                <td>${entry.actor || '-'}</td>
                <td><span class="badge badge-indigo">${entry.dice || '-'}</span></td>
                <td>${entry.action || '-'}</td>
                <td style="font-size: 11.5px; color: var(--text-secondary);">${entry.status || '-'}</td>
            `;
            historyBody.appendChild(tr);
        });
    }

    if (movesContainer) {
        movesContainer.innerHTML = '';
        if (summary.is_human_turn && summary.legal_actions && summary.legal_actions.length > 0) {
            summary.legal_actions.forEach(tid => {
                const btn = document.createElement('button');
                btn.className = 'btn btn-secondary';
                btn.style.fontWeight = '700';
                btn.style.fontSize = '12px';
                btn.style.padding = '6px 14px';
                btn.textContent = `Move Token ${tid}`;
                btn.onclick = () => onPlayableMoveToken(tid);
                movesContainer.appendChild(btn);
            });
        } else if (summary.is_human_turn) {
            movesContainer.innerHTML = `<span style="font-size: 12px; color: var(--text-secondary);">Roll the dice first to reveal valid moves.</span>`;
        } else {
            movesContainer.innerHTML = `<span style="font-size: 12px; color: var(--text-secondary);"><span class="badge badge-amber" style="margin-right: 6px;">AI</span> Waiting for ${summary.current_player_controller.toUpperCase()} to move...</span>`;
        }
    }

    if (summary.is_terminal) {
        onPlayableGameFinished(summary.winner, null, summary);
    }
}

// =============================================================================
// GAMEPLAY INPUT HANDLERS (DICE ROLL & TOKEN MOVES)
// =============================================================================

async function onPlayableRollClick() {
    if (playableState.isRolling) return;
    playableState.isRolling = true;

    // Client Action ID for idempotency & sequence verification
    const clientActionId = 'act_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
    const seq = playableState.actionSeq;

    try {
        if (playableState.mode === '1v1_ai') {
            // Mode 1: Human vs AI REST
            const resp = await fetch('/api/ludo/roll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: playableState.aiSessionId })
            });
            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || 'Dice roll request failed.');
            }
            const data = await resp.json();
            animatePlayableDice(data.dice_value);
            renderPlayableGameStateFromSummary(data.state, data.svg);

            // Automatically trigger AI if turn passed to Yellow
            if (!data.state.is_terminal && !data.state.is_human_turn) {
                setTimeout(step1v1AI, 600);
            }

        } else if (playableState.isOnline && playableState.ws && playableState.ws.readyState === WebSocket.OPEN) {
            // Online Mode: Send via WebSocket
            playableState.ws.send(JSON.stringify({
                type: 'ROLL_DICE',
                client_action_id: clientActionId,
                expected_action_seq: seq
            }));

        } else {
            // Local Mode: REST
            const resp = await fetch('/api/rooms/roll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    room_code: playableState.roomCode,
                    client_action_id: clientActionId,
                    expected_action_seq: seq
                })
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Roll failed.');
            animatePlayableDice(data.dice);
            renderPlayableRoomState(data.room);
        }
    } catch (err) {
        alert('Dice Roll Notice: ' + err.message);
    } finally {
        playableState.isRolling = false;
    }
}

async function onPlayableMoveToken(tokenId) {
    const clientActionId = 'move_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
    const seq = playableState.actionSeq;

    try {
        if (playableState.mode === '1v1_ai') {
            // Mode 1: Human Step
            const resp = await fetch('/api/ludo/manual_step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: playableState.aiSessionId,
                    action: tokenId
                })
            });
            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || 'Token move rejected.');
            }
            const data = await resp.json();
            renderPlayableGameStateFromSummary(data.state, data.svg);

            // Auto-trigger AI turn if it is now AI's turn
            if (!data.is_terminal && !data.state.is_human_turn) {
                setTimeout(step1v1AI, 600);
            }

        } else if (playableState.isOnline && playableState.ws && playableState.ws.readyState === WebSocket.OPEN) {
            // Online Mode: Send via WebSocket
            playableState.ws.send(JSON.stringify({
                type: 'MOVE_TOKEN',
                token_id: tokenId,
                client_action_id: clientActionId,
                expected_action_seq: seq
            }));

        } else {
            // Local Mode: REST
            const resp = await fetch('/api/rooms/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    room_code: playableState.roomCode,
                    token_id: tokenId,
                    client_action_id: clientActionId,
                    expected_action_seq: seq
                })
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Move failed.');
            renderPlayableRoomState(data.room);
        }
    } catch (err) {
        alert('Move Notice: ' + err.message);
    }
}

async function step1v1AI() {
    if (playableState.mode !== '1v1_ai') return;
    if (playableState.isAIStepping) return;
    playableState.isAIStepping = true;

    // Show AI Thinking Indicator in UI
    const phaseBadge = document.getElementById('playableActionPhaseBadge');
    if (phaseBadge) {
        phaseBadge.className = 'badge badge-amber';
        phaseBadge.textContent = '🤖 AI Thinking...';
    }
    const rollBtn = document.getElementById('btnPlayableRoll');
    if (rollBtn) {
        rollBtn.disabled = true;
        rollBtn.style.opacity = '0.5';
        rollBtn.textContent = 'Awaiting AI Move...';
    }
    const movesContainer = document.getElementById('playableLegalActionsContainer');
    if (movesContainer) {
        movesContainer.innerHTML = '<span style="font-size: 12px; color: var(--text-secondary);"><span class="badge badge-amber" style="margin-right: 6px;">AI</span> Processing neural inference and selecting move...</span>';
    }

    try {
        const resp = await fetch('/api/ludo/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: playableState.aiSessionId })
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `AI step request failed (status ${resp.status})`);
        }
        const data = await resp.json();

        if (data.state && data.state.dice_value) {
            animatePlayableDice(data.state.dice_value);
        }

        renderPlayableGameStateFromSummary(data.state, data.svg);

        // If AI got a bonus roll and it's still AI turn, step again
        if (!data.is_terminal && !data.state.is_human_turn) {
            setTimeout(step1v1AI, 800);
        }
    } catch (e) {
        console.error('AI step failed:', e);
        if (phaseBadge) {
            phaseBadge.className = 'badge badge-rose';
            phaseBadge.textContent = 'AI Step Interrupted';
        }
        if (movesContainer) {
            movesContainer.innerHTML = `<span style="font-size: 12px; color: #dc2626;">AI turn encountered an error: ${e.message}. <button class="btn btn-secondary" style="font-size: 11px; padding: 2px 8px; margin-left: 8px;" onclick="step1v1AI()">Retry AI Turn</button></span>`;
        }
    } finally {
        playableState.isAIStepping = false;
    }
}

// Global click handler wired to SVG tokens by the board renderer
window.onBoardTokenClick = function (playerId, tokenId) {
    if (document.getElementById('tabPlayableArena').style.display !== 'none') {
        onPlayableMoveToken(tokenId);
    } else {
        // Research tournament
        manualMoveToken(tokenId);
    }
};

function animatePlayableDice(finalValue) {
    const diceBox = document.getElementById('playableDiceBox3D');
    if (!diceBox) return;

    diceBox.classList.add('rolling');
    let counter = 0;
    const interval = setInterval(() => {
        diceBox.textContent = Math.floor(Math.random() * 6) + 1;
        counter++;
        if (counter >= 6) {
            clearInterval(interval);
            diceBox.textContent = finalValue;
            diceBox.classList.remove('rolling');
        }
    }, 50);
}

// =============================================================================
// VICTORY & MATCH RESULTS MODAL
// =============================================================================

function onPlayableGameFinished(winner, teamWinner, state) {
    const modal = document.getElementById('modalGameFinished');
    const title = document.getElementById('victoryTitle');
    const subtitle = document.getElementById('victorySubtitle');
    const statsBox = document.getElementById('victoryStatsBox');
    if (!modal) return;

    const elapsedSec = playableState.matchStartTime ? Math.round((Date.now() - playableState.matchStartTime) / 1000) : null;
    const durationStr = elapsedSec ? `${Math.floor(elapsedSec / 60)}m ${elapsedSec % 60}s` : 'Realtime Session';

    if (teamWinner !== null && teamWinner !== undefined) {
        const winTeamId = Number(teamWinner);
        const loseTeamId = 1 - winTeamId;

        const winTeamName = winTeamId === 0 ? 'TEAM 0' : 'TEAM 1';
        const winTeamColors = winTeamId === 0 ? 'RED + YELLOW' : 'GREEN + BLUE';
        const winPlayers = winTeamId === 0 ? 'Player 1 (Red) & Player 3 (Yellow)' : 'Player 2 (Green) & Player 4 (Blue)';

        const loseTeamName = loseTeamId === 0 ? 'TEAM 0' : 'TEAM 1';
        const loseTeamColors = loseTeamId === 0 ? 'RED + YELLOW' : 'GREEN + BLUE';
        const losePlayers = loseTeamId === 0 ? 'Player 1 (Red) & Player 3 (Yellow)' : 'Player 2 (Green) & Player 4 (Blue)';

        const team0Fin = state.players ? state.players.filter(p => p.team_id === 0).reduce((acc, p) => acc + (p.tokens_finished || 0), 0) : (winTeamId === 0 ? 8 : 0);
        const team1Fin = state.players ? state.players.filter(p => p.team_id === 1).reduce((acc, p) => acc + (p.tokens_finished || 0), 0) : (winTeamId === 1 ? 8 : 0);

        if (title) title.textContent = `MATCH COMPLETED`;
        if (subtitle) subtitle.textContent = `Authoritative 2v2 Team Victory Confirmed by Game Engine`;

        if (statsBox) {
            statsBox.innerHTML = `
                <!-- Winner Banner -->
                <div style="background: #ECFDF5; border: 1.5px solid #10B981; border-radius: var(--radius-compact); padding: 14px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <strong style="color: #065F46; font-size: 13.5px; letter-spacing: 0.3px;">
                            🏆 WINNER: ${winTeamName} — ${winTeamColors}
                        </strong>
                        <span class="badge badge-emerald" style="font-weight: 800; font-size: 11px;">8 / 8 Finished</span>
                    </div>
                    <div style="font-size: 12px; color: #047857; font-weight: 600;">
                        Winning Roster: ${winPlayers}
                    </div>
                </div>

                <!-- Loser Banner -->
                <div style="background: #F8FAFC; border: 1.5px solid #CBD5E1; border-radius: var(--radius-compact); padding: 14px 16px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <strong style="color: var(--text-secondary); font-size: 13.5px; letter-spacing: 0.3px;">
                            LOSER: ${loseTeamName} — ${loseTeamColors}
                        </strong>
                        <span class="badge badge-zinc" style="font-weight: 800; font-size: 11px;">
                            ${loseTeamId === 0 ? team0Fin : team1Fin} / 8 Finished
                        </span>
                    </div>
                    <div style="font-size: 12px; color: var(--text-secondary); font-weight: 600;">
                        Opposing Roster: ${losePlayers}
                    </div>
                </div>

                <!-- Match Telemetry -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; padding-top: 10px; border-top: 1px solid var(--border-subtle);">
                    <div>
                        <span style="color: var(--text-secondary);">Total Turns:</span>
                        <strong style="margin-left: 4px; color: var(--text-primary);">${state.turn_count || 0}</strong>
                    </div>
                    <div>
                        <span style="color: var(--text-secondary);">Match Duration:</span>
                        <strong style="margin-left: 4px; color: var(--text-primary);">${durationStr}</strong>
                    </div>
                    <div>
                        <span style="color: var(--text-secondary);">Team 0 Score:</span>
                        <strong style="margin-left: 4px; color: var(--text-primary);">${team0Fin} / 8 Finished</strong>
                    </div>
                    <div>
                        <span style="color: var(--text-secondary);">Team 1 Score:</span>
                        <strong style="margin-left: 4px; color: var(--text-primary);">${team1Fin} / 8 Finished</strong>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 11.5px; color: #059669; display: flex; align-items: center; gap: 6px; font-weight: 600;">
                    <span>✓</span> Match persisted to SQLite match history repository.
                </div>
            `;
        }
    } else {
        const winnerName = PLAYER_COLORS_META[winner] ? PLAYER_COLORS_META[winner].name : `Player ${winner}`;
        if (title) title.textContent = `MATCH COMPLETED`;
        if (subtitle) subtitle.textContent = `Authoritative 1v1 Match Result`;
        if (statsBox) {
            statsBox.innerHTML = `
                <div style="background: #ECFDF5; border: 1.5px solid #10B981; border-radius: var(--radius-compact); padding: 14px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #065F46; font-size: 13.5px;">🏆 WINNER: ${winnerName}</strong>
                        <span class="badge badge-emerald">Victorious</span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; padding-top: 8px;">
                    <div><span style="color: var(--text-secondary);">Total Turns:</span> <strong>${state.turn_count || 0}</strong></div>
                    <div><span style="color: var(--text-secondary);">Duration:</span> <strong>${durationStr}</strong></div>
                </div>
            `;
        }
    }

    modal.style.display = 'flex';
}

function closeVictoryModal() {
    const modal = document.getElementById('modalGameFinished');
    if (modal) modal.style.display = 'none';
}

function restartPlayableMatch() {
    closeVictoryModal();
    if (playableState.mode && playableState.mode.includes('2v2')) {
        startLocalMatch('2v2_team');
    } else if (playableState.mode && playableState.mode.includes('1v1_human')) {
        startLocalMatch('1v1_human');
    } else {
        startMode1HumanVsAI();
    }
}

function viewMatchResultsFromModal() {
    closeVictoryModal();
    switchLudoTab('research');
    fetchMatchesHistory();
    const hist = document.getElementById('realtimeMatchesSection');
    if (hist) hist.scrollIntoView({ behavior: 'smooth' });
}


// =============================================================================
// SECTION B: PRESERVED RESEARCH AI TOURNAMENT JAVASCRIPT
// =============================================================================

function initResearchTournament() {
    startNewGame();
    fetchMatchesHistory();
}

function applyPresetMode(mode) {
    const p0 = document.getElementById('p0Select');
    const p1 = document.getElementById('p1Select');
    const p2 = document.getElementById('p2Select');
    const p3 = document.getElementById('p3Select');
    if (!p0 || !p1 || !p2 || !p3) return;

    if (mode === '2p_2ai') {
        p0.value = 'human';
        p1.value = 'human';
        p2.value = 'dqn';
        p3.value = 'ppo';
    } else if (mode === '1p_3ai') {
        p0.value = 'human';
        p1.value = 'dqn';
        p2.value = 'ppo';
        p3.value = 'expert';
    } else if (mode === 'team_2v2') {
        p0.value = 'human';
        p1.value = 'human';
        p2.value = 'dqn';
        p3.value = 'ppo';
    } else if (mode === 'ai_tournament') {
        p0.value = 'dqn';
        p1.value = 'expert';
        p2.value = 'random';
        p3.value = 'expert';
    }
    startNewGame();
}

function setTurnTimerDuration(val) {
    researchState.timerLimit = parseInt(val, 10);
    resetResearchTurnTimer();
}

function resetResearchTurnTimer() {
    clearInterval(researchState.timerInterval);
    const badge = document.getElementById('turnTimerBadge');
    const progress = document.getElementById('turnProgressBar');
    const secondsEl = document.getElementById('timerSeconds');

    if (researchState.timerLimit <= 0) {
        if (badge) badge.textContent = '⏱️ Timer: Off';
        if (secondsEl) secondsEl.textContent = 'No limit';
        if (progress) progress.style.width = '100%';
        return;
    }

    researchState.timerSeconds = researchState.timerLimit;
    updateTimerDisplay();

    researchState.timerInterval = setInterval(() => {
        if (researchState.isBusy) return;
        researchState.timerSeconds--;
        updateTimerDisplay();

        if (researchState.timerSeconds <= 0) {
            clearInterval(researchState.timerInterval);
            onTimerExpired();
        }
    }, 1000);
}

function updateTimerDisplay() {
    const badge = document.getElementById('turnTimerBadge');
    const progress = document.getElementById('turnProgressBar');
    const secondsEl = document.getElementById('timerSeconds');

    if (badge) badge.textContent = `⏱️ Turn Timer: ${researchState.timerSeconds}s`;
    if (secondsEl) secondsEl.textContent = `${researchState.timerSeconds}s remaining`;

    if (progress && researchState.timerLimit > 0) {
        const pct = Math.max(0, (researchState.timerSeconds / researchState.timerLimit) * 100);
        progress.style.width = `${pct}%`;
        progress.style.backgroundColor = (pct < 30) ? '#EF4444' : ((pct < 60) ? '#F59E0B' : '#10B981');
    }
}

function onTimerExpired() {
    const banner = document.getElementById('terminalBanner');
    if (banner) banner.textContent = 'Timer Expired — Stepping turn';
    stepGame();
}

async function startNewGame() {
    const p0 = document.getElementById('p0Select') ? document.getElementById('p0Select').value : 'dqn';
    const p1 = document.getElementById('p1Select') ? document.getElementById('p1Select').value : 'expert';
    const p2 = document.getElementById('p2Select') ? document.getElementById('p2Select').value : 'random';
    const p3 = document.getElementById('p3Select') ? document.getElementById('p3Select').value : 'expert';
    const seed = document.getElementById('seedInput') ? parseInt(document.getElementById('seedInput').value, 10) : 42;

    researchState.sessionId = 'ludo4p_' + Math.random().toString(36).substring(2, 9);

    try {
        const resp = await fetch('/api/ludo/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: researchState.sessionId,
                num_players: 4,
                p0_type: p0,
                p1_type: p1,
                p2_type: p2,
                p3_type: p3,
                seed: seed
            })
        });
        const data = await resp.json();
        updateResearchUI(data.state, data.svg);
        resetResearchTurnTimer();
    } catch (e) {
        console.error('Error starting research game:', e);
    }
}

async function manualRollDice() {
    try {
        const resp = await fetch('/api/ludo/roll', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: researchState.sessionId })
        });
        const data = await resp.json();
        const diceBox = document.getElementById('diceBox3D');
        if (diceBox) {
            diceBox.classList.add('rolling');
            setTimeout(() => {
                diceBox.textContent = data.dice_value;
                diceBox.classList.remove('rolling');
            }, 300);
        }
        updateResearchUI(data.state, data.svg);
    } catch (e) {
        console.error('Error rolling dice:', e);
    }
}

async function manualMoveToken(tokenId) {
    try {
        const resp = await fetch('/api/ludo/manual_step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: researchState.sessionId,
                action: tokenId
            })
        });
        const data = await resp.json();
        updateResearchUI(data.state, data.svg);
        resetResearchTurnTimer();
    } catch (e) {
        console.error('Manual move error:', e);
    }
}

async function stepGame() {
    researchState.isBusy = true;
    try {
        const resp = await fetch('/api/ludo/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: researchState.sessionId })
        });
        const data = await resp.json();
        updateResearchUI(data.state, data.svg);
        resetResearchTurnTimer();
    } catch (e) {
        console.error('Step game error:', e);
    } finally {
        researchState.isBusy = false;
    }
}

async function autoPlay(n) {
    researchState.isBusy = true;
    try {
        const resp = await fetch('/api/ludo/auto_play', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: researchState.sessionId,
                steps: n
            })
        });
        const data = await resp.json();
        updateResearchUI(data.state, data.svg);
        resetResearchTurnTimer();
    } catch (e) {
        console.error('Autoplay error:', e);
    } finally {
        researchState.isBusy = false;
    }
}

function updateResearchUI(state, svg) {
    if (!state) return;

    // SVG Board
    const boardContainer = document.getElementById('boardContainer');
    if (boardContainer && svg) boardContainer.innerHTML = svg;

    // Turn info
    const turnInfo = document.getElementById('turnInfo');
    if (turnInfo) {
        const c = PLAYER_COLORS_META[state.current_player] || PLAYER_COLORS_META[0];
        turnInfo.className = `badge ${c.badgeClass}`;
        turnInfo.textContent = `Player ${state.current_player} (${c.name}) — ${state.current_player_controller.toUpperCase()}`;
    }

    const diceBox = document.getElementById('diceBox3D');
    if (diceBox && state.dice_value) diceBox.textContent = state.dice_value;

    // Update 4 stats cards
    if (state.players) {
        state.players.forEach(p => {
            const pid = p.player_id;
            const yardEl = document.getElementById(`p${pid}Yard`);
            const trackEl = document.getElementById(`p${pid}Track`);
            const goalEl = document.getElementById(`p${pid}Goal`);
            const capsEl = document.getElementById(`p${pid}Caps`);
            const rewEl = document.getElementById(`p${pid}Rew`);
            const badgeEl = document.getElementById(`p${pid}CtrlBadge`);

            if (yardEl) yardEl.textContent = p.yard_count;
            if (trackEl) trackEl.textContent = p.track_count;
            if (goalEl) goalEl.textContent = p.finished_count;
            if (capsEl) capsEl.textContent = p.captures;
            if (rewEl) rewEl.textContent = p.reward;
            if (badgeEl) badgeEl.textContent = p.controller;
        });
    }

    // Legal Action Buttons
    const movesContainer = document.getElementById('legalActionsContainer');
    const manualBadge = document.getElementById('manualTurnBadge');
    if (movesContainer) {
        movesContainer.innerHTML = '';
        if (state.is_human_turn && state.legal_actions && state.legal_actions.length > 0) {
            if (manualBadge) manualBadge.style.display = 'inline-block';
            state.legal_actions.forEach(act => {
                const btn = document.createElement('button');
                btn.className = 'btn btn-secondary';
                btn.style.fontWeight = '700';
                btn.style.fontSize = '12px';
                btn.textContent = `Move Token ${act}`;
                btn.onclick = () => manualMoveToken(act);
                movesContainer.appendChild(btn);
            });
        } else {
            if (manualBadge) manualBadge.style.display = 'none';
            movesContainer.innerHTML = `<span style="font-size: 12px; color: var(--text-secondary);">${state.is_human_turn ? 'No legal moves with roll' : 'AI Turn in progress'}</span>`;
        }
    }

    // Telemetry Table
    const histBody = document.getElementById('turnHistoryBody');
    const turnBadge = document.getElementById('turnCountBadge');
    if (turnBadge) turnBadge.textContent = `Turn: ${state.turn_count || 0}`;

    if (histBody && state.recent_history && state.recent_history.length > 0) {
        histBody.innerHTML = '';
        state.recent_history.slice().reverse().forEach(h => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>#${h.turn}</strong></td>
                <td>P${h.player}</td>
                <td>${h.color}</td>
                <td><span class="badge badge-indigo">${h.dice}</span></td>
                <td>Token ${h.token}</td>
                <td>${h.reward}</td>
                <td>${h.capture}</td>
                <td>${h.bonus}</td>
                <td>${h.status}</td>
            `;
            histBody.appendChild(tr);
        });
    }
}

async function fetchMatchesHistory() {
    try {
        const resp = await fetch('/api/ludo/matches?limit=10');
        const data = await resp.json();
        if (data.status === 'success') {
            const body = document.getElementById('realtimeMatchesBody');
            const countBadge = document.getElementById('realtimeMatchCountBadge');
            if (countBadge) countBadge.textContent = `${data.matches.length} Matches Stored`;

            if (body && data.matches.length > 0) {
                body.innerHTML = '';
                data.matches.forEach(m => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><code>#${m.id}</code></td>
                        <td><span class="badge badge-indigo">${m.match_mode || 'Match'}</span></td>
                        <td><strong>${m.winner_name || 'N/A'}</strong></td>
                        <td><span class="badge badge-rose">${m.winner_controller || 'AI'}</span></td>
                        <td>${m.total_turns} turns</td>
                        <td style="font-size: 11px;">${m.player_config || '-'}</td>
                        <td style="font-size: 11px; color: var(--text-secondary);">${m.created_at ? m.created_at.substring(0, 16) : '-'}</td>
                    `;
                    body.appendChild(tr);
                });
            }
        }
    } catch (e) {
        console.error('Failed to fetch match history:', e);
    }
}
