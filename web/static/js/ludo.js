/**
 * Interactive 4-Player Ludo Demonstration Arena Client Controller.
 * Deep Cosmic AI Laboratory Theme Edition with 3D Animated Dice & Micro-Interactions.
 */

let currentSessionId = 'ludo4p_' + Math.random().toString(36).substring(2, 9);
let isBusy = false;

const PLAYER_THEMES = {
    0: { color: 'Red', badgeClass: 'badge-rose', hex: '#ff2a5f' },
    1: { color: 'Green', badgeClass: 'badge-emerald', hex: '#00e676' },
    2: { color: 'Yellow', badgeClass: 'badge-amber', hex: '#ffd600' },
    3: { color: 'Blue', badgeClass: 'badge-blue', hex: '#00b0ff' }
};

document.addEventListener('DOMContentLoaded', function () {
    startNewGame();
});

async function startNewGame() {
    if (isBusy) return;
    isBusy = true;

    const p0 = document.getElementById('p0Select').value;
    const p1 = document.getElementById('p1Select').value;
    const p2 = document.getElementById('p2Select').value;
    const p3 = document.getElementById('p3Select').value;
    const seed = parseInt(document.getElementById('seedInput').value, 10) || 42;

    try {
        const response = await fetch('/api/ludo/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                p0_type: p0,
                p1_type: p1,
                p2_type: p2,
                p3_type: p3,
                seed: seed,
            }),
        });

        if (!response.ok) throw new Error('Failed to initialize 4-player match');
        const data = await response.json();
        updateUI(data.state, data.svg);
    } catch (err) {
        console.error('Error starting 4-player game:', err);
    } finally {
        isBusy = false;
    }
}

async function stepGame() {
    if (isBusy) return;
    isBusy = true;

    animateDiceRoll();

    try {
        const response = await fetch('/api/ludo/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId }),
        });

        if (!response.ok) throw new Error('Failed to step turn');
        const data = await response.json();
        updateUI(data.state, data.svg);
    } catch (err) {
        console.error('Error stepping match:', err);
    } finally {
        isBusy = false;
    }
}

async function playerAction(playerId, tokenId) {
    if (isBusy) return;
    isBusy = true;

    try {
        const response = await fetch('/api/ludo/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                player_id: playerId,
                token_id: tokenId,
            }),
        });

        if (!response.ok) {
            const errData = await response.json();
            alert(errData.detail || 'Invalid action');
            return;
        }
        const data = await response.json();
        updateUI(data.state, data.svg);
    } catch (err) {
        console.error('Error executing action:', err);
    } finally {
        isBusy = false;
    }
}

async function autoPlay(steps) {
    if (isBusy) return;
    isBusy = true;

    animateDiceRoll();

    try {
        const response = await fetch('/api/ludo/auto_play', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                steps: steps || 5,
            }),
        });

        if (!response.ok) throw new Error('Failed to execute auto play turns');
        const data = await response.json();
        updateUI(data.state, data.svg);
    } catch (err) {
        console.error('Error in auto play:', err);
    } finally {
        isBusy = false;
    }
}

function animateDiceRoll() {
    const diceBox = document.getElementById('diceBox3D');
    if (diceBox) {
        diceBox.classList.remove('dice-rolling');
        void diceBox.offsetWidth; // Trigger reflow
        diceBox.classList.add('dice-rolling');
    }
}

function updateUI(state, svg) {
    if (!state) return;

    // 1. Board SVG
    if (svg) {
        const container = document.getElementById('boardContainer');
        if (container) {
            container.innerHTML = svg;
        }
    }

    // 2. Turn Info & Badge
    const turnInfo = document.getElementById('turnInfo');
    const currP = state.current_player;
    const theme = PLAYER_THEMES[currP] || PLAYER_THEMES[0];
    if (turnInfo) {
        turnInfo.className = `badge ${theme.badgeClass}`;
        turnInfo.textContent = `Player ${currP} (${theme.color})`;
    }

    // 3. Dice Info & 3D Dice Roll Box
    const diceInfo = document.getElementById('diceInfo');
    const diceBox = document.getElementById('diceBox3D');

    if (diceInfo) {
        diceInfo.textContent = state.dice_value ? `Roll: ${state.dice_value}` : 'Roll: -';
    }

    if (diceBox) {
        const diceDotsMap = { 1: '⚀', 2: '⚁', 3: '⚂', 4: '⚃', 5: '⚄', 6: '⚅' };
        diceBox.textContent = state.dice_value ? (diceDotsMap[state.dice_value] || state.dice_value) : '-';
    }

    // 4. Turn Count
    const turnBadge = document.getElementById('turnCountBadge');
    if (turnBadge) {
        turnBadge.textContent = `Turn: ${state.turn_count}`;
    }

    // 5. Update 4-Player Statistics
    if (state.players && state.players.length >= 4) {
        state.players.forEach((p, idx) => {
            const prefix = `p${idx}`;
            const yardEl = document.getElementById(`${prefix}Yard`);
            const trackEl = document.getElementById(`${prefix}Track`);
            const goalEl = document.getElementById(`${prefix}Goal`);
            const capsEl = document.getElementById(`${prefix}Caps`);
            const rewEl = document.getElementById(`${prefix}Rew`);
            const ctrlEl = document.getElementById(`${prefix}CtrlBadge`);

            if (yardEl) yardEl.textContent = p.yard_count;
            if (trackEl) trackEl.textContent = p.track_count;
            if (goalEl) goalEl.textContent = p.finished_count;
            if (capsEl) capsEl.textContent = p.captures;
            if (rewEl) rewEl.textContent = (p.reward >= 0 ? '+' : '') + p.reward.toFixed(2);
            if (ctrlEl) ctrlEl.textContent = p.controller;
        });
    }

    // 6. Terminal State & Winner Banner
    const terminalBanner = document.getElementById('terminalBanner');
    const stepBtn = document.getElementById('btnStep');
    const autoBtn = document.getElementById('btnAutoPlay');

    if (state.is_terminal) {
        const winnerText = state.winner !== null
            ? `🏆 Player ${state.winner} (${PLAYER_THEMES[state.winner]?.color}) Won Match!`
            : 'Match Concluded (Truncated / Draw)';
        if (terminalBanner) {
            terminalBanner.className = 'badge badge-rose';
            terminalBanner.textContent = winnerText;
        }
        if (stepBtn) stepBtn.disabled = true;
        if (autoBtn) autoBtn.disabled = true;
    } else {
        if (terminalBanner) {
            terminalBanner.className = 'badge badge-emerald';
            terminalBanner.textContent = 'In Progress';
        }
        if (stepBtn) stepBtn.disabled = false;
        if (autoBtn) autoBtn.disabled = false;
    }

    // 7. Legal Actions Buttons for Current Player
    const legalContainer = document.getElementById('legalActionsContainer');
    const actionLabel = document.getElementById('legalActionLabel');
    if (actionLabel) {
        actionLabel.textContent = `Player ${currP} (${theme.color}) Legal Moves (Dice ${state.dice_value}):`;
    }

    if (legalContainer) {
        legalContainer.innerHTML = '';
        if (state.legal_actions && state.legal_actions.length > 0 && !state.is_terminal) {
            state.legal_actions.forEach(action => {
                const btn = document.createElement('button');
                btn.className = 'btn btn-secondary';
                btn.style.fontSize = '12px';
                btn.style.padding = '5px 12px';
                btn.style.borderColor = theme.hex;
                btn.textContent = `Move P${currP} Token ${action}`;
                btn.onclick = () => playerAction(currP, action);
                legalContainer.appendChild(btn);
            });
        } else {
            legalContainer.innerHTML = `<span style="font-size: 12px; color: var(--text-muted);">No legal moves available for Player ${currP} with roll ${state.dice_value}.</span>`;
        }
    }

    // 8. Live 4-Player Telemetry Table
    const tbody = document.getElementById('turnHistoryBody');
    if (tbody && state.recent_history) {
        if (state.recent_history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">Match initialized. Awaiting first turn execution.</td></tr>';
        } else {
            tbody.innerHTML = '';
            const rev = [...state.recent_history].reverse();
            rev.forEach(item => {
                const tr = document.createElement('tr');
                const pColor = item.color || 'Red';
                const colorBadgeClass = pColor === 'Red' ? 'badge-rose' : pColor === 'Green' ? 'badge-emerald' : pColor === 'Yellow' ? 'badge-amber' : 'badge-blue';

                tr.innerHTML = `
                    <td><strong>${item.turn}</strong></td>
                    <td>${item.player}</td>
                    <td><span class="badge ${colorBadgeClass}">${item.color}</span></td>
                    <td><span class="badge badge-blue">${item.dice}</span></td>
                    <td><strong>${item.action}</strong></td>
                    <td><code>${item.reward}</code></td>
                    <td><span class="badge ${item.capture === 'Yes' ? 'badge-rose' : ''}">${item.capture}</span></td>
                    <td><span class="badge ${item.bonus === 'Yes' ? 'badge-indigo' : ''}">${item.bonus}</span></td>
                    <td>${item.status}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    }
}
