/**
 * Master Client-Side JavaScript for PhD Ludo DRL Platform.
 * Custom Researcher Profile Dropdown & Image Zoom Modal Handling.
 */

// Image Zoom Modal Handling
function openModal(imgSrc, caption) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const modalCaption = document.getElementById('modalCaption');
    if (modal && modalImg) {
        modalImg.src = imgSrc;
        if (modalCaption) {
            modalCaption.textContent = caption || '';
        }
        modal.classList.add('active');
    }
}

function closeModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

// Custom Researcher Profile Dropdown Mechanics
document.addEventListener('DOMContentLoaded', function () {
    const userBtn = document.getElementById('userProfileBtn');
    const popover = document.getElementById('customProfileDropdown');

    if (userBtn && popover) {
        // Toggle dropdown popover on click
        userBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            const isOpen = popover.classList.contains('show');
            if (isOpen) {
                closeCustomProfile();
            } else {
                openCustomProfile();
            }
        });

        // Close on click outside
        document.addEventListener('click', function (e) {
            if (!popover.contains(e.target) && !userBtn.contains(e.target)) {
                closeCustomProfile();
            }
        });

        // Close on Escape key press
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                closeModal();
                if (popover.classList.contains('show')) {
                    closeCustomProfile();
                    userBtn.focus();
                }
            }
        });
    }
});

function openCustomProfile() {
    const userBtn = document.getElementById('userProfileBtn');
    const popover = document.getElementById('customProfileDropdown');
    if (userBtn && popover) {
        popover.classList.add('show');
        popover.setAttribute('aria-hidden', 'false');
        userBtn.setAttribute('aria-expanded', 'true');
    }
}

function closeCustomProfile() {
    const userBtn = document.getElementById('userProfileBtn');
    const popover = document.getElementById('customProfileDropdown');
    if (userBtn && popover) {
        popover.classList.remove('show');
        popover.setAttribute('aria-hidden', 'true');
        userBtn.setAttribute('aria-expanded', 'false');
    }
}

// Handle Sign Out / Session Reset Action
async function handleSignOut() {
    closeCustomProfile();
    try {
        const response = await fetch('/api/logout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        if (response.ok) {
            const data = await response.json();
            alert(data.message || 'Session reset successfully.');
            if (data.redirect_url) {
                window.location.href = data.redirect_url;
            }
        } else {
            window.location.href = '/dashboard';
        }
    } catch (err) {
        console.error('Sign out error:', err);
        window.location.href = '/dashboard';
    }
}
