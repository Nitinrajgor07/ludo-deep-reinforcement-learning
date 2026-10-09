/**
 * Master Client-Side JavaScript for PhD Ludo DRL Platform.
 * Framework: FastAPI + Vanilla JS
 * Strict Constraint: 100% Light Theme
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

// Close modal on Escape key
document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});
