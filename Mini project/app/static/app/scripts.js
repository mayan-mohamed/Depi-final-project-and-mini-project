document.addEventListener("DOMContentLoaded", function() {
    const images = document.querySelectorAll('.book-cover');

    images.forEach(img => {
        img.addEventListener('error', function() {
            this.src = this.dataset.placeholder; // to handle image errors
        });
    });
});
