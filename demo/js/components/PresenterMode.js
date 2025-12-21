/**
 * PresenterMode.js
 * Manages presenter mode with speaker notes, pacing, and progress tracking
 * Supports 30-second, 2-minute, and 5-minute talk-track variants
 */

class PresenterMode {
    constructor(options = {}) {
        this.options = {
            storageKey: 'rlan-presenter-mode',
            defaultDuration: '5min',
            ...options
        };
        
        this.isActive = false;
        this.currentSection = 1;
        this.totalSections = 8;
        this.selectedDuration = this.options.defaultDuration;
        
        // DOM elements
        this.toggleBtn = null;
        this.panel = null;
        this.notesContainer = null;
        this.progressBar = null;
        this.progressText = null;
        this.pacingGuide = null;
        this.durationSelect = null;
        
        // Pacing guides for each duration
        this.pacingGuides = {
            '30s': {
                totalTime: 30,
                perSection: 4,
                description: 'Quick pitch: ~4 seconds per section'
            },
            '2min': {
                totalTime: 120,
                perSection: 15,
                description: 'Overview: ~15 seconds per section'
            },
            '5min': {
                totalTime: 300,
                perSection: 38,
                description: 'Deep dive: ~38 seconds per section'
            }
        };
        
        this.init();
    }
    
    /**
     * Initialize presenter mode
     */
    init() {
        // Check for saved state
        const savedState = localStorage.getItem(this.options.storageKey);
        if (savedState) {
            try {
                const state = JSON.parse(savedState);
                this.selectedDuration = state.duration || this.options.defaultDuration;
            } catch (e) {
                console.warn('PresenterMode: Failed to parse saved state');
            }
        }
        
        this.setupElements();
        this.setupEventListeners();
    }
    
    /**
     * Setup DOM element references
     */
    setupElements() {
        this.toggleBtn = document.getElementById('presenter-toggle');
        this.panel = document.getElementById('presenter-panel');
        this.notesContainer = document.getElementById('speaker-notes');
        this.progressBar = document.querySelector('#progress-bar');
        this.progressText = document.getElementById('progress-text');
        this.pacingGuide = document.getElementById('pacing-guide');
        this.durationSelect = document.getElementById('talk-duration');
        
        // Set initial duration
        if (this.durationSelect) {
            this.durationSelect.value = this.selectedDuration;
        }
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Toggle button
        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this.toggle());
        }
        
        // Duration selector
        if (this.durationSelect) {
            this.durationSelect.addEventListener('change', (e) => {
                this.setDuration(e.target.value);
            });
        }
        
        // Scroll listener for section tracking
        window.addEventListener('scroll', () => this.updateOnScroll());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'p' && e.ctrlKey) {
                e.preventDefault();
                this.toggle();
            }
        });
    }
    
    /**
     * Toggle presenter mode
     */
    toggle() {
        this.isActive = !this.isActive;
        
        if (this.panel) {
            this.panel.classList.toggle('hidden', !this.isActive);
        }
        
        document.body.classList.toggle('presenter-mode', this.isActive);
        
        if (this.isActive) {
            this.updateNotes();
            this.updateProgress();
            this.updatePacing();
        }
        
        // Save state
        this.saveState();
    }
    
    /**
     * Set talk duration
     */
    setDuration(duration) {
        if (!this.pacingGuides[duration]) {
            console.warn('PresenterMode: Invalid duration', duration);
            return;
        }
        
        this.selectedDuration = duration;
        this.updateNotes();
        this.updatePacing();
        this.saveState();
    }
    
    /**
     * Update speaker notes for current section
     */
    updateNotes() {
        if (!this.notesContainer) return;
        
        const section = document.querySelector(`.story-section[data-section="${this.currentSection}"]`);
        if (!section) return;
        
        const notesElement = section.querySelector('.speaker-notes');
        if (!notesElement) {
            this.notesContainer.innerHTML = '<em>No speaker notes for this section.</em>';
            return;
        }
        
        // Get notes for selected duration
        const durationKey = `data-duration-${this.selectedDuration}`;
        const notes = notesElement.getAttribute(durationKey) || 'No notes available for this duration.';
        
        this.notesContainer.innerHTML = `
            <div class="notes-content">
                <strong>Section ${this.currentSection}:</strong>
                <p>${notes}</p>
            </div>
        `;
    }
    
    /**
     * Update progress indicator
     */
    updateProgress() {
        const progress = (this.currentSection / this.totalSections) * 100;
        
        if (this.progressBar) {
            // Update the pseudo-element width via CSS variable
            this.progressBar.style.setProperty('--progress', `${progress}%`);
            
            // Or update the ::before directly if using inline style approach
            const fill = this.progressBar.querySelector('.progress-fill') || this.progressBar;
            if (fill.style) {
                fill.style.width = `${progress}%`;
            }
        }
        
        if (this.progressText) {
            this.progressText.textContent = `Section ${this.currentSection} of ${this.totalSections}`;
        }
    }
    
    /**
     * Update pacing guide
     */
    updatePacing() {
        if (!this.pacingGuide) return;
        
        const guide = this.pacingGuides[this.selectedDuration];
        const elapsed = (this.currentSection - 1) * guide.perSection;
        const remaining = (this.totalSections - this.currentSection) * guide.perSection;
        
        this.pacingGuide.innerHTML = `
            <div class="pacing-info">
                <div class="pacing-stat">
                    <span class="pacing-label">Target per section:</span>
                    <span class="pacing-value">${guide.perSection}s</span>
                </div>
                <div class="pacing-stat">
                    <span class="pacing-label">Remaining:</span>
                    <span class="pacing-value">${this.formatTime(remaining)}</span>
                </div>
                <div class="pacing-description">${guide.description}</div>
            </div>
        `;
    }
    
    /**
     * Format time in seconds to readable string
     */
    formatTime(seconds) {
        if (seconds < 60) {
            return `${seconds}s`;
        }
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
    }
    
    /**
     * Update on scroll - detect current section
     */
    updateOnScroll() {
        const sections = document.querySelectorAll('.story-section');
        const scrollPos = window.scrollY + window.innerHeight / 3;
        
        let currentSection = 1;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionNum = parseInt(section.dataset.section);
            
            if (scrollPos >= sectionTop && sectionNum) {
                currentSection = sectionNum;
            }
        });
        
        if (currentSection !== this.currentSection) {
            this.currentSection = currentSection;
            this.updateNavHighlight();
            
            if (this.isActive) {
                this.updateNotes();
                this.updateProgress();
                this.updatePacing();
            }
        }
    }
    
    /**
     * Update navigation highlight
     */
    updateNavHighlight() {
        const navLinks = document.querySelectorAll('.nav-links a');
        navLinks.forEach((link, index) => {
            link.classList.toggle('active', index + 1 === this.currentSection);
        });
    }
    
    /**
     * Go to specific section
     */
    goToSection(sectionNum) {
        const section = document.querySelector(`.story-section[data-section="${sectionNum}"]`);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth' });
        }
    }
    
    /**
     * Save state to localStorage
     */
    saveState() {
        const state = {
            duration: this.selectedDuration
        };
        localStorage.setItem(this.options.storageKey, JSON.stringify(state));
    }
    
    /**
     * Get all speaker notes for export/printing
     */
    getAllNotes() {
        const notes = [];
        const sections = document.querySelectorAll('.story-section');
        
        sections.forEach(section => {
            const sectionNum = section.dataset.section;
            const notesElement = section.querySelector('.speaker-notes');
            
            if (notesElement) {
                notes.push({
                    section: sectionNum,
                    '30s': notesElement.getAttribute('data-duration-30s'),
                    '2min': notesElement.getAttribute('data-duration-2min'),
                    '5min': notesElement.getAttribute('data-duration-5min')
                });
            }
        });
        
        return notes;
    }
    
    /**
     * Export notes as markdown
     */
    exportNotesAsMarkdown() {
        const notes = this.getAllNotes();
        let markdown = '# RLAN Demo Speaker Notes\n\n';
        
        notes.forEach(note => {
            markdown += `## Section ${note.section}\n\n`;
            markdown += `### 30-second pitch\n${note['30s']}\n\n`;
            markdown += `### 2-minute overview\n${note['2min']}\n\n`;
            markdown += `### 5-minute deep dive\n${note['5min']}\n\n`;
            markdown += '---\n\n';
        });
        
        return markdown;
    }
    
    /**
     * Create a timer display for live presentations
     */
    createTimer(container) {
        const timerEl = document.createElement('div');
        timerEl.className = 'presenter-timer';
        timerEl.innerHTML = `
            <div class="timer-display">00:00</div>
            <div class="timer-controls">
                <button class="timer-start">Start</button>
                <button class="timer-reset">Reset</button>
            </div>
        `;
        
        let startTime = null;
        let timerInterval = null;
        const display = timerEl.querySelector('.timer-display');
        
        timerEl.querySelector('.timer-start').addEventListener('click', function() {
            if (timerInterval) {
                clearInterval(timerInterval);
                timerInterval = null;
                this.textContent = 'Start';
            } else {
                if (!startTime) startTime = Date.now();
                timerInterval = setInterval(() => {
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
                    const secs = (elapsed % 60).toString().padStart(2, '0');
                    display.textContent = `${mins}:${secs}`;
                }, 1000);
                this.textContent = 'Pause';
            }
        });
        
        timerEl.querySelector('.timer-reset').addEventListener('click', () => {
            clearInterval(timerInterval);
            timerInterval = null;
            startTime = null;
            display.textContent = '00:00';
            timerEl.querySelector('.timer-start').textContent = 'Start';
        });
        
        container.appendChild(timerEl);
        return timerEl;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PresenterMode;
}
