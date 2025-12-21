/**
 * app.js
 * Main orchestration for RLAN Interactive Demo
 * Initializes all components and handles interactions
 */

// ============================================
// Global State
// ============================================
const AppState = {
    currentSection: 1,
    puzzles: [],
    currentPuzzleIndex: 0,
    trainingData: null,
    isInitialized: false
};

// ============================================
// Component Instances
// ============================================
let gridRenderer = null;
let heatmapOverlay = null;
let coordinateToggle = null;
let solverStepper = null;
let chartPanel = null;
let architectureViz = null;
let presenterMode = null;

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 RLAN Demo initializing...');
    
    try {
        // Load data
        await loadDemoData();
        
        // Initialize components
        initHeroSection();
        initARCSection();
        initArchitectureSection();
        initVisualWalkthroughSection();  // NEW: Visual walkthrough with Easy/Medium/Hard examples
        initDSCSection();
        initSolverSection();
        initTrainingSection();
        initComparisonSection();
        initParameterEfficiencySection();  // NEW: Parameter comparison section
        initFutureSection();
        
        // Initialize presenter mode
        presenterMode = new PresenterMode();
        
        // Setup navigation
        setupNavigation();
        
        // Setup collapsibles
        setupCollapsibles();
        
        // Render KaTeX equations
        renderEquations();
        
        AppState.isInitialized = true;
        console.log('✅ RLAN Demo initialized successfully');
        
    } catch (error) {
        console.error('❌ Error initializing demo:', error);
    }
});

// ============================================
// Data Loading
// ============================================
async function loadDemoData() {
    try {
        // Try to load from JSON files
        const puzzlesResponse = await fetch('data/puzzles.json');
        if (puzzlesResponse.ok) {
            AppState.puzzles = await puzzlesResponse.json();
        } else {
            AppState.puzzles = getDefaultPuzzles();
        }
        
        const trainingResponse = await fetch('data/training_metrics.json');
        if (trainingResponse.ok) {
            const rawData = await trainingResponse.json();
            // Flatten the nested structure for ChartPanel
            AppState.trainingData = {
                epochs: rawData.epochs,
                trainLoss: rawData.metrics.trainLoss,
                valLoss: rawData.metrics.valLoss,
                cellAccuracy: rawData.metrics.cellAccuracy,
                exactMatch: rawData.metrics.exactMatch,
                solverImprovement: rawData.metrics.solverImprovement,
                attentionEntropy: rawData.metrics.attentionEntropy
            };
        } else {
            AppState.trainingData = ChartPanel.getSampleTrainingData();
        }
        
        console.log('📊 Demo data loaded');
    } catch (error) {
        console.warn('⚠️ Using default demo data:', error);
        AppState.puzzles = getDefaultPuzzles();
        AppState.trainingData = ChartPanel.getSampleTrainingData();
    }
}

// ============================================
// Section Initializers
// ============================================

/**
 * Hero Section - Animated grid display
 */
function initHeroSection() {
    const container = document.getElementById('hero-grid-container');
    if (!container) return;
    
    // Create an animated demo grid
    const demoGrid = [
        [0, 0, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 2, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 1, 0]
    ];
    
    gridRenderer = new GridRenderer(container, {
        cellSize: 40,
        className: 'large animate-glow'
    });
    gridRenderer.render(demoGrid);
    
    // Add subtle animation
    animateHeroGrid(container, demoGrid);
}

function animateHeroGrid(container, baseGrid) {
    // Cycle through slight variations
    let frame = 0;
    
    setInterval(() => {
        frame = (frame + 1) % 10;
        // Subtle pulse effect handled by CSS
    }, 2000);
}

/**
 * ARC Section - Puzzle sandbox
 */
function initARCSection() {
    const puzzleSelect = document.getElementById('puzzle-select');
    const showSolutionBtn = document.getElementById('show-solution-btn');
    
    if (puzzleSelect) {
        puzzleSelect.addEventListener('change', (e) => {
            AppState.currentPuzzleIndex = parseInt(e.target.value);
            renderCurrentPuzzle();
        });
    }
    
    if (showSolutionBtn) {
        showSolutionBtn.addEventListener('click', () => {
            const outputGrid = document.getElementById('test-output-grid');
            if (outputGrid) {
                outputGrid.classList.toggle('revealed');
                showSolutionBtn.textContent = outputGrid.classList.contains('revealed') 
                    ? 'Hide Solution' 
                    : 'Show Solution';
            }
        });
    }
    
    // Render initial puzzle
    renderCurrentPuzzle();
}

function renderCurrentPuzzle() {
    const puzzle = AppState.puzzles[AppState.currentPuzzleIndex];
    if (!puzzle) return;
    
    // Render training examples
    const examplePairs = document.getElementById('example-pairs');
    if (examplePairs) {
        examplePairs.innerHTML = '';
        
        puzzle.train.forEach((example, i) => {
            const pairDiv = document.createElement('div');
            pairDiv.className = 'example-pair';
            
            const inputContainer = document.createElement('div');
            inputContainer.className = 'example-input';
            const inputLabel = document.createElement('div');
            inputLabel.className = 'example-label';
            inputLabel.textContent = `Ex ${i + 1} Input`;
            inputLabel.style.cssText = 'font-size: 11px; color: #a0a0b0; margin-bottom: 4px; text-align: center;';
            inputContainer.appendChild(inputLabel);
            
            const inputGrid = document.createElement('div');
            new GridRenderer(inputGrid, { cellSize: 20, className: 'small' }).render(example.input);
            inputContainer.appendChild(inputGrid);
            
            const arrow = document.createElement('span');
            arrow.className = 'arrow';
            arrow.textContent = '→';
            
            const outputContainer = document.createElement('div');
            outputContainer.className = 'example-output';
            const outputLabel = document.createElement('div');
            outputLabel.className = 'example-label';
            outputLabel.textContent = 'Output';
            outputLabel.style.cssText = 'font-size: 11px; color: #a0a0b0; margin-bottom: 4px; text-align: center;';
            outputContainer.appendChild(outputLabel);
            
            const outputGrid = document.createElement('div');
            new GridRenderer(outputGrid, { cellSize: 20, className: 'small' }).render(example.output);
            outputContainer.appendChild(outputGrid);
            
            pairDiv.appendChild(inputContainer);
            pairDiv.appendChild(arrow);
            pairDiv.appendChild(outputContainer);
            examplePairs.appendChild(pairDiv);
        });
    }
    
    // Render test input
    const testInputGrid = document.getElementById('test-input-grid');
    if (testInputGrid) {
        testInputGrid.innerHTML = '';
        new GridRenderer(testInputGrid, { cellSize: 24 }).render(puzzle.test.input);
    }
    
    // Render test output (hidden initially)
    const testOutputGrid = document.getElementById('test-output-grid');
    if (testOutputGrid) {
        testOutputGrid.innerHTML = '';
        testOutputGrid.classList.remove('revealed');
        new GridRenderer(testOutputGrid, { cellSize: 24 }).render(puzzle.test.output);
        
        // Reset button text
        const btn = document.getElementById('show-solution-btn');
        if (btn) btn.textContent = 'Show Solution';
    }
}

/**
 * Architecture Section - Module cards
 */
function initArchitectureSection() {
    // Setup module card expand buttons
    const moduleCards = document.querySelectorAll('.module-card');
    
    moduleCards.forEach(card => {
        const expandBtn = card.querySelector('.expand-btn');
        const details = card.querySelector('.module-details');
        
        if (expandBtn && details) {
            expandBtn.addEventListener('click', () => {
                details.classList.toggle('hidden');
                expandBtn.textContent = details.classList.contains('hidden') 
                    ? 'Deep Dive ▼' 
                    : 'Collapse ▲';
            });
        }
    });
    
    // Render architecture diagram
    const archDiagram = document.getElementById('architecture-diagram');
    if (archDiagram) {
        renderArchitectureDiagram(archDiagram);
    }
}

function renderArchitectureDiagram(container) {
    // Simple flow diagram using SVG
    container.innerHTML = `
        <svg viewBox="0 0 800 200" width="100%" height="200" style="max-width: 800px;">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
                </marker>
            </defs>
            
            <!-- Input -->
            <g transform="translate(20, 80)">
                <rect width="80" height="50" rx="6" fill="#1e1e35" stroke="#707080"/>
                <text x="40" y="30" fill="#f0f0f5" font-size="11" text-anchor="middle">Input</text>
            </g>
            
            <!-- Arrow -->
            <line x1="105" y1="105" x2="130" y2="105" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowhead)"/>
            
            <!-- Context Encoder -->
            <g transform="translate(135, 70)">
                <rect width="100" height="70" rx="6" fill="#1e1e35" stroke="#6366f1" stroke-width="2"/>
                <text x="50" y="30" fill="#6366f1" font-size="11" text-anchor="middle">Context</text>
                <text x="50" y="48" fill="#6366f1" font-size="11" text-anchor="middle">Encoder</text>
            </g>
            
            <!-- Arrow -->
            <line x1="240" y1="105" x2="265" y2="105" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowhead)"/>
            
            <!-- DSC + MSRE -->
            <g transform="translate(270, 70)">
                <rect width="100" height="70" rx="6" fill="#1e1e35" stroke="#8b5cf6" stroke-width="2"/>
                <text x="50" y="30" fill="#8b5cf6" font-size="11" text-anchor="middle">DSC +</text>
                <text x="50" y="48" fill="#8b5cf6" font-size="11" text-anchor="middle">MSRE</text>
            </g>
            
            <!-- Arrow -->
            <line x1="375" y1="105" x2="400" y2="105" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowhead)"/>
            
            <!-- LCR + SPH -->
            <g transform="translate(405, 70)">
                <rect width="100" height="70" rx="6" fill="#1e1e35" stroke="#f59e0b" stroke-width="2"/>
                <text x="50" y="30" fill="#f59e0b" font-size="11" text-anchor="middle">LCR +</text>
                <text x="50" y="48" fill="#f59e0b" font-size="11" text-anchor="middle">SPH</text>
            </g>
            
            <!-- Arrow -->
            <line x1="510" y1="105" x2="535" y2="105" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowhead)"/>
            
            <!-- Recursive Solver -->
            <g transform="translate(540, 60)">
                <rect width="130" height="90" rx="8" fill="#1e1e35" stroke="#6366f1" stroke-width="3"/>
                <text x="65" y="35" fill="#6366f1" font-size="12" font-weight="bold" text-anchor="middle">Recursive</text>
                <text x="65" y="55" fill="#6366f1" font-size="12" font-weight="bold" text-anchor="middle">Solver</text>
                <text x="65" y="75" fill="#22c55e" font-size="10" text-anchor="middle">🔄 ConvGRU</text>
            </g>
            
            <!-- Arrow -->
            <line x1="675" y1="105" x2="700" y2="105" stroke="#22c55e" stroke-width="2" marker-end="url(#arrowhead)"/>
            
            <!-- Output -->
            <g transform="translate(705, 80)">
                <rect width="80" height="50" rx="6" fill="#1e1e35" stroke="#22c55e" stroke-width="2"/>
                <text x="40" y="30" fill="#22c55e" font-size="11" text-anchor="middle">Output</text>
            </g>
        </svg>
    `;
}

/**
 * DSC & Coordinates Section
 */
function initDSCSection() {
    const toggleBtn = document.getElementById('toggle-coords');
    const modeLabel = document.getElementById('coord-mode-label');
    const gridContainer = document.getElementById('coord-demo-grid');
    const heatmapContainer = document.getElementById('dsc-heatmap');
    
    if (!gridContainer) return;
    
    // Sample grid for demo
    const demoGrid = [
        [0, 0, 0, 1, 0],
        [0, 0, 1, 2, 1],
        [0, 1, 2, 3, 2],
        [0, 0, 1, 2, 1],
        [0, 0, 0, 1, 0]
    ];
    
    // Initialize coordinate toggle
    coordinateToggle = new CoordinateToggle(gridContainer, {
        cellSize: 36,
        onModeChange: (mode) => {
            if (modeLabel) {
                modeLabel.textContent = `Mode: ${mode.charAt(0).toUpperCase() + mode.slice(1)}`;
            }
        }
    });
    coordinateToggle.init(demoGrid, { x: 3, y: 2 }); // Anchor at center of pattern
    
    // Toggle button
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const newMode = coordinateToggle.toggle();
            toggleBtn.textContent = `Toggle: ${newMode === 'absolute' ? 'Absolute' : 'Relative'} ↔ ${newMode === 'absolute' ? 'Relative' : 'Absolute'}`;
        });
    }
    
    // Render heatmap
    if (heatmapContainer) {
        const proxyWeights = HeatmapOverlay.generateProxyAttention(demoGrid);
        heatmapOverlay = new HeatmapOverlay(heatmapContainer, {
            cellSize: 32,
            colorScale: 'attention',
            showValues: false
        });
        heatmapOverlay.render(proxyWeights);
    }
}

/**
 * Solver Section - Step through iterations
 */
function initSolverSection() {
    const stepperContainer = document.querySelector('.solver-stepper');
    if (!stepperContainer) return;
    
    // Clear existing content from HTML template
    stepperContainer.innerHTML = '<h4>Step Through Solver Iterations</h4><p class="proxy-note">⚠️ Proxy visualization showing simulated refinement. See docs for logging real iteration states.</p>';
    
    const stepperDiv = document.createElement('div');
    stepperDiv.id = 'solver-stepper-container';
    stepperContainer.appendChild(stepperDiv);
    
    // Generate proxy iteration data
    const inputGrid = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ];
    
    const outputGrid = [
        [0, 0, 2, 0, 0],
        [0, 2, 3, 2, 0],
        [2, 3, 3, 3, 2],
        [0, 2, 3, 2, 0],
        [0, 0, 2, 0, 0]
    ];
    
    const iterations = SolverStepper.generateProxyIterations(inputGrid, outputGrid, 6);
    
    solverStepper = new SolverStepper(stepperDiv, {
        cellSize: 32,
        autoPlaySpeed: 1200,
        onStepChange: (step, iteration) => {
            console.log(`Solver step ${step}: confidence=${iteration.confidence.toFixed(3)}`);
        }
    });
    solverStepper.init(iterations);
}

/**
 * Training Section - Charts
 */
function initTrainingSection() {
    chartPanel = new ChartPanel(document.getElementById('training'));
    
    // Wait for Plotly to be available
    if (typeof Plotly !== 'undefined') {
        chartPanel.renderAllCharts(AppState.trainingData);
    } else {
        // Retry after a short delay
        setTimeout(() => {
            if (typeof Plotly !== 'undefined') {
                chartPanel.renderAllCharts(AppState.trainingData);
            } else {
                console.warn('Plotly not loaded - charts will not render');
            }
        }, 1000);
    }
}

/**
 * Comparison Section - RLAN vs Tiny Recursive Model
 */
function initComparisonSection() {
    const archComparison = document.getElementById('arch-comparison');
    
    if (archComparison) {
        architectureViz = new ArchitectureViz(archComparison, {
            width: 900,
            height: 350
        });
        architectureViz.render();
    }
    
    // Setup comparison buttons
    const comparisonBtns = document.querySelectorAll('.compare-btn');
    comparisonBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            comparisonBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Get aspect from button id
            const aspect = btn.id.replace('highlight-', '');
            
            // Update diagram
            if (architectureViz) {
                architectureViz.highlightAspect(aspect);
            }
            
            // Highlight table rows
            highlightTableRows(aspect);
        });
    });
}

function highlightTableRows(aspect) {
    const rows = document.querySelectorAll('.comparison-table tbody tr');
    rows.forEach(row => {
        const rowAspect = row.dataset.aspect;
        row.classList.toggle('highlight', rowAspect === aspect);
    });
}

/**
 * Future Section - LLM Integration
 */
function initFutureSection() {
    const diagramContainer = document.getElementById('llm-integration-diagram');
    
    if (diagramContainer) {
        ArchitectureViz.createLLMIntegrationDiagram(diagramContainer);
    }
}

/**
 * Visual Walkthrough Section - Easy/Medium/Hard examples
 */
function initVisualWalkthroughSection() {
    // Object Movement Example
    initEasyExample();
}

function initEasyExample() {
    // Input: Grey square at (0,0), Red marker at (3,3)
    // Output: Grey square moved to (3,3)
    const easyInput = [
        [5, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 2]
    ];
    
    const easyOutput = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 5]
    ];
    
    const easyInputEl = document.getElementById('easy-input');
    const easyOutputEl = document.getElementById('easy-output');
    
    if (easyInputEl) {
        new GridRenderer(easyInputEl, { cellSize: 36, className: 'walkthrough' }).render(easyInput);
    }
    
    if (easyOutputEl) {
        new GridRenderer(easyOutputEl, { cellSize: 36, className: 'walkthrough' }).render(easyOutput);
    }
    
    // DSC Visualization - show attention on red pixel
    const dscViz = document.getElementById('easy-dsc-viz');
    if (dscViz) {
        // Create attention heatmap focused on (3,3)
        const attentionWeights = [
            [0.01, 0.01, 0.01, 0.02],
            [0.01, 0.02, 0.03, 0.05],
            [0.02, 0.03, 0.08, 0.15],
            [0.02, 0.05, 0.15, 0.85]  // High attention at red pixel location
        ];
        
        new HeatmapOverlay(dscViz, { 
            cellSize: 30, 
            colorScale: 'attention',
            showValues: true 
        }).render(attentionWeights);
    }
    
    // MSRE Visualization - coordinate transform
    const msreViz = document.getElementById('easy-msre-viz');
    if (msreViz) {
        renderCoordinateTransform(msreViz, 4, 4, { x: 3, y: 3 });
    }
}

function initMediumExample() {
    // Input: 2x2 pattern
    const mediumInput = [
        [3, 2],
        [7, 8]
    ];
    
    // Output: 6x6 tiled pattern
    const mediumOutput = [
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8],
        [2, 3, 2, 3, 2, 3],
        [8, 7, 8, 7, 8, 7],
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8]
    ];
    
    const mediumInputEl = document.getElementById('medium-input');
    const mediumOutputEl = document.getElementById('medium-output');
    
    if (mediumInputEl) {
        new GridRenderer(mediumInputEl, { cellSize: 40, className: 'walkthrough' }).render(mediumInput);
    }
    
    if (mediumOutputEl) {
        new GridRenderer(mediumOutputEl, { cellSize: 28, className: 'walkthrough' }).render(mediumOutput);
    }
}

function initHardExample() {
    // Case A: Symmetric input -> Vertical flip
    const hardInputA = [
        [2, 0, 2],
        [1, 1, 1],
        [0, 0, 0]
    ];
    
    const hardOutputA = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 0, 2]
    ];
    
    // Case B: Asymmetric input -> Horizontal flip
    const hardInputB = [
        [2, 0, 0],
        [1, 1, 0],
        [0, 0, 0]
    ];
    
    const hardOutputB = [
        [0, 0, 2],
        [0, 1, 1],
        [0, 0, 0]
    ];
    
    const hardInputAEl = document.getElementById('hard-input-a');
    const hardOutputAEl = document.getElementById('hard-output-a');
    const hardInputBEl = document.getElementById('hard-input-b');
    const hardOutputBEl = document.getElementById('hard-output-b');
    
    if (hardInputAEl) {
        new GridRenderer(hardInputAEl, { cellSize: 30, className: 'walkthrough' }).render(hardInputA);
    }
    if (hardOutputAEl) {
        new GridRenderer(hardOutputAEl, { cellSize: 30, className: 'walkthrough' }).render(hardOutputA);
    }
    if (hardInputBEl) {
        new GridRenderer(hardInputBEl, { cellSize: 30, className: 'walkthrough' }).render(hardInputB);
    }
    if (hardOutputBEl) {
        new GridRenderer(hardOutputBEl, { cellSize: 30, className: 'walkthrough' }).render(hardOutputB);
    }
}

function renderCoordinateTransform(container, rows, cols, anchor) {
    container.innerHTML = '';
    
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display: flex; gap: 2rem; flex-wrap: wrap; align-items: flex-start;';
    
    // Absolute coordinates
    const absDiv = document.createElement('div');
    absDiv.innerHTML = '<h5 style="margin-bottom: 0.5rem; color: #a0a0b0;">Absolute</h5>';
    const absGrid = document.createElement('div');
    absGrid.style.cssText = 'display: grid; gap: 2px;';
    absGrid.style.gridTemplateColumns = `repeat(${cols}, 45px)`;
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const cell = document.createElement('div');
            cell.textContent = `(${i},${j})`;
            cell.style.cssText = 'width: 45px; height: 30px; display: flex; align-items: center; justify-content: center; background: #252542; border-radius: 4px; font-size: 10px; color: #a0a0b0;';
            if (i === anchor.y && j === anchor.x) {
                cell.style.background = '#f59e0b30';
                cell.style.border = '2px solid #f59e0b';
            }
            absGrid.appendChild(cell);
        }
    }
    absDiv.appendChild(absGrid);
    
    // Relative coordinates
    const relDiv = document.createElement('div');
    relDiv.innerHTML = '<h5 style="margin-bottom: 0.5rem; color: #6366f1;">Relative to (3,3)</h5>';
    const relGrid = document.createElement('div');
    relGrid.style.cssText = 'display: grid; gap: 2px;';
    relGrid.style.gridTemplateColumns = `repeat(${cols}, 45px)`;
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const relI = i - anchor.y;
            const relJ = j - anchor.x;
            const cell = document.createElement('div');
            cell.textContent = `(${relI},${relJ})`;
            cell.style.cssText = 'width: 45px; height: 30px; display: flex; align-items: center; justify-content: center; background: #252542; border-radius: 4px; font-size: 10px; color: #6366f1;';
            if (i === anchor.y && j === anchor.x) {
                cell.style.background = '#f59e0b30';
                cell.style.border = '2px solid #f59e0b';
                cell.style.fontWeight = 'bold';
            }
            relGrid.appendChild(cell);
        }
    }
    relDiv.appendChild(relGrid);
    
    wrapper.appendChild(absDiv);
    wrapper.appendChild(relDiv);
    container.appendChild(wrapper);
}

/**
 * Parameter Efficiency Section
 */
function initParameterEfficiencySection() {
    // The section is mostly static HTML, but we can add interactivity here
    // For example, animate the bar chart
    const paramBars = document.querySelectorAll('#parameter-efficiency .bar-fill');
    
    if (paramBars.length > 0) {
        // Use Intersection Observer to trigger animation when section comes into view
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Animate bars on view
                    paramBars.forEach((bar, index) => {
                        setTimeout(() => {
                            bar.style.transition = 'width 1s ease-out';
                        }, index * 200);
                    });
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.3 });
        
        const section = document.getElementById('parameter-efficiency');
        if (section) {
            observer.observe(section);
        }
    }
}

// ============================================
// Navigation
// ============================================
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const target = document.getElementById(targetId);
            
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Update active nav on scroll
    window.addEventListener('scroll', updateActiveNav);
}

function updateActiveNav() {
    const sections = document.querySelectorAll('.story-section');
    const navLinks = document.querySelectorAll('.nav-links a');
    
    let currentSection = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        const sectionHeight = section.offsetHeight;
        
        if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
            currentSection = section.id;
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentSection}`) {
            link.classList.add('active');
        }
    });
}

// ============================================
// Collapsibles
// ============================================
function setupCollapsibles() {
    const collapsibles = document.querySelectorAll('.collapsible-toggle');
    
    collapsibles.forEach(toggle => {
        toggle.addEventListener('click', () => {
            toggle.classList.toggle('active');
            const content = toggle.nextElementSibling;
            if (content) {
                content.classList.toggle('hidden');
            }
        });
    });
}

// ============================================
// KaTeX Rendering
// ============================================
function renderEquations() {
    // Check if KaTeX is available
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false }
            ],
            throwOnError: false
        });
        console.log('📐 KaTeX equations rendered');
    } else {
        // Retry after a short delay for CDN loading
        setTimeout(() => {
            if (typeof renderMathInElement !== 'undefined') {
                renderMathInElement(document.body, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '$', right: '$', display: false }
                    ],
                    throwOnError: false
                });
                console.log('📐 KaTeX equations rendered (delayed)');
            }
        }, 1500);
    }
}

// ============================================
// Default Puzzles Data
// ============================================
function getDefaultPuzzles() {
    return [
        {
            name: "Pattern Fill",
            train: [
                {
                    input: [[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],
                    output: [[0,0,0,0,0],[0,1,1,1,0],[0,1,2,1,0],[0,1,1,1,0],[0,0,0,0,0]]
                },
                {
                    input: [[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]],
                    output: [[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,2,2,1,0],[0,1,2,2,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]]
                }
            ],
            test: {
                input: [[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,1,0,0,0,1,0],[0,1,0,0,0,1,0],[0,1,0,0,0,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]],
                output: [[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,1,2,2,2,1,0],[0,1,2,2,2,1,0],[0,1,2,2,2,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]]
            }
        },
        {
            name: "Color Mapping",
            train: [
                {
                    input: [[1,0,0],[0,0,0],[0,0,0]],
                    output: [[2,0,0],[0,0,0],[0,0,0]]
                },
                {
                    input: [[0,1,0],[0,1,0],[0,0,0]],
                    output: [[0,2,0],[0,2,0],[0,0,0]]
                }
            ],
            test: {
                input: [[0,0,1],[0,0,1],[0,0,1]],
                output: [[0,0,2],[0,0,2],[0,0,2]]
            }
        },
        {
            name: "Object Counting",
            train: [
                {
                    input: [[1,0,0],[0,0,0],[0,0,1]],
                    output: [[3,3],[3,3]]
                },
                {
                    input: [[1,0,1],[0,1,0],[0,0,0]],
                    output: [[3,3,3],[3,3,3],[3,3,3]]
                }
            ],
            test: {
                input: [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],
                output: [[3,3],[3,3]]
            }
        }
    ];
}

// ============================================
// Utility Functions
// ============================================
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================================
// Export for testing
// ============================================
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AppState,
        getDefaultPuzzles,
        renderCurrentPuzzle
    };
}
