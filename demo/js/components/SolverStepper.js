/**
 * SolverStepper.js
 * Visualizes the recursive solver's iterative refinement process
 * Shows intermediate outputs, confidence, and halting probability
 * Note: Uses PROXY data until real model iteration logs are available
 */

class SolverStepper {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;
        
        this.options = {
            cellSize: options.cellSize || 28,
            maxIterations: options.maxIterations || 6,
            autoPlaySpeed: options.autoPlaySpeed || 1000,
            showTimeline: options.showTimeline !== false,
            ...options
        };
        
        this.currentStep = 0;
        this.iterations = [];
        this.isPlaying = false;
        this.playInterval = null;
        
        this.gridRenderer = null;
        this.controls = {};
        
        // Callbacks
        this.onStepChange = options.onStepChange || null;
    }
    
    /**
     * Initialize with iteration data
     * @param {Object[]} iterations - Array of iteration states
     * Each iteration: { grid: number[][], confidence: number, haltProb: number }
     */
    init(iterations) {
        this.iterations = iterations;
        this.currentStep = 0;
        this.render();
    }
    
    /**
     * Generate PROXY iteration data based on input/output grids
     * Simulates gradual refinement from initial guess to final answer
     */
    static generateProxyIterations(inputGrid, outputGrid, numSteps = 6) {
        const iterations = [];
        const rows = outputGrid.length;
        const cols = outputGrid[0].length;
        
        for (let step = 0; step < numSteps; step++) {
            const progress = step / (numSteps - 1);
            const grid = [];
            
            for (let i = 0; i < rows; i++) {
                grid[i] = [];
                for (let j = 0; j < cols; j++) {
                    // Gradually transition from noise/input-based to output
                    if (step === 0) {
                        // Initial: mostly background with some noise
                        grid[i][j] = Math.random() < 0.1 ? Math.floor(Math.random() * 3) : 0;
                    } else if (step === numSteps - 1) {
                        // Final: exact output
                        grid[i][j] = outputGrid[i][j];
                    } else {
                        // Intermediate: probabilistic mix
                        const correctProb = Math.pow(progress, 1.5);
                        if (Math.random() < correctProb) {
                            grid[i][j] = outputGrid[i][j];
                        } else if (Math.random() < 0.3) {
                            // Random error
                            grid[i][j] = Math.floor(Math.random() * 5);
                        } else {
                            // Previous value or background
                            grid[i][j] = step > 1 ? iterations[step - 1].grid[i][j] : 0;
                        }
                    }
                }
            }
            
            // Calculate proxy confidence (how close to final output)
            let matches = 0;
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    if (grid[i][j] === outputGrid[i][j]) matches++;
                }
            }
            const confidence = matches / (rows * cols);
            
            // Halting probability increases as we get closer to solution
            const haltProb = step === numSteps - 1 ? 0.95 : Math.min(0.9, confidence * 0.8 + step * 0.05);
            
            iterations.push({
                step,
                grid,
                confidence,
                haltProb,
                cellAccuracy: confidence
            });
        }
        
        return iterations;
    }
    
    /**
     * Render the stepper UI
     */
    render() {
        this.container.innerHTML = '';
        
        const wrapper = document.createElement('div');
        wrapper.className = 'solver-stepper-wrapper';
        wrapper.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 20px;
        `;
        
        // Controls
        const controls = this.createControls();
        wrapper.appendChild(controls);
        
        // Main visualization area
        const vizArea = document.createElement('div');
        vizArea.className = 'stepper-viz-area';
        vizArea.style.cssText = `
            display: flex;
            gap: 32px;
            align-items: flex-start;
        `;
        
        // Grid display
        const gridArea = document.createElement('div');
        gridArea.className = 'stepper-grid-area';
        gridArea.id = 'stepper-grid-display';
        vizArea.appendChild(gridArea);
        
        // Metrics panel
        const metricsPanel = this.createMetricsPanel();
        vizArea.appendChild(metricsPanel);
        
        wrapper.appendChild(vizArea);
        
        // Timeline
        if (this.options.showTimeline) {
            const timeline = this.createTimeline();
            wrapper.appendChild(timeline);
        }
        
        this.container.appendChild(wrapper);
        
        // Render initial state
        this.renderCurrentStep();
    }
    
    /**
     * Create control buttons
     */
    createControls() {
        const controls = document.createElement('div');
        controls.className = 'stepper-controls';
        controls.style.cssText = `
            display: flex;
            align-items: center;
            gap: 12px;
        `;
        
        // Previous button
        this.controls.prevBtn = document.createElement('button');
        this.controls.prevBtn.className = 'step-btn';
        this.controls.prevBtn.innerHTML = '← Prev';
        this.controls.prevBtn.onclick = () => this.prevStep();
        
        // Step counter
        this.controls.counter = document.createElement('span');
        this.controls.counter.className = 'step-counter';
        this.controls.counter.style.cssText = `
            min-width: 120px;
            text-align: center;
            color: #a0a0b0;
        `;
        
        // Next button
        this.controls.nextBtn = document.createElement('button');
        this.controls.nextBtn.className = 'step-btn';
        this.controls.nextBtn.innerHTML = 'Next →';
        this.controls.nextBtn.onclick = () => this.nextStep();
        
        // Play button
        this.controls.playBtn = document.createElement('button');
        this.controls.playBtn.className = 'step-btn play-btn';
        this.controls.playBtn.innerHTML = '▶ Auto-Play';
        this.controls.playBtn.onclick = () => this.togglePlay();
        
        // Reset button
        this.controls.resetBtn = document.createElement('button');
        this.controls.resetBtn.className = 'step-btn';
        this.controls.resetBtn.innerHTML = '↺ Reset';
        this.controls.resetBtn.onclick = () => this.reset();
        
        controls.appendChild(this.controls.prevBtn);
        controls.appendChild(this.controls.counter);
        controls.appendChild(this.controls.nextBtn);
        controls.appendChild(this.controls.playBtn);
        controls.appendChild(this.controls.resetBtn);
        
        return controls;
    }
    
    /**
     * Create metrics panel
     */
    createMetricsPanel() {
        const panel = document.createElement('div');
        panel.className = 'stepper-metrics';
        panel.style.cssText = `
            background: #252542;
            border-radius: 12px;
            padding: 20px;
            min-width: 200px;
        `;
        
        // Confidence meter
        const confidenceSection = document.createElement('div');
        confidenceSection.style.marginBottom = '20px';
        confidenceSection.innerHTML = `
            <label style="display: block; font-size: 13px; color: #a0a0b0; margin-bottom: 8px;">
                Cell Accuracy
            </label>
            <div class="meter-bar" style="height: 8px; background: #1a1a2e; border-radius: 4px; overflow: hidden;">
                <div id="confidence-meter-fill" class="meter-fill" style="height: 100%; background: linear-gradient(90deg, #6366f1, #8b5cf6); width: 0%; transition: width 0.3s ease;"></div>
            </div>
            <span id="confidence-value" style="font-size: 24px; font-weight: 700; color: #6366f1;">0%</span>
        `;
        panel.appendChild(confidenceSection);
        
        // Halt probability
        const haltSection = document.createElement('div');
        haltSection.style.marginBottom = '20px';
        haltSection.innerHTML = `
            <label style="display: block; font-size: 13px; color: #a0a0b0; margin-bottom: 8px;">
                Halt Probability
            </label>
            <span id="halt-prob-display" style="font-size: 20px; font-weight: 600; color: #22c55e;">0.00</span>
        `;
        panel.appendChild(haltSection);
        
        // Step description
        const descSection = document.createElement('div');
        descSection.id = 'step-description';
        descSection.style.cssText = `
            font-size: 13px;
            color: #a0a0b0;
            padding-top: 16px;
            border-top: 1px solid #2a2a4a;
        `;
        panel.appendChild(descSection);
        
        return panel;
    }
    
    /**
     * Create timeline visualization
     */
    createTimeline() {
        const timeline = document.createElement('div');
        timeline.className = 'stepper-timeline';
        timeline.style.cssText = `
            background: #252542;
            border-radius: 8px;
            padding: 16px;
        `;
        
        const track = document.createElement('div');
        track.id = 'timeline-markers';
        track.style.cssText = `
            display: flex;
            justify-content: center;
            gap: 8px;
        `;
        
        for (let i = 0; i < this.iterations.length; i++) {
            const marker = document.createElement('div');
            marker.className = 'timeline-marker';
            marker.dataset.step = i;
            marker.textContent = i;
            marker.style.cssText = `
                width: 36px;
                height: 36px;
                border-radius: 50%;
                background: ${i === this.currentStep ? '#6366f1' : '#1a1a2e'};
                border: 2px solid ${i < this.currentStep ? '#22c55e' : i === this.currentStep ? '#6366f1' : '#2a2a4a'};
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.2s ease;
            `;
            
            marker.onclick = () => this.goToStep(i);
            track.appendChild(marker);
        }
        
        timeline.appendChild(track);
        return timeline;
    }
    
    /**
     * Render the current step
     */
    renderCurrentStep() {
        if (!this.iterations.length) return;
        
        const iteration = this.iterations[this.currentStep];
        
        // Update grid
        const gridContainer = this.container.querySelector('#stepper-grid-display');
        if (gridContainer) {
            gridContainer.innerHTML = '';
            const renderer = new GridRenderer(gridContainer, { 
                cellSize: this.options.cellSize 
            });
            renderer.render(iteration.grid);
        }
        
        // Update counter
        if (this.controls.counter) {
            this.controls.counter.textContent = `Iteration ${this.currentStep} / ${this.iterations.length - 1}`;
        }
        
        // Update buttons
        if (this.controls.prevBtn) {
            this.controls.prevBtn.disabled = this.currentStep === 0;
        }
        if (this.controls.nextBtn) {
            this.controls.nextBtn.disabled = this.currentStep === this.iterations.length - 1;
        }
        
        // Update metrics
        const confidenceFill = this.container.querySelector('#confidence-meter-fill');
        const confidenceValue = this.container.querySelector('#confidence-value');
        const haltDisplay = this.container.querySelector('#halt-prob-display');
        const description = this.container.querySelector('#step-description');
        
        if (confidenceFill) {
            confidenceFill.style.width = `${iteration.confidence * 100}%`;
        }
        if (confidenceValue) {
            confidenceValue.textContent = `${(iteration.confidence * 100).toFixed(1)}%`;
        }
        if (haltDisplay) {
            haltDisplay.textContent = iteration.haltProb.toFixed(3);
            haltDisplay.style.color = iteration.haltProb > 0.8 ? '#22c55e' : '#f59e0b';
        }
        if (description) {
            description.innerHTML = this.getStepDescription(this.currentStep);
        }
        
        // Update timeline
        this.updateTimeline();
        
        // Callback
        if (this.onStepChange) {
            this.onStepChange(this.currentStep, iteration);
        }
    }
    
    /**
     * Get description for current step
     */
    getStepDescription(step) {
        const descriptions = [
            '<strong>Initial State:</strong> Random initialization, model starting fresh',
            '<strong>Iteration 1:</strong> Establishing basic structure and color regions',
            '<strong>Iteration 2:</strong> Refining boundaries and filling patterns',
            '<strong>Iteration 3:</strong> Correcting errors, sharpening details',
            '<strong>Iteration 4:</strong> Fine-tuning, convergence check',
            '<strong>Final Output:</strong> Halt triggered, high confidence reached'
        ];
        
        return descriptions[Math.min(step, descriptions.length - 1)];
    }
    
    /**
     * Update timeline markers
     */
    updateTimeline() {
        const markers = this.container.querySelectorAll('.timeline-marker');
        markers.forEach((marker, i) => {
            if (i < this.currentStep) {
                marker.style.background = '#22c55e';
                marker.style.borderColor = '#22c55e';
            } else if (i === this.currentStep) {
                marker.style.background = '#6366f1';
                marker.style.borderColor = '#6366f1';
                marker.style.boxShadow = '0 0 10px rgba(99, 102, 241, 0.5)';
            } else {
                marker.style.background = '#1a1a2e';
                marker.style.borderColor = '#2a2a4a';
                marker.style.boxShadow = 'none';
            }
        });
    }
    
    /**
     * Navigation methods
     */
    nextStep() {
        if (this.currentStep < this.iterations.length - 1) {
            this.currentStep++;
            this.renderCurrentStep();
        }
    }
    
    prevStep() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.renderCurrentStep();
        }
    }
    
    goToStep(step) {
        if (step >= 0 && step < this.iterations.length) {
            this.currentStep = step;
            this.renderCurrentStep();
        }
    }
    
    reset() {
        this.stopPlay();
        this.currentStep = 0;
        this.renderCurrentStep();
    }
    
    /**
     * Auto-play controls
     */
    togglePlay() {
        if (this.isPlaying) {
            this.stopPlay();
        } else {
            this.startPlay();
        }
    }
    
    startPlay() {
        this.isPlaying = true;
        if (this.controls.playBtn) {
            this.controls.playBtn.innerHTML = '⏸ Pause';
        }
        
        this.playInterval = setInterval(() => {
            if (this.currentStep < this.iterations.length - 1) {
                this.nextStep();
            } else {
                this.stopPlay();
            }
        }, this.options.autoPlaySpeed);
    }
    
    stopPlay() {
        this.isPlaying = false;
        if (this.controls.playBtn) {
            this.controls.playBtn.innerHTML = '▶ Auto-Play';
        }
        
        if (this.playInterval) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }
    }
    
    /**
     * Get current state
     */
    getCurrentIteration() {
        return this.iterations[this.currentStep];
    }
    
    getCurrentStep() {
        return this.currentStep;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SolverStepper;
}
