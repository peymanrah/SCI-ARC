/**
 * ArchitectureViz.js
 * Renders RLAN vs Transformer architecture comparison
 * Interactive SVG diagrams with highlighting based on aspect selection
 */

class ArchitectureViz {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;
        
        this.options = {
            width: options.width || 900,
            height: options.height || 350,
            animationDuration: options.animationDuration || 300,
            ...options
        };
        
        this.currentAspect = 'representation';
        this.svgElement = null;
    }
    
    /**
     * Render the architecture comparison diagram
     */
    render() {
        this.container.innerHTML = '';
        
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', `0 0 ${this.options.width} ${this.options.height}`);
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', this.options.height);
        svg.style.maxWidth = `${this.options.width}px`;
        
        // Definitions for gradients and markers
        svg.innerHTML = this.getSVGDefs();
        
        // TRM side (left)
        const trmGroup = this.createTRMDiagram();
        svg.appendChild(trmGroup);
        
        // Divider
        const divider = this.createDivider();
        svg.appendChild(divider);
        
        // RLAN side (right)
        const rlanGroup = this.createRLANDiagram();
        svg.appendChild(rlanGroup);
        
        // Labels
        const labels = this.createLabels();
        svg.appendChild(labels);
        
        this.svgElement = svg;
        this.container.appendChild(svg);
        
        // Apply initial highlighting
        this.highlightAspect(this.currentAspect);
    }
    
    /**
     * SVG definitions (gradients, markers)
     */
    getSVGDefs() {
        return `
            <defs>
                <linearGradient id="trm-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#ef4444;stop-opacity:0.3" />
                    <stop offset="100%" style="stop-color:#ef4444;stop-opacity:0.1" />
                </linearGradient>
                <linearGradient id="rlan-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#6366f1;stop-opacity:0.3" />
                    <stop offset="100%" style="stop-color:#6366f1;stop-opacity:0.1" />
                </linearGradient>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L9,3 z" fill="#707080" />
                </marker>
                <marker id="arrow-primary" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L9,3 z" fill="#6366f1" />
                </marker>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
        `;
    }
    
    /**
     * Create Transformer (Tiny Recursive Model) architecture diagram
     */
    createTRMDiagram() {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'trm-diagram');
        g.setAttribute('transform', 'translate(20, 50)');
        
        g.innerHTML = `
            <!-- Background -->
            <rect x="0" y="0" width="400" height="280" rx="12" fill="url(#trm-gradient)" stroke="#ef4444" stroke-width="1" opacity="0.5"/>
            
            <!-- Input Grid (flattened) -->
            <g class="trm-input" data-aspect="representation">
                <text x="60" y="25" fill="#a0a0b0" font-size="11">Input Grid</text>
                <rect x="20" y="35" width="100" height="40" rx="4" fill="#1e1e35" stroke="#707080"/>
                <text x="70" y="60" fill="#f0f0f5" font-size="10" text-anchor="middle">5×5 → 25 tokens</text>
            </g>
            
            <!-- Flatten arrow -->
            <path d="M130,55 L160,55" stroke="#707080" stroke-width="2" marker-end="url(#arrow)" data-aspect="representation"/>
            
            <!-- Token sequence -->
            <g class="trm-tokens" data-aspect="representation">
                <text x="220" y="25" fill="#a0a0b0" font-size="11">Flattened Sequence</text>
                <g transform="translate(170, 40)">
                    ${this.createTokenSequence(8, '#ef4444')}
                </g>
            </g>
            
            <!-- Position Encoding -->
            <g class="trm-pos-enc" data-aspect="representation">
                <text x="180" y="90" fill="#a0a0b0" font-size="10">+ Absolute Position Encoding</text>
            </g>
            
            <!-- Self-Attention Block -->
            <g class="trm-attention" data-aspect="attention">
                <rect x="140" y="100" width="180" height="60" rx="6" fill="#1e1e35" stroke="#ef4444" stroke-width="2"/>
                <text x="230" y="125" fill="#f0f0f5" font-size="12" text-anchor="middle">Self-Attention</text>
                <text x="230" y="145" fill="#a0a0b0" font-size="10" text-anchor="middle">O(n²) complexity</text>
            </g>
            
            <!-- FFN Block -->
            <g class="trm-ffn" data-aspect="compute">
                <rect x="140" y="175" width="180" height="40" rx="6" fill="#1e1e35" stroke="#707080"/>
                <text x="230" y="200" fill="#f0f0f5" font-size="12" text-anchor="middle">Feed-Forward</text>
            </g>
            
            <!-- Output -->
            <g class="trm-output" data-aspect="compute">
                <rect x="170" y="230" width="120" height="35" rx="4" fill="#1e1e35" stroke="#707080"/>
                <text x="230" y="252" fill="#f0f0f5" font-size="11" text-anchor="middle">Single Pass Output</text>
            </g>
            
            <!-- Flow arrows -->
            <path d="M230,160 L230,175" stroke="#707080" stroke-width="2" marker-end="url(#arrow)"/>
            <path d="M230,215 L230,230" stroke="#707080" stroke-width="2" marker-end="url(#arrow)"/>
        `;
        
        return g;
    }
    
    /**
     * Create RLAN architecture diagram
     */
    createRLANDiagram() {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'rlan-diagram');
        g.setAttribute('transform', 'translate(480, 50)');
        
        g.innerHTML = `
            <!-- Background -->
            <rect x="0" y="0" width="400" height="280" rx="12" fill="url(#rlan-gradient)" stroke="#6366f1" stroke-width="1" opacity="0.5"/>
            
            <!-- Input Grid (preserved) -->
            <g class="rlan-input" data-aspect="representation">
                <text x="50" y="25" fill="#a0a0b0" font-size="11">Input Grid</text>
                <g transform="translate(20, 35)">
                    ${this.createMiniGrid(5, 5, '#6366f1')}
                </g>
            </g>
            
            <!-- Arrow to Context Encoder -->
            <path d="M85,80 L115,80" stroke="#707080" stroke-width="2" marker-end="url(#arrow)" data-aspect="representation"/>
            
            <!-- Context Encoder (FIRST) -->
            <g class="rlan-encoder" data-aspect="representation">
                <rect x="120" y="35" width="70" height="55" rx="6" fill="#1e1e35" stroke="#707080" stroke-width="2"/>
                <text x="155" y="55" fill="#f0f0f5" font-size="10" text-anchor="middle">Context</text>
                <text x="155" y="70" fill="#f0f0f5" font-size="10" text-anchor="middle">Encoder</text>
            </g>
            
            <!-- DSC (SECOND) -->
            <g class="rlan-dsc" data-aspect="attention">
                <rect x="200" y="35" width="90" height="55" rx="6" fill="#1e1e35" stroke="#6366f1" stroke-width="2"/>
                <text x="245" y="55" fill="#f0f0f5" font-size="11" text-anchor="middle">DSC</text>
                <text x="245" y="75" fill="#a0a0b0" font-size="9" text-anchor="middle">Anchor Attention</text>
            </g>
            
            <!-- MSRE (THIRD) -->
            <g class="rlan-msre" data-aspect="representation">
                <rect x="300" y="35" width="80" height="55" rx="6" fill="#1e1e35" stroke="#8b5cf6" stroke-width="2"/>
                <text x="340" y="55" fill="#f0f0f5" font-size="11" text-anchor="middle">MSRE</text>
                <text x="340" y="75" fill="#a0a0b0" font-size="9" text-anchor="middle">Relative Coords</text>
            </g>
            
            <!-- Arrows between modules -->
            <path d="M190,62 L200,62" stroke="#707080" stroke-width="2" marker-end="url(#arrow)"/>
            <path d="M290,62 L300,62" stroke="#707080" stroke-width="2" marker-end="url(#arrow)"/>
            
            <!-- Recursive Solver (main block) -->
            <g class="rlan-solver" data-aspect="compute">
                <rect x="100" y="110" width="200" height="100" rx="8" fill="#1e1e35" stroke="#6366f1" stroke-width="3" filter="url(#glow)"/>
                <text x="200" y="135" fill="#6366f1" font-size="14" font-weight="bold" text-anchor="middle">Recursive Solver</text>
                
                <!-- ConvGRU inside -->
                <rect x="120" y="150" width="80" height="40" rx="4" fill="#252542" stroke="#8b5cf6"/>
                <text x="160" y="175" fill="#f0f0f5" font-size="10" text-anchor="middle">ConvGRU</text>
                
                <!-- Iteration indicator -->
                <g transform="translate(220, 150)">
                    <circle cx="15" cy="20" r="12" fill="#252542" stroke="#22c55e"/>
                    <text x="15" y="24" fill="#22c55e" font-size="10" text-anchor="middle">t</text>
                    <path d="M35,20 C50,5 50,35 35,20" stroke="#22c55e" stroke-width="1.5" fill="none" marker-end="url(#arrow-primary)"/>
                </g>
                <text x="260" y="200" fill="#a0a0b0" font-size="9" text-anchor="middle">3-8 iterations</text>
            </g>
            
            <!-- Feedback loop arrow -->
            <path d="M300,160 C340,160 340,140 300,140" stroke="#6366f1" stroke-width="2" fill="none" stroke-dasharray="4,2"/>
            
            <!-- LCR (side) -->
            <g class="rlan-lcr" data-aspect="attention">
                <rect x="320" y="130" width="60" height="40" rx="4" fill="#1e1e35" stroke="#707080"/>
                <text x="350" y="155" fill="#f0f0f5" font-size="10" text-anchor="middle">LCR</text>
            </g>
            
            <!-- Halt decision -->
            <g class="rlan-halt" data-aspect="compute">
                <circle cx="200" cy="240" r="20" fill="#1e1e35" stroke="#22c55e" stroke-width="2"/>
                <text x="200" y="244" fill="#22c55e" font-size="10" text-anchor="middle">Halt?</text>
            </g>
            
            <!-- Output -->
            <g class="rlan-output" data-aspect="compute">
                <rect x="250" y="225" width="100" height="35" rx="4" fill="#1e1e35" stroke="#22c55e"/>
                <text x="300" y="247" fill="#f0f0f5" font-size="11" text-anchor="middle">Refined Output</text>
            </g>
            
            <!-- Flow arrows -->
            <path d="M200,210 L200,220" stroke="#707080" stroke-width="2" marker-end="url(#arrow)"/>
            <path d="M220,240 L250,240" stroke="#22c55e" stroke-width="2" marker-end="url(#arrow)"/>
        `;
        
        return g;
    }
    
    /**
     * Create token sequence visualization
     */
    createTokenSequence(count, color) {
        let html = '';
        for (let i = 0; i < count; i++) {
            html += `<rect x="${i * 18}" y="0" width="14" height="20" rx="2" fill="${color}" opacity="${0.3 + (i / count) * 0.5}"/>`;
        }
        html += `<text x="${count * 18 + 5}" y="14" fill="#a0a0b0" font-size="10">...</text>`;
        return html;
    }
    
    /**
     * Create mini grid visualization
     */
    createMiniGrid(rows, cols, borderColor) {
        let html = '';
        const cellSize = 10;
        const colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00'];
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const color = colors[Math.floor(Math.random() * colors.length)];
                html += `<rect x="${j * (cellSize + 1)}" y="${i * (cellSize + 1)}" width="${cellSize}" height="${cellSize}" fill="${color}"/>`;
            }
        }
        
        return html;
    }
    
    /**
     * Create divider between architectures
     */
    createDivider() {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'divider');
        
        g.innerHTML = `
            <line x1="450" y1="60" x2="450" y2="320" stroke="#2a2a4a" stroke-width="2" stroke-dasharray="8,4"/>
            <text x="450" y="45" fill="#707080" font-size="12" text-anchor="middle">vs</text>
        `;
        
        return g;
    }
    
    /**
     * Create labels
     */
    createLabels() {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'labels');
        
        g.innerHTML = `
            <text x="230" y="25" fill="#ef4444" font-size="16" font-weight="bold" text-anchor="middle">Transformer</text>
            <text x="680" y="25" fill="#6366f1" font-size="16" font-weight="bold" text-anchor="middle">RLAN</text>
        `;
        
        return g;
    }
    
    /**
     * Highlight elements based on selected aspect
     */
    highlightAspect(aspect) {
        this.currentAspect = aspect;
        
        if (!this.svgElement) return;
        
        // Reset all elements
        const allElements = this.svgElement.querySelectorAll('[data-aspect]');
        allElements.forEach(el => {
            el.style.opacity = '0.4';
            el.style.transition = `opacity ${this.options.animationDuration}ms ease`;
        });
        
        // Highlight matching elements
        const matchingElements = this.svgElement.querySelectorAll(`[data-aspect="${aspect}"]`);
        matchingElements.forEach(el => {
            el.style.opacity = '1';
        });
        
        // Always show labels at full opacity
        const labels = this.svgElement.querySelector('.labels');
        if (labels) labels.style.opacity = '1';
    }
    
    /**
     * Create the LLM integration diagram
     */
    static createLLMIntegrationDiagram(container) {
        container.innerHTML = '';
        
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 800 300');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '300');
        svg.style.maxWidth = '800px';
        
        svg.innerHTML = `
            <defs>
                <linearGradient id="llm-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#8b5cf6;stop-opacity:0.3" />
                    <stop offset="100%" style="stop-color:#6366f1;stop-opacity:0.3" />
                </linearGradient>
            </defs>
            
            <!-- LLM Block -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="200" height="200" rx="12" fill="#1e1e35" stroke="#8b5cf6" stroke-width="2"/>
                <text x="100" y="30" fill="#8b5cf6" font-size="16" font-weight="bold" text-anchor="middle">Large Language Model</text>
                
                <!-- Internal layers -->
                <rect x="20" y="50" width="160" height="30" rx="4" fill="#252542" stroke="#707080"/>
                <text x="100" y="70" fill="#a0a0b0" font-size="11" text-anchor="middle">Attention Layers</text>
                
                <rect x="20" y="90" width="160" height="30" rx="4" fill="#252542" stroke="#707080"/>
                <text x="100" y="110" fill="#a0a0b0" font-size="11" text-anchor="middle">FFN Layers</text>
                
                <rect x="20" y="130" width="160" height="30" rx="4" fill="#252542" stroke="#f59e0b" stroke-width="2"/>
                <text x="100" y="150" fill="#f59e0b" font-size="11" text-anchor="middle">Tool Interface</text>
                
                <rect x="20" y="170" width="160" height="20" rx="4" fill="#252542" stroke="#707080"/>
                <text x="100" y="184" fill="#a0a0b0" font-size="10" text-anchor="middle">Output Head</text>
            </g>
            
            <!-- Connection arrow -->
            <g>
                <path d="M260,145 L340,145" stroke="#f59e0b" stroke-width="3" marker-end="url(#arrow)"/>
                <text x="300" y="135" fill="#f59e0b" font-size="10" text-anchor="middle">API Call</text>
            </g>
            
            <!-- RLAN Block -->
            <g transform="translate(350, 75)">
                <rect x="0" y="0" width="180" height="140" rx="12" fill="url(#llm-gradient)" stroke="#6366f1" stroke-width="2"/>
                <text x="90" y="25" fill="#6366f1" font-size="14" font-weight="bold" text-anchor="middle">RLAN Module</text>
                
                <!-- Sub-components -->
                <rect x="15" y="40" width="70" height="25" rx="4" fill="#1e1e35" stroke="#707080"/>
                <text x="50" y="57" fill="#a0a0b0" font-size="9" text-anchor="middle">DSC</text>
                
                <rect x="95" y="40" width="70" height="25" rx="4" fill="#1e1e35" stroke="#707080"/>
                <text x="130" y="57" fill="#a0a0b0" font-size="9" text-anchor="middle">MSRE</text>
                
                <rect x="15" y="75" width="150" height="35" rx="4" fill="#1e1e35" stroke="#6366f1"/>
                <text x="90" y="97" fill="#6366f1" font-size="10" text-anchor="middle">Recursive Solver</text>
                
                <text x="90" y="125" fill="#22c55e" font-size="10" text-anchor="middle">→ Grid Output</text>
            </g>
            
            <!-- Return arrow -->
            <g>
                <path d="M540,145 L620,145" stroke="#22c55e" stroke-width="3" marker-end="url(#arrow)"/>
                <text x="580" y="135" fill="#22c55e" font-size="10" text-anchor="middle">Result</text>
            </g>
            
            <!-- Output -->
            <g transform="translate(630, 100)">
                <rect x="0" y="0" width="120" height="90" rx="8" fill="#1e1e35" stroke="#22c55e" stroke-width="2"/>
                <text x="60" y="25" fill="#22c55e" font-size="12" text-anchor="middle">Spatial Output</text>
                
                <!-- Mini grid result -->
                <g transform="translate(30, 35)">
                    <rect x="0" y="0" width="12" height="12" fill="#0074D9"/>
                    <rect x="14" y="0" width="12" height="12" fill="#FF4136"/>
                    <rect x="28" y="0" width="12" height="12" fill="#2ECC40"/>
                    <rect x="42" y="0" width="12" height="12" fill="#FFDC00"/>
                    <rect x="0" y="14" width="12" height="12" fill="#2ECC40"/>
                    <rect x="14" y="14" width="12" height="12" fill="#000"/>
                    <rect x="28" y="14" width="12" height="12" fill="#000"/>
                    <rect x="42" y="14" width="12" height="12" fill="#FF4136"/>
                </g>
            </g>
            
            <!-- Labels -->
            <text x="150" y="280" fill="#a0a0b0" font-size="11" text-anchor="middle">Language Understanding</text>
            <text x="440" y="280" fill="#a0a0b0" font-size="11" text-anchor="middle">Spatial Reasoning</text>
            <text x="690" y="280" fill="#a0a0b0" font-size="11" text-anchor="middle">Visual Result</text>
        `;
        
        container.appendChild(svg);
        return svg;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ArchitectureViz;
}
