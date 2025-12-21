/**
 * HeatmapOverlay.js
 * Visualizes DSC attention weights as a heatmap overlay on grids
 * Note: This is a PROXY visualization using heuristics until real model logs are available
 */

class HeatmapOverlay {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;
        
        this.options = {
            cellSize: options.cellSize || 24,
            colorScale: options.colorScale || 'viridis',
            opacity: options.opacity || 0.7,
            showValues: options.showValues || false,
            ...options
        };
        
        // Color scales for heatmaps
        this.colorScales = {
            viridis: [
                { pos: 0.0, color: [68, 1, 84] },
                { pos: 0.25, color: [59, 82, 139] },
                { pos: 0.5, color: [33, 145, 140] },
                { pos: 0.75, color: [94, 201, 98] },
                { pos: 1.0, color: [253, 231, 37] }
            ],
            hot: [
                { pos: 0.0, color: [0, 0, 0] },
                { pos: 0.33, color: [230, 0, 0] },
                { pos: 0.66, color: [255, 210, 0] },
                { pos: 1.0, color: [255, 255, 255] }
            ],
            attention: [
                { pos: 0.0, color: [30, 30, 60] },
                { pos: 0.5, color: [99, 102, 241] },
                { pos: 1.0, color: [253, 231, 37] }
            ]
        };
        
        this.heatmapElement = null;
        this.weights = null;
    }
    
    /**
     * Render heatmap overlay on a grid
     * @param {number[][]} weights - 2D array of attention weights (0-1)
     * @param {HTMLElement} gridElement - Optional grid element to overlay on
     */
    render(weights, gridElement = null) {
        this.weights = weights;
        this.container.innerHTML = '';
        
        if (!weights || !weights.length) {
            console.warn('HeatmapOverlay: No weights provided');
            return null;
        }
        
        const rows = weights.length;
        const cols = weights[0].length;
        
        // Create heatmap container
        this.heatmapElement = document.createElement('div');
        this.heatmapElement.className = 'heatmap-overlay';
        this.heatmapElement.style.cssText = `
            display: inline-grid;
            grid-template-columns: repeat(${cols}, ${this.options.cellSize}px);
            grid-template-rows: repeat(${rows}, ${this.options.cellSize}px);
            gap: 1px;
            padding: 1px;
            border-radius: 4px;
        `;
        
        // Find min/max for normalization
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                minVal = Math.min(minVal, weights[i][j]);
                maxVal = Math.max(maxVal, weights[i][j]);
            }
        }
        
        // Create cells
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const normalizedValue = maxVal > minVal 
                    ? (weights[i][j] - minVal) / (maxVal - minVal)
                    : 0.5;
                
                const cell = this.createHeatmapCell(normalizedValue, i, j);
                this.heatmapElement.appendChild(cell);
            }
        }
        
        this.container.appendChild(this.heatmapElement);
        
        // Add legend
        this.addLegend();
        
        return this.heatmapElement;
    }
    
    /**
     * Create a single heatmap cell
     */
    createHeatmapCell(value, row, col) {
        const cell = document.createElement('div');
        cell.className = 'heatmap-cell';
        cell.dataset.row = row;
        cell.dataset.col = col;
        cell.dataset.value = value.toFixed(3);
        
        const color = this.getColorForValue(value);
        
        cell.style.cssText = `
            width: ${this.options.cellSize}px;
            height: ${this.options.cellSize}px;
            background-color: rgba(${color[0]}, ${color[1]}, ${color[2]}, ${this.options.opacity});
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        `;
        
        if (this.options.showValues) {
            const label = document.createElement('span');
            label.textContent = value.toFixed(2);
            label.style.cssText = `
                font-size: 9px;
                font-family: monospace;
                color: ${value > 0.5 ? '#000' : '#fff'};
            `;
            cell.appendChild(label);
        }
        
        // Tooltip on hover
        cell.title = `Position: (${col}, ${row})\nAttention: ${value.toFixed(4)}`;
        
        return cell;
    }
    
    /**
     * Get RGB color for a normalized value (0-1)
     */
    getColorForValue(value) {
        const scale = this.colorScales[this.options.colorScale] || this.colorScales.attention;
        
        // Find surrounding color stops
        let lower = scale[0];
        let upper = scale[scale.length - 1];
        
        for (let i = 0; i < scale.length - 1; i++) {
            if (value >= scale[i].pos && value <= scale[i + 1].pos) {
                lower = scale[i];
                upper = scale[i + 1];
                break;
            }
        }
        
        // Interpolate
        const range = upper.pos - lower.pos;
        const t = range > 0 ? (value - lower.pos) / range : 0;
        
        return [
            Math.round(lower.color[0] + t * (upper.color[0] - lower.color[0])),
            Math.round(lower.color[1] + t * (upper.color[1] - lower.color[1])),
            Math.round(lower.color[2] + t * (upper.color[2] - lower.color[2]))
        ];
    }
    
    /**
     * Add color legend
     */
    addLegend() {
        const legend = document.createElement('div');
        legend.className = 'heatmap-legend';
        legend.style.cssText = `
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 8px;
            font-size: 12px;
        `;
        
        // Gradient bar
        const gradientBar = document.createElement('div');
        gradientBar.style.cssText = `
            width: 100px;
            height: 12px;
            border-radius: 2px;
            background: linear-gradient(to right, 
                rgb(30, 30, 60), 
                rgb(99, 102, 241), 
                rgb(253, 231, 37)
            );
        `;
        
        const lowLabel = document.createElement('span');
        lowLabel.textContent = 'Low';
        lowLabel.style.color = '#707080';
        
        const highLabel = document.createElement('span');
        highLabel.textContent = 'High';
        highLabel.style.color = '#707080';
        
        legend.appendChild(lowLabel);
        legend.appendChild(gradientBar);
        legend.appendChild(highLabel);
        
        this.container.appendChild(legend);
    }
    
    /**
     * Generate PROXY attention weights based on grid content
     * This is a heuristic approximation - real weights come from model logs
     */
    static generateProxyAttention(grid, options = {}) {
        const rows = grid.length;
        const cols = grid[0].length;
        const weights = [];
        
        // Heuristics for attention (proxy)
        for (let i = 0; i < rows; i++) {
            weights[i] = [];
            for (let j = 0; j < cols; j++) {
                let attention = 0;
                
                // Non-background cells get higher attention
                if (grid[i][j] !== 0) {
                    attention += 0.3;
                }
                
                // Edge/boundary cells get higher attention
                const neighbors = [
                    [i-1, j], [i+1, j], [i, j-1], [i, j+1]
                ];
                
                let boundaryScore = 0;
                for (const [ni, nj] of neighbors) {
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        if (grid[ni][nj] !== grid[i][j]) {
                            boundaryScore += 0.15;
                        }
                    } else {
                        // Edge of grid
                        boundaryScore += 0.1;
                    }
                }
                attention += boundaryScore;
                
                // Corner positions get slight boost
                const isCorner = (i === 0 || i === rows - 1) && (j === 0 || j === cols - 1);
                if (isCorner) {
                    attention += 0.1;
                }
                
                // Clamp to [0, 1]
                weights[i][j] = Math.min(1, Math.max(0, attention + Math.random() * 0.1));
            }
        }
        
        return weights;
    }
    
    /**
     * Find anchor point (highest attention)
     */
    static findAnchor(weights) {
        let maxVal = -Infinity;
        let anchor = { x: 0, y: 0 };
        
        for (let i = 0; i < weights.length; i++) {
            for (let j = 0; j < weights[i].length; j++) {
                if (weights[i][j] > maxVal) {
                    maxVal = weights[i][j];
                    anchor = { x: j, y: i };
                }
            }
        }
        
        return anchor;
    }
    
    /**
     * Update opacity
     */
    setOpacity(opacity) {
        this.options.opacity = opacity;
        if (this.weights) {
            this.render(this.weights);
        }
    }
    
    /**
     * Clear overlay
     */
    clear() {
        this.container.innerHTML = '';
        this.heatmapElement = null;
        this.weights = null;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HeatmapOverlay;
}
