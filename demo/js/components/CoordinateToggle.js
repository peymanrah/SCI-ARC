/**
 * CoordinateToggle.js
 * Visualizes absolute vs relative coordinate systems
 * Demonstrates RLAN's translation invariance through anchor-relative encoding
 */

class CoordinateToggle {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;
        
        this.options = {
            cellSize: options.cellSize || 32,
            showGrid: options.showGrid !== false,
            animationDuration: options.animationDuration || 400,
            ...options
        };
        
        this.mode = 'absolute'; // 'absolute' or 'relative'
        this.grid = null;
        this.anchor = null;
        this.gridRenderer = null;
        
        this.onModeChange = options.onModeChange || null;
    }
    
    /**
     * Initialize with a grid and optional anchor point
     */
    init(grid, anchor = null) {
        this.grid = grid;
        this.anchor = anchor || this.findDefaultAnchor(grid);
        this.render();
    }
    
    /**
     * Find a sensible default anchor (center of non-background region)
     */
    findDefaultAnchor(grid) {
        const rows = grid.length;
        const cols = grid[0].length;
        
        // Find center of mass of non-zero cells
        let sumX = 0, sumY = 0, count = 0;
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                if (grid[i][j] !== 0) {
                    sumX += j;
                    sumY += i;
                    count++;
                }
            }
        }
        
        if (count === 0) {
            return { x: Math.floor(cols / 2), y: Math.floor(rows / 2) };
        }
        
        return {
            x: Math.round(sumX / count),
            y: Math.round(sumY / count)
        };
    }
    
    /**
     * Render the coordinate visualization
     */
    render() {
        this.container.innerHTML = '';
        
        if (!this.grid) return;
        
        const wrapper = document.createElement('div');
        wrapper.className = 'coord-toggle-wrapper';
        wrapper.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
        `;
        
        // Grid with coordinates
        const gridContainer = document.createElement('div');
        gridContainer.className = 'coord-grid-container';
        gridContainer.style.cssText = `
            position: relative;
        `;
        
        const rows = this.grid.length;
        const cols = this.grid[0].length;
        
        // Create grid element
        const gridElement = document.createElement('div');
        gridElement.className = 'coord-grid';
        gridElement.style.cssText = `
            display: inline-grid;
            grid-template-columns: repeat(${cols}, ${this.options.cellSize}px);
            grid-template-rows: repeat(${rows}, ${this.options.cellSize}px);
            gap: 1px;
            background: #2a2a4a;
            padding: 1px;
            border-radius: 4px;
        `;
        
        // ARC colors
        const colors = [
            '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#B10DC9'
        ];
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const cell = this.createCell(i, j, this.grid[i][j], colors);
                gridElement.appendChild(cell);
            }
        }
        
        gridContainer.appendChild(gridElement);
        wrapper.appendChild(gridContainer);
        
        // Legend
        const legend = this.createLegend();
        wrapper.appendChild(legend);
        
        this.container.appendChild(wrapper);
    }
    
    /**
     * Create a cell with coordinate label
     */
    createCell(row, col, value, colors) {
        const cell = document.createElement('div');
        cell.className = 'coord-cell';
        cell.dataset.row = row;
        cell.dataset.col = col;
        
        const isAnchor = this.anchor && row === this.anchor.y && col === this.anchor.x;
        
        cell.style.cssText = `
            width: ${this.options.cellSize}px;
            height: ${this.options.cellSize}px;
            background-color: ${colors[value] || colors[0]};
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all ${this.options.animationDuration}ms ease;
            ${isAnchor ? 'box-shadow: 0 0 0 3px #f59e0b, 0 0 15px #f59e0b;' : ''}
        `;
        
        // Coordinate label
        const label = document.createElement('span');
        label.className = 'coord-label';
        
        if (this.mode === 'absolute') {
            label.textContent = `${col},${row}`;
        } else {
            const dx = col - this.anchor.x;
            const dy = row - this.anchor.y;
            const formatNum = (n) => n >= 0 ? `+${n}` : `${n}`;
            label.textContent = `${formatNum(dx)},${formatNum(dy)}`;
        }
        
        // Determine text color for contrast
        const darkColors = [0, 1, 2, 6, 9];
        const textColor = darkColors.includes(value) ? '#ffffff' : '#000000';
        
        label.style.cssText = `
            font-size: ${Math.max(9, this.options.cellSize / 4)}px;
            font-family: 'Fira Code', monospace;
            color: ${textColor};
            text-shadow: ${darkColors.includes(value) ? 'none' : '0 0 2px rgba(255,255,255,0.5)'};
            transition: all ${this.options.animationDuration}ms ease;
        `;
        
        cell.appendChild(label);
        
        // Click to set anchor
        cell.style.cursor = 'pointer';
        cell.addEventListener('click', () => {
            this.setAnchor(col, row);
        });
        
        return cell;
    }
    
    /**
     * Create legend explaining the modes
     */
    createLegend() {
        const legend = document.createElement('div');
        legend.className = 'coord-legend';
        legend.style.cssText = `
            display: flex;
            gap: 24px;
            font-size: 13px;
            color: #a0a0b0;
        `;
        
        const absLegend = document.createElement('div');
        absLegend.innerHTML = `
            <strong style="color: ${this.mode === 'absolute' ? '#6366f1' : '#a0a0b0'}">Absolute:</strong> 
            Position from grid origin (0,0)
        `;
        
        const relLegend = document.createElement('div');
        relLegend.innerHTML = `
            <strong style="color: ${this.mode === 'relative' ? '#6366f1' : '#a0a0b0'}">Relative:</strong> 
            Offset from anchor <span style="color: #f59e0b">●</span>
        `;
        
        legend.appendChild(absLegend);
        legend.appendChild(relLegend);
        
        return legend;
    }
    
    /**
     * Toggle between absolute and relative modes
     */
    toggle() {
        this.mode = this.mode === 'absolute' ? 'relative' : 'absolute';
        this.render();
        
        if (this.onModeChange) {
            this.onModeChange(this.mode);
        }
        
        return this.mode;
    }
    
    /**
     * Set mode directly
     */
    setMode(mode) {
        if (mode !== 'absolute' && mode !== 'relative') {
            console.warn('CoordinateToggle: Invalid mode. Use "absolute" or "relative"');
            return;
        }
        
        this.mode = mode;
        this.render();
        
        if (this.onModeChange) {
            this.onModeChange(this.mode);
        }
    }
    
    /**
     * Set anchor position
     */
    setAnchor(x, y) {
        this.anchor = { x, y };
        this.render();
    }
    
    /**
     * Get current mode
     */
    getMode() {
        return this.mode;
    }
    
    /**
     * Get current anchor
     */
    getAnchor() {
        return this.anchor;
    }
    
    /**
     * Create a before/after comparison view
     */
    static createComparison(container, grid, anchor) {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
            display: flex;
            gap: 48px;
            align-items: flex-start;
        `;
        
        // Absolute view
        const absContainer = document.createElement('div');
        absContainer.innerHTML = '<h4 style="text-align: center; margin-bottom: 12px; color: #a0a0b0;">Absolute Coordinates</h4>';
        const absGrid = document.createElement('div');
        absContainer.appendChild(absGrid);
        
        const absToggle = new CoordinateToggle(absGrid, { cellSize: 28 });
        absToggle.mode = 'absolute';
        absToggle.init(grid, anchor);
        
        // Arrow
        const arrow = document.createElement('div');
        arrow.innerHTML = '→';
        arrow.style.cssText = `
            font-size: 32px;
            color: #6366f1;
            align-self: center;
            margin-top: 24px;
        `;
        
        // Relative view
        const relContainer = document.createElement('div');
        relContainer.innerHTML = '<h4 style="text-align: center; margin-bottom: 12px; color: #a0a0b0;">Relative Coordinates</h4>';
        const relGrid = document.createElement('div');
        relContainer.appendChild(relGrid);
        
        const relToggle = new CoordinateToggle(relGrid, { cellSize: 28 });
        relToggle.mode = 'relative';
        relToggle.init(grid, anchor);
        
        wrapper.appendChild(absContainer);
        wrapper.appendChild(arrow);
        wrapper.appendChild(relContainer);
        
        container.appendChild(wrapper);
        
        return { absToggle, relToggle };
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CoordinateToggle;
}
