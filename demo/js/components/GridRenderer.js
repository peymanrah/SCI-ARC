/**
 * GridRenderer.js
 * Renders ARC grids with consistent color palette (0-9)
 * Supports multiple sizes, hover effects, and coordinate overlays
 */

class GridRenderer {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.querySelector(container) 
            : container;
        
        this.options = {
            cellSize: options.cellSize || 24,
            gap: options.gap || 1,
            showCoordinates: options.showCoordinates || false,
            coordinateMode: options.coordinateMode || 'absolute', // 'absolute' or 'relative'
            anchor: options.anchor || null, // {x, y} for relative coordinates
            interactive: options.interactive || false,
            className: options.className || '',
            ...options
        };
        
        // ARC color palette
        this.colors = [
            '#000000', // 0 - Black
            '#0074D9', // 1 - Blue
            '#FF4136', // 2 - Red
            '#2ECC40', // 3 - Green
            '#FFDC00', // 4 - Yellow
            '#AAAAAA', // 5 - Gray
            '#F012BE', // 6 - Magenta
            '#FF851B', // 7 - Orange
            '#7FDBFF', // 8 - Cyan
            '#B10DC9'  // 9 - Purple
        ];
        
        this.grid = null;
        this.gridElement = null;
    }
    
    /**
     * Render a 2D grid
     * @param {number[][]} grid - 2D array of integers 0-9
     * @returns {HTMLElement} The rendered grid element
     */
    render(grid) {
        this.grid = grid;
        this.container.innerHTML = '';
        
        if (!grid || !grid.length || !grid[0].length) {
            console.warn('GridRenderer: Empty or invalid grid');
            return null;
        }
        
        const rows = grid.length;
        const cols = grid[0].length;
        
        // Create grid container
        this.gridElement = document.createElement('div');
        this.gridElement.className = `arc-grid ${this.options.className}`;
        this.gridElement.style.cssText = `
            display: inline-grid;
            grid-template-columns: repeat(${cols}, ${this.options.cellSize}px);
            grid-template-rows: repeat(${rows}, ${this.options.cellSize}px);
            gap: ${this.options.gap}px;
            background: #2a2a4a;
            padding: ${this.options.gap}px;
            border-radius: 4px;
        `;
        
        // Create cells
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const cell = this.createCell(grid[i][j], i, j);
                this.gridElement.appendChild(cell);
            }
        }
        
        this.container.appendChild(this.gridElement);
        return this.gridElement;
    }
    
    /**
     * Create a single cell element
     */
    createCell(value, row, col) {
        const cell = document.createElement('div');
        cell.className = 'arc-cell';
        cell.dataset.color = value;
        cell.dataset.row = row;
        cell.dataset.col = col;
        
        cell.style.cssText = `
            width: ${this.options.cellSize}px;
            height: ${this.options.cellSize}px;
            background-color: ${this.colors[value] || this.colors[0]};
            transition: transform 0.15s ease;
            position: relative;
        `;
        
        // Add coordinate overlay if enabled
        if (this.options.showCoordinates) {
            const coordLabel = this.createCoordinateLabel(row, col);
            cell.appendChild(coordLabel);
        }
        
        // Add hover effects
        cell.addEventListener('mouseenter', () => {
            cell.style.transform = 'scale(1.1)';
            cell.style.zIndex = '10';
            if (this.options.onCellHover) {
                this.options.onCellHover(row, col, value);
            }
        });
        
        cell.addEventListener('mouseleave', () => {
            cell.style.transform = 'scale(1)';
            cell.style.zIndex = '1';
        });
        
        // Add click handler if interactive
        if (this.options.interactive) {
            cell.style.cursor = 'pointer';
            cell.addEventListener('click', () => {
                if (this.options.onCellClick) {
                    this.options.onCellClick(row, col, value);
                }
            });
        }
        
        return cell;
    }
    
    /**
     * Create coordinate label for a cell
     */
    createCoordinateLabel(row, col) {
        const label = document.createElement('span');
        label.className = 'coord-label';
        
        let text;
        if (this.options.coordinateMode === 'relative' && this.options.anchor) {
            const dx = col - this.options.anchor.x;
            const dy = row - this.options.anchor.y;
            const sign = (n) => n >= 0 ? '+' + n : n;
            text = `${sign(dx)},${sign(dy)}`;
        } else {
            text = `${col},${row}`;
        }
        
        label.textContent = text;
        label.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: ${Math.max(8, this.options.cellSize / 3)}px;
            font-family: monospace;
            color: ${this.getContrastColor(this.grid[row][col])};
            pointer-events: none;
            white-space: nowrap;
        `;
        
        return label;
    }
    
    /**
     * Get contrasting text color for readability
     */
    getContrastColor(colorIndex) {
        // Light text for dark backgrounds, dark text for light backgrounds
        const darkColors = [0, 1, 2, 6, 9]; // Black, Blue, Red, Magenta, Purple
        return darkColors.includes(colorIndex) ? '#ffffff' : '#000000';
    }
    
    /**
     * Update coordinate mode (absolute/relative)
     */
    setCoordinateMode(mode, anchor = null) {
        this.options.coordinateMode = mode;
        if (anchor) {
            this.options.anchor = anchor;
        }
        if (this.options.showCoordinates && this.grid) {
            this.render(this.grid);
        }
    }
    
    /**
     * Toggle coordinate display
     */
    toggleCoordinates(show) {
        this.options.showCoordinates = show;
        if (this.grid) {
            this.render(this.grid);
        }
    }
    
    /**
     * Highlight specific cells (e.g., for heatmap overlay)
     */
    highlightCells(cellPositions, color = 'rgba(255, 215, 0, 0.5)') {
        if (!this.gridElement) return;
        
        const cells = this.gridElement.querySelectorAll('.arc-cell');
        cells.forEach(cell => {
            const row = parseInt(cell.dataset.row);
            const col = parseInt(cell.dataset.col);
            
            const highlight = cellPositions.find(p => p.row === row && p.col === col);
            if (highlight) {
                const intensity = highlight.intensity || 1;
                cell.style.boxShadow = `inset 0 0 0 3px rgba(255, 215, 0, ${intensity})`;
            } else {
                cell.style.boxShadow = 'none';
            }
        });
    }
    
    /**
     * Mark anchor cell
     */
    markAnchor(row, col) {
        if (!this.gridElement) return;
        
        const cells = this.gridElement.querySelectorAll('.arc-cell');
        cells.forEach(cell => {
            const r = parseInt(cell.dataset.row);
            const c = parseInt(cell.dataset.col);
            
            if (r === row && c === col) {
                cell.style.boxShadow = '0 0 0 3px #f59e0b, 0 0 15px #f59e0b';
                cell.classList.add('anchor-cell');
            } else {
                cell.style.boxShadow = '';
                cell.classList.remove('anchor-cell');
            }
        });
    }
    
    /**
     * Animate grid transition (for solver stepping)
     */
    animateTransition(newGrid, duration = 300) {
        return new Promise(resolve => {
            if (!this.gridElement || !this.grid) {
                this.render(newGrid);
                resolve();
                return;
            }
            
            const cells = this.gridElement.querySelectorAll('.arc-cell');
            const oldGrid = this.grid;
            
            cells.forEach((cell, index) => {
                const row = Math.floor(index / oldGrid[0].length);
                const col = index % oldGrid[0].length;
                
                if (row < newGrid.length && col < newGrid[0].length) {
                    const oldValue = oldGrid[row][col];
                    const newValue = newGrid[row][col];
                    
                    if (oldValue !== newValue) {
                        // Animate the change
                        cell.style.transition = `background-color ${duration}ms ease`;
                        cell.style.backgroundColor = this.colors[newValue];
                        cell.dataset.color = newValue;
                    }
                }
            });
            
            this.grid = newGrid;
            setTimeout(resolve, duration);
        });
    }
    
    /**
     * Clear the grid
     */
    clear() {
        this.container.innerHTML = '';
        this.grid = null;
        this.gridElement = null;
    }
    
    /**
     * Get current grid state
     */
    getGrid() {
        return this.grid;
    }
    
    /**
     * Static method to render a simple grid without creating an instance
     */
    static renderSimple(container, grid, cellSize = 24) {
        const renderer = new GridRenderer(container, { cellSize });
        return renderer.render(grid);
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GridRenderer;
}
