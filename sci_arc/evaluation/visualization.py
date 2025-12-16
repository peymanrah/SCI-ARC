"""
Visualization Utilities for ARC Grids.

Provides functions to visualize ARC grids and predictions:
- Grid to image conversion
- Side-by-side prediction comparisons
- HTML table generation for reports
"""

from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ARC color palette (RGB values 0-255)
ARC_COLORS = [
    (0, 0, 0),       # 0: black (background)
    (0, 116, 217),   # 1: blue
    (255, 65, 54),   # 2: red
    (46, 204, 64),   # 3: green
    (255, 220, 0),   # 4: yellow
    (170, 170, 170), # 5: gray
    (240, 18, 190),  # 6: magenta
    (255, 133, 27),  # 7: orange
    (127, 219, 255), # 8: cyan
    (135, 12, 37),   # 9: brown
]

# Hex colors for HTML reports
ARC_COLORS_HEX = [
    '#000000',  # 0: black (background)
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25',  # 9: brown
]


def grid_to_image(
    grid: np.ndarray,
    cell_size: int = 20,
    border_width: int = 1,
    border_color: Tuple[int, int, int] = (50, 50, 50),
) -> np.ndarray:
    """
    Convert ARC grid to RGB image.
    
    Args:
        grid: ARC grid, shape (H, W) with values 0-9
        cell_size: Size of each cell in pixels
        border_width: Width of cell borders
        border_color: RGB color of borders
        
    Returns:
        RGB image as numpy array, shape (H*cell_size, W*cell_size, 3)
    """
    h, w = grid.shape
    img_h = h * cell_size
    img_w = w * cell_size
    
    image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            color_idx = int(grid[i, j])
            if 0 <= color_idx < len(ARC_COLORS):
                color = ARC_COLORS[color_idx]
            else:
                color = (128, 128, 128)  # Unknown color
            
            # Fill cell
            y0 = i * cell_size
            x0 = j * cell_size
            image[y0:y0+cell_size, x0:x0+cell_size] = color
            
            # Draw border
            if border_width > 0:
                image[y0:y0+border_width, x0:x0+cell_size] = border_color
                image[y0+cell_size-border_width:y0+cell_size, x0:x0+cell_size] = border_color
                image[y0:y0+cell_size, x0:x0+border_width] = border_color
                image[y0:y0+cell_size, x0+cell_size-border_width:x0+cell_size] = border_color
    
    return image


def visualize_grid(
    grid: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    cell_size: int = 30,
) -> Optional[np.ndarray]:
    """
    Visualize a single ARC grid.
    
    Args:
        grid: ARC grid, shape (H, W)
        title: Optional title for the plot
        save_path: If provided, save image to this path
        show: Whether to display the plot
        cell_size: Size of each cell in pixels
        
    Returns:
        RGB image as numpy array if matplotlib not available
    """
    image = grid_to_image(grid, cell_size=cell_size)
    
    if not MATPLOTLIB_AVAILABLE:
        return image
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return image


def visualize_prediction(
    input_grid: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
    cell_size: int = 20,
    highlight_errors: bool = True,
) -> Optional[np.ndarray]:
    """
    Visualize input, target, and prediction side by side.
    
    Args:
        input_grid: Input grid
        target: Target/ground truth grid
        prediction: Model prediction
        save_path: If provided, save image to this path
        show: Whether to display the plot
        title: Optional title
        cell_size: Size of each cell
        highlight_errors: Whether to highlight incorrect pixels
        
    Returns:
        Combined RGB image as numpy array
    """
    input_img = grid_to_image(input_grid, cell_size=cell_size)
    target_img = grid_to_image(target, cell_size=cell_size)
    pred_img = grid_to_image(prediction, cell_size=cell_size)
    
    # Create error overlay if requested
    if highlight_errors and prediction.shape == target.shape:
        errors = prediction != target
        for i in range(errors.shape[0]):
            for j in range(errors.shape[1]):
                if errors[i, j]:
                    y0 = i * cell_size
                    x0 = j * cell_size
                    # Add red overlay to error cells
                    pred_img[y0:y0+cell_size, x0:x0+cell_size, 0] = np.minimum(
                        pred_img[y0:y0+cell_size, x0:x0+cell_size, 0].astype(int) + 100, 255
                    ).astype(np.uint8)
    
    if not MATPLOTLIB_AVAILABLE:
        # Simple horizontal concatenation with spacing
        spacing = np.ones((max(input_img.shape[0], target_img.shape[0], pred_img.shape[0]), 20, 3), dtype=np.uint8) * 255
        combined = np.concatenate([
            _pad_to_height(input_img, max(input_img.shape[0], target_img.shape[0], pred_img.shape[0])),
            spacing,
            _pad_to_height(target_img, max(input_img.shape[0], target_img.shape[0], pred_img.shape[0])),
            spacing,
            _pad_to_height(pred_img, max(input_img.shape[0], target_img.shape[0], pred_img.shape[0])),
        ], axis=1)
        return combined
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(input_img)
    axes[0].set_title('Input', fontsize=11)
    axes[0].axis('off')
    
    axes[1].imshow(target_img)
    axes[1].set_title('Target', fontsize=11)
    axes[1].axis('off')
    
    is_correct = np.array_equal(prediction, target) if prediction.shape == target.shape else False
    status = "✓ Correct" if is_correct else "✗ Incorrect"
    color = 'green' if is_correct else 'red'
    axes[2].imshow(pred_img)
    axes[2].set_title(f'Prediction ({status})', fontsize=11, color=color)
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Return combined image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return img_array


def _pad_to_height(img: np.ndarray, target_height: int) -> np.ndarray:
    """Pad image to target height with white."""
    if img.shape[0] >= target_height:
        return img
    pad_height = target_height - img.shape[0]
    padding = np.ones((pad_height, img.shape[1], 3), dtype=np.uint8) * 255
    return np.concatenate([img, padding], axis=0)


def save_grid_comparison(
    grids: List[Tuple[np.ndarray, str]],
    save_path: str,
    title: Optional[str] = None,
    cell_size: int = 20,
):
    """
    Save a comparison of multiple grids.
    
    Args:
        grids: List of (grid, label) tuples
        save_path: Path to save the image
        title: Optional overall title
        cell_size: Size of each cell
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot save grid comparison")
        return
    
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    if n == 1:
        axes = [axes]
    
    for ax, (grid, label) in zip(axes, grids):
        img = grid_to_image(grid, cell_size=cell_size)
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def grid_to_html_table(
    grid: Union[np.ndarray, List[List[int]]],
    cell_size: int = 20,
) -> str:
    """
    Convert ARC grid to HTML table with colored cells.
    
    Args:
        grid: ARC grid as numpy array or list of lists
        cell_size: Size of each cell in pixels
        
    Returns:
        HTML string for the table
    """
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    
    rows = []
    for row in grid:
        cells = []
        for val in row:
            val = int(val)
            color = ARC_COLORS_HEX[val] if 0 <= val < len(ARC_COLORS_HEX) else '#FFFFFF'
            # Use light text for dark backgrounds
            text_color = '#FFFFFF' if val in [0, 9] else '#000000'
            cell = (
                f'<td style="width:{cell_size}px;height:{cell_size}px;'
                f'background:{color};color:{text_color};text-align:center;'
                f'font-size:10px;border:1px solid #333;">{val}</td>'
            )
            cells.append(cell)
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    
    return '<table style="border-collapse:collapse;margin:5px;">' + ''.join(rows) + '</table>'
