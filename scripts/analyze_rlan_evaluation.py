#!/usr/bin/env python
"""
RLAN Evaluation Analysis Script - CISL Production Parity

Generates detailed analysis including:
1. Per-task visualizations (input, target, prediction)
2. Transformation type analysis (where RLAN succeeds/fails)
3. Error pattern analysis
4. HTML report for easy browsing

Usage:
    python scripts/analyze_rlan_evaluation.py --results evaluation_results/
    python scripts/analyze_rlan_evaluation.py --results evaluation_results/ --generate-html
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from sci_arc.evaluation import ARC_COLORS_HEX


def load_evaluation_results(results_dir: Path) -> Dict[str, Any]:
    """Load evaluation results from directory."""
    metrics_path = results_dir / 'metrics.json'
    detailed_path = results_dir / 'detailed_predictions' / 'all_predictions.json'
    
    results = {}
    
    if metrics_path.exists():
        with open(metrics_path) as f:
            results['metrics'] = json.load(f)
    
    if detailed_path.exists():
        with open(detailed_path) as f:
            results['detailed'] = json.load(f)
    
    return results


def classify_transformation(input_grid: List, target_grid: List) -> str:
    """
    Heuristically classify the transformation type.
    """
    input_arr = np.array(input_grid)
    target_arr = np.array(target_grid)
    
    ih, iw = input_arr.shape
    th, tw = target_arr.shape
    
    # Size change
    if ih != th or iw != tw:
        if th > ih or tw > iw:
            return 'scaling_up'
        elif th < ih or tw < iw:
            return 'scaling_down'
        else:
            return 'size_change'
    
    # Same size transformations
    input_colors = set(input_arr.flatten()) - {0}
    target_colors = set(target_arr.flatten()) - {0}
    
    # Check for rotation (90, 180, 270)
    if np.array_equal(np.rot90(input_arr), target_arr):
        return 'rotation_90'
    if np.array_equal(np.rot90(input_arr, 2), target_arr):
        return 'rotation_180'
    if np.array_equal(np.rot90(input_arr, 3), target_arr):
        return 'rotation_270'
    
    # Check for flips
    if np.array_equal(np.flipud(input_arr), target_arr):
        return 'flip_vertical'
    if np.array_equal(np.fliplr(input_arr), target_arr):
        return 'flip_horizontal'
    
    # Color changes only
    if input_colors != target_colors:
        # Check if structure is preserved
        input_mask = input_arr > 0
        target_mask = target_arr > 0
        if np.array_equal(input_mask, target_mask):
            return 'color_swap'
        else:
            return 'color_change'
    
    # Fill patterns
    input_nonzero = np.sum(input_arr > 0)
    target_nonzero = np.sum(target_arr > 0)
    if target_nonzero > input_nonzero * 1.5:
        return 'fill'
    
    # Default
    return 'complex'


def analyze_error_patterns(predictions: List[Dict]) -> Dict[str, Any]:
    """Analyze common error patterns."""
    analysis = {
        'total_tasks': len(predictions),
        'correct_tasks': 0,
        'incorrect_tasks': 0,
        'by_transformation': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_size_match': {'matched': 0, 'mismatched': 0},
        'by_pixel_accuracy_range': defaultdict(int),
        'background_collapse': 0,  # Predictions that are mostly background
        'common_errors': [],
    }
    
    for pred in predictions:
        is_correct = pred.get('is_correct', False)
        
        if is_correct:
            analysis['correct_tasks'] += 1
        else:
            analysis['incorrect_tasks'] += 1
        
        # Size match
        if pred.get('size_match', False):
            analysis['by_size_match']['matched'] += 1
        else:
            analysis['by_size_match']['mismatched'] += 1
        
        # Pixel accuracy ranges
        pixel_acc = pred.get('pixel_accuracy', 0)
        if pixel_acc >= 0.95:
            analysis['by_pixel_accuracy_range']['95-100%'] += 1
        elif pixel_acc >= 0.90:
            analysis['by_pixel_accuracy_range']['90-95%'] += 1
        elif pixel_acc >= 0.80:
            analysis['by_pixel_accuracy_range']['80-90%'] += 1
        elif pixel_acc >= 0.50:
            analysis['by_pixel_accuracy_range']['50-80%'] += 1
        else:
            analysis['by_pixel_accuracy_range']['<50%'] += 1
        
        # Transformation type analysis
        input_grid = pred.get('input_grid')
        target_grid = pred.get('target_grid')
        if input_grid and target_grid:
            transform_type = classify_transformation(input_grid, target_grid)
            analysis['by_transformation'][transform_type]['total'] += 1
            if is_correct:
                analysis['by_transformation'][transform_type]['correct'] += 1
        
        # Background collapse detection
        prediction_grid = pred.get('prediction_grid')
        if prediction_grid:
            pred_arr = np.array(prediction_grid)
            background_ratio = np.sum(pred_arr == 0) / pred_arr.size
            if background_ratio > 0.95:
                analysis['background_collapse'] += 1
    
    # Compute transformation accuracies
    analysis['transformation_accuracies'] = {}
    for transform_type, stats in analysis['by_transformation'].items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            analysis['transformation_accuracies'][transform_type] = {
                'accuracy': acc,
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    return analysis


def grid_to_html_table(grid: List[List[int]], cell_size: int = 20) -> str:
    """Convert a grid to an HTML table with colored cells."""
    rows = []
    for row in grid:
        cells = []
        for val in row:
            color = ARC_COLORS_HEX[val] if 0 <= val < len(ARC_COLORS_HEX) else '#FFFFFF'
            # Use light text for dark backgrounds
            text_color = '#FFFFFF' if val in [0, 9] else '#000000'
            cell = f'<td style="width:{cell_size}px;height:{cell_size}px;background:{color};color:{text_color};text-align:center;font-size:10px;border:1px solid #333;">{val}</td>'
            cells.append(cell)
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return '<table style="border-collapse:collapse;margin:5px;">' + ''.join(rows) + '</table>'


def generate_html_report(
    results_dir: Path,
    results: Dict[str, Any],
    analysis: Dict[str, Any],
    max_visualizations: int = 100
) -> str:
    """Generate an HTML report with visualizations."""
    
    predictions = results.get('detailed', {}).get('predictions', [])
    metrics = results.get('metrics', {})
    
    # Start HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RLAN Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #0074D9; }}
        .metric-label {{ font-size: 0.9em; color: #666; }}
        .task-card {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .task-correct {{ border-left: 4px solid #2ECC40; }}
        .task-incorrect {{ border-left: 4px solid #FF4136; }}
        .grids {{ display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }}
        .grid-container {{ text-align: center; }}
        .grid-label {{ font-weight: bold; margin-bottom: 5px; }}
        .transformation-table {{ width: 100%; border-collapse: collapse; }}
        .transformation-table th, .transformation-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        .transformation-table th {{ background: #f0f0f0; }}
        .accuracy-bar {{ height: 20px; background: #eee; border-radius: 10px; overflow: hidden; }}
        .accuracy-fill {{ height: 100%; background: #0074D9; }}
        .warning {{ background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .filter-buttons {{ margin: 10px 0; }}
        .filter-btn {{ padding: 8px 16px; margin: 2px; cursor: pointer; border: none; border-radius: 4px; }}
        .filter-btn.active {{ background: #0074D9; color: white; }}
        .filter-btn:not(.active) {{ background: #ddd; }}
    </style>
</head>
<body>
    <h1>RLAN Evaluation Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary">
        <h2>Summary Metrics</h2>
        <div class="metric">
            <div class="metric-value">{metrics.get('task_accuracy', 0)*100:.1f}%</div>
            <div class="metric-label">Task Accuracy ({analysis['correct_tasks']}/{analysis['total_tasks']})</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('pixel_accuracy', 0)*100:.1f}%</div>
            <div class="metric-label">Pixel Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('size_accuracy', 0)*100:.1f}%</div>
            <div class="metric-label">Size Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('non_background_accuracy', 0)*100:.1f}%</div>
            <div class="metric-label">Non-Background Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('mean_iou', 0)*100:.1f}%</div>
            <div class="metric-label">Mean IoU</div>
        </div>
        <div class="metric">
            <div class="metric-value">{analysis['background_collapse']}</div>
            <div class="metric-label">Background Collapse Cases</div>
        </div>
    </div>
'''
    
    # Warning if background collapse detected
    if analysis['total_tasks'] > 0 and analysis['background_collapse'] > analysis['total_tasks'] * 0.5:
        html += f'''
    <div class="warning">
        <strong>Warning:</strong> High rate of background collapse detected ({100 * analysis['background_collapse'] / analysis['total_tasks']:.1f}% of predictions are mostly background).
        This indicates the model may be stuck in a local minimum. Consider:
        <ul>
            <li>Using infinite data generation (cache_samples=false)</li>
            <li>Adding class weighting or focal loss</li>
            <li>Longer training with curriculum learning</li>
        </ul>
    </div>
'''
    
    # Transformation type analysis
    html += '''
    <div class="summary">
        <h2>Transformation Type Analysis</h2>
        <p>RLAN's relational encoder should excel at recognizing transformation patterns. Here's how it performs by transformation type:</p>
        <table class="transformation-table">
            <tr>
                <th>Transformation Type</th>
                <th>Accuracy</th>
                <th>Tasks</th>
                <th>Performance</th>
            </tr>
'''
    
    sorted_transforms = sorted(
        analysis['transformation_accuracies'].items(),
        key=lambda x: x[1]['total'],
        reverse=True
    )
    
    for transform_type, stats in sorted_transforms:
        acc_pct = stats['accuracy'] * 100
        fill_width = int(acc_pct)
        html += f'''
            <tr>
                <td><strong>{transform_type.replace('_', ' ').title()}</strong></td>
                <td>{acc_pct:.1f}%</td>
                <td>{stats['correct']}/{stats['total']}</td>
                <td>
                    <div class="accuracy-bar">
                        <div class="accuracy-fill" style="width:{fill_width}%;"></div>
                    </div>
                </td>
            </tr>
'''
    
    html += '''
        </table>
    </div>
'''
    
    # Pixel accuracy distribution
    html += '''
    <div class="summary">
        <h2>Pixel Accuracy Distribution</h2>
        <table class="transformation-table">
            <tr>
                <th>Range</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
'''
    
    for range_name, count in sorted(analysis['by_pixel_accuracy_range'].items()):
        pct = 100 * count / analysis['total_tasks'] if analysis['total_tasks'] > 0 else 0
        html += f'''
            <tr>
                <td>{range_name}</td>
                <td>{count}</td>
                <td>{pct:.1f}%</td>
            </tr>
'''
    
    html += '''
        </table>
    </div>
'''
    
    # Individual task visualizations
    html += '''
    <div class="summary">
        <h2>Task Predictions</h2>
        <div class="filter-buttons">
            <button class="filter-btn active" onclick="filterTasks('all')">All</button>
            <button class="filter-btn" onclick="filterTasks('correct')">Correct Only</button>
            <button class="filter-btn" onclick="filterTasks('incorrect')">Incorrect Only</button>
        </div>
'''
    
    # Sort: incorrect first (more interesting for debugging)
    sorted_preds = sorted(predictions, key=lambda x: (x.get('is_correct', False), x.get('pixel_accuracy', 0)))
    
    for i, pred in enumerate(sorted_preds[:max_visualizations]):
        task_id = pred.get('task_id', f'task_{i}')
        is_correct = pred.get('is_correct', False)
        pixel_acc = pred.get('pixel_accuracy', 0)
        non_bg_acc = pred.get('non_background_accuracy', 0)
        status_class = 'task-correct' if is_correct else 'task-incorrect'
        status_text = '[OK]' if is_correct else '[X]'
        filter_class = 'correct' if is_correct else 'incorrect'
        
        input_grid = pred.get('input_grid', [[0]])
        target_grid = pred.get('target_grid', [[0]])
        prediction_grid = pred.get('prediction_grid', [[0]])
        
        transform_type = classify_transformation(input_grid, target_grid)
        
        html += f'''
        <div class="task-card {status_class}" data-correct="{filter_class}">
            <h3>{status_text} Task: {task_id}</h3>
            <p>Pixel Accuracy: {pixel_acc*100:.1f}% | Non-BG Accuracy: {non_bg_acc*100:.1f}% | Transformation: {transform_type.replace('_', ' ').title()}</p>
            <div class="grids">
                <div class="grid-container">
                    <div class="grid-label">Input</div>
                    {grid_to_html_table(input_grid)}
                </div>
                <div class="grid-container">
                    <div class="grid-label">Target</div>
                    {grid_to_html_table(target_grid)}
                </div>
                <div class="grid-container">
                    <div class="grid-label">Prediction</div>
                    {grid_to_html_table(prediction_grid)}
                </div>
            </div>
        </div>
'''
    
    if len(predictions) > max_visualizations:
        html += f'''
        <p><em>Showing {max_visualizations} of {len(predictions)} tasks. See individual JSON files for complete data.</em></p>
'''
    
    html += '''
    </div>
    
    <script>
        function filterTasks(filter) {
            // Update button styles
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // Filter cards
            document.querySelectorAll('.task-card').forEach(card => {
                if (filter === 'all') {
                    card.style.display = 'block';
                } else if (card.dataset.correct === filter) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
'''
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Analyze RLAN evaluation results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to evaluation results directory')
    parser.add_argument('--generate-html', action='store_true',
                        help='Generate HTML report with visualizations')
    parser.add_argument('--max-viz', type=int, default=100,
                        help='Maximum number of task visualizations in HTML')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for HTML report')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("RLAN Evaluation Analysis")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    
    # Load results
    results = load_evaluation_results(results_dir)
    
    if not results:
        print("Error: No results found in directory")
        sys.exit(1)
    
    # Analyze
    predictions = results.get('detailed', {}).get('predictions', [])
    if predictions:
        print(f"Analyzing {len(predictions)} predictions...")
        analysis = analyze_error_patterns(predictions)
    else:
        print("No detailed predictions found, using metrics only")
        analysis = {
            'total_tasks': results.get('metrics', {}).get('total_tasks', 0),
            'correct_tasks': results.get('metrics', {}).get('correct_tasks', 0),
            'incorrect_tasks': 0,
            'by_transformation': {},
            'transformation_accuracies': {},
            'background_collapse': 0,
            'by_pixel_accuracy_range': {},
            'by_size_match': {'matched': 0, 'mismatched': 0},
        }
        analysis['incorrect_tasks'] = analysis['total_tasks'] - analysis['correct_tasks']
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Tasks: {analysis['total_tasks']}")
    if analysis['total_tasks'] > 0:
        print(f"Correct: {analysis['correct_tasks']} ({100*analysis['correct_tasks']/analysis['total_tasks']:.1f}%)")
    else:
        print("Correct: 0")
    print(f"Incorrect: {analysis['incorrect_tasks']}")
    print(f"Background Collapse Cases: {analysis['background_collapse']}")
    
    print("\n--- Transformation Type Analysis ---")
    print("(RLAN's relational encoder should excel at recognizing these patterns)")
    print()
    for transform_type, stats in sorted(analysis['transformation_accuracies'].items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"  {transform_type.replace('_', ' ').title():20s}: {stats['accuracy']*100:5.1f}% ({stats['correct']}/{stats['total']})")
    
    print("\n--- Pixel Accuracy Distribution ---")
    for range_name, count in sorted(analysis['by_pixel_accuracy_range'].items()):
        pct = 100 * count / analysis['total_tasks'] if analysis['total_tasks'] > 0 else 0
        print(f"  {range_name:12s}: {count:4d} ({pct:5.1f}%)")
    
    # Save analysis
    analysis_path = results_dir / 'analysis.json'
    
    # Convert defaultdict to regular dict for JSON serialization
    analysis_serializable = {
        k: dict(v) if isinstance(v, defaultdict) else v
        for k, v in analysis.items()
    }
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis_serializable, f, indent=2)
    print(f"\nSaved analysis to {analysis_path}")
    
    # Generate HTML report
    if args.generate_html:
        print("\nGenerating HTML report...")
        html = generate_html_report(results_dir, results, analysis, args.max_viz)
        
        output_path = Path(args.output) if args.output else results_dir / 'report.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Saved HTML report to {output_path}")
        print(f"Open in browser: file:///{output_path.absolute()}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
