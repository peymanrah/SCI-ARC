#!/usr/bin/env python3
"""Find ARC tasks with large grids and grid expansion."""

import json
from pathlib import Path

train_dir = Path('./data/arc-agi/data/training')
tasks_info = []

for task_file in list(train_dir.glob('*.json')):
    with open(task_file) as f:
        data = json.load(f)
    
    # Get max grid sizes
    max_in_h, max_in_w = 0, 0
    max_out_h, max_out_w = 0, 0
    for ex in data['train'] + data['test']:
        inp = ex['input']
        out = ex['output']
        max_in_h = max(max_in_h, len(inp))
        max_in_w = max(max_in_w, len(inp[0]) if inp else 0)
        max_out_h = max(max_out_h, len(out))
        max_out_w = max(max_out_w, len(out[0]) if out else 0)
    
    max_in = max(max_in_h, max_in_w)
    max_out = max(max_out_h, max_out_w)
    
    tasks_info.append({
        'id': task_file.stem,
        'max_in': max_in,
        'max_out': max_out,
        'in_area': max_in_h * max_in_w,
        'out_area': max_out_h * max_out_w,
        'expansion': max_out > max_in
    })

# Sort by max output size
tasks_info.sort(key=lambda x: x['max_out'], reverse=True)

print('Top 15 tasks by output grid size:')
for t in tasks_info[:15]:
    exp = ' [EXPANSION]' if t['expansion'] else ''
    print(f"  {t['id']}: in={t['max_in']:2d} out={t['max_out']:2d}{exp}")

print()
print('Grid expansion tasks (output > input):')
expansion_tasks = [t for t in tasks_info if t['expansion']]
expansion_tasks.sort(key=lambda x: x['out_area'] / (x['in_area'] + 1), reverse=True)
for t in expansion_tasks[:15]:
    ratio = t['out_area'] / (t['in_area'] + 1)
    print(f"  {t['id']}: in={t['max_in']:2d} out={t['max_out']:2d} expansion_ratio={ratio:.1f}x")
