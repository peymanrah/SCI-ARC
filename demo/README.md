# RLAN Interactive Demo

An interactive, self-contained HTML experience demonstrating the **Recurrent Latent-Action Network (RLAN)** for visual reasoning on ARC puzzles.

## 🚀 Quick Start

### Running Locally

1. Navigate to the `demo` folder
2. Open `index.html` in any modern web browser
3. That's it! No build step required.

```bash
cd demo
# On Windows:
start index.html
# On Mac:
open index.html
# On Linux:
xdg-open index.html
```

### Running with Local Server (Recommended)

For best results (avoids CORS issues with data files):

```bash
cd demo
# Python 3
python -m http.server 8000
# Then open http://localhost:8000

# Or with Node.js
npx serve .
```

## 📁 Project Structure

```
demo/
├── index.html              # Main scrollytelling page
├── css/
│   └── style.css           # All styles (modern, responsive)
├── js/
│   ├── app.js              # Main orchestration
│   └── components/
│       ├── GridRenderer.js     # ARC grid visualization
│       ├── HeatmapOverlay.js   # DSC attention heatmap
│       ├── CoordinateToggle.js # Abs/relative coordinates
│       ├── SolverStepper.js    # Iteration stepping
│       ├── ChartPanel.js       # Training metrics charts
│       ├── ArchitectureViz.js  # RLAN vs TRM diagrams
│       └── PresenterMode.js    # Speaker notes & pacing
├── data/
│   ├── puzzles.json            # Sample ARC puzzles
│   └── training_metrics.json   # Training data
└── assets/
    ├── rlan-logo.svg           # Logo
    └── icon-*.svg              # UI icons
```

## 🎯 Features

### Interactive Components

1. **ARC Grid Sandbox**
   - Render any ARC puzzle with the standard color palette
   - Toggle between training examples
   - "Show Solution" reveal animation

2. **DSC Attention Heatmap**
   - Visualize where the model focuses attention
   - ⚠️ Currently uses proxy heuristics (see below for real data)

3. **Coordinate Toggle**
   - Switch between absolute and anchor-relative coordinates
   - Click any cell to set as anchor
   - Demonstrates translation invariance

4. **Solver Stepper**
   - Step through iteration 0 → final output
   - See confidence and halt probability
   - Auto-play animation

5. **Training Dashboard**
   - Loss curves, accuracy, solver improvement
   - Attention entropy (50× sharpening visualization)
   - Evidence callouts with key claims

6. **Architecture Comparison**
   - RLAN vs Transformer side-by-side
   - Toggle highlights: Representation, Attention, Compute

7. **LLM Integration Section**
   - Three integration strategies
   - API pseudocode
   - Architecture diagram

### Presenter Mode

Press **Ctrl+P** or click the presenter icon to enable:
- Speaker notes for each section
- 30-second / 2-minute / 5-minute talk tracks
- Progress indicator and pacing guide

## 🔧 Customization

### Adding Your Own Puzzles

Edit `data/puzzles.json`:

```json
{
  "name": "My Custom Puzzle",
  "train": [
    {
      "input": [[0,1,0],[1,1,1],[0,1,0]],
      "output": [[0,2,0],[2,2,2],[0,2,0]]
    }
  ],
  "test": {
    "input": [[0,1,1,0],[1,1,1,1],[0,1,1,0]],
    "output": [[0,2,2,0],[2,2,2,2],[0,2,2,0]]
  }
}
```

### Using Real Training Metrics

Edit `data/training_metrics.json` or generate from your training logs:

```python
# In your training loop:
import json

metrics = {
    "epochs": [1, 2, 3, ...],
    "trainLoss": [...],
    "valLoss": [...],
    "cellAccuracy": [...],  # Percentage
    "exactMatch": [...],
    "solverImprovement": [...],
    "attentionEntropy": [...]
}

with open('demo/data/training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

### Logging Real Model Internals

To replace proxy visualizations with real data:

1. **DSC Attention Maps**
   ```python
   # In DSC forward pass:
   torch.save({
       'attention_weights': alpha.cpu(),
       'anchor_positions': anchor_idx.cpu()
   }, 'dsc_attention.pt')
   ```

2. **Solver Iterations**
   ```python
   # In RecursiveSolver:
   iteration_states = []
   for t in range(max_iter):
       h = self.step(h, context)
       logits = self.output_head(h)
       iteration_states.append({
           'step': t,
           'logits': logits.argmax(dim=1).cpu(),
           'confidence': logits.softmax(dim=1).max().item(),
           'halt_prob': self.halt_predictor(h).item()
       })
   ```

3. **Export to JSON** for demo consumption

## 📡 Offline Usage

The demo uses CDN links for:
- **KaTeX** (math rendering): `cdn.jsdelivr.net/npm/katex`
- **Plotly** (charts): `cdn.plot.ly/plotly`

For fully offline usage:

1. Download the libraries:
   ```bash
   # KaTeX
   wget https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css
   wget https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js
   wget https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js
   
   # Plotly
   wget https://cdn.plot.ly/plotly-2.27.0.min.js
   ```

2. Update `index.html` to use local paths:
   ```html
   <link rel="stylesheet" href="lib/katex.min.css">
   <script src="lib/katex.min.js"></script>
   <script src="lib/plotly-2.27.0.min.js"></script>
   ```

## 🎨 Theming

The demo uses CSS custom properties for easy theming. Edit in `css/style.css`:

```css
:root {
    /* Change accent colors */
    --accent-primary: #6366f1;
    --accent-secondary: #8b5cf6;
    
    /* Change background */
    --bg-primary: #0f0f1a;
    --bg-secondary: #1a1a2e;
    
    /* ARC colors are standard, but can be customized */
    --arc-1: #0074D9;  /* Blue */
    --arc-2: #FF4136;  /* Red */
    /* ... */
}
```

## 📊 Training Evidence Summary

From the included training data:

| Metric | Epoch 1 | Epoch 3 | Change |
|--------|---------|---------|--------|
| Cell Accuracy | 66.5% | 80.2% | +13.7% |
| Exact Match | 1.5% | 13.9% | +12.4% |
| Solver Improvement | -0.6% | +84.9% | +85.5% |
| Attention Entropy | 3.07 | 0.06 | 50× sharper |

## 🐛 Troubleshooting

### Charts not rendering
- Check browser console for Plotly errors
- Ensure internet connection for CDN
- Try running with local server

### Math equations not rendering
- KaTeX needs to load from CDN
- Check console for 403/404 errors
- Equations will render as plain text if KaTeX fails

### Grid colors wrong
- Check that color values are 0-9
- Non-numeric values default to black

## 📝 License

This demo is part of the SCI-ARC research project.

---

Built with ❤️ for RLAN research
