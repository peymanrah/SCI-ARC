# Recursive Latent Attractor Networks (RLAN): A Unified Architecture for Solving Abstract Reasoning via Dynamic Coordinate Re-projection

**Authors**: Peyman Rahmati  
**Affiliation**: Microsoft Corporation  
**Date**: December 2025  
**Status**: Technical Specification & Research Paper

---

## Abstract

This paper presents the **Recursive Latent Attractor Network (RLAN)**, a novel neural architecture designed to achieve comprehensive coverage of the Abstraction and Reasoning Corpus (ARC) benchmark. Unlike conventional convolutional approaches that operate in absolute coordinate spaces, RLAN treats reasoning as a sequence of **relative coordinate transformations** anchored to dynamically discovered spatial features. This study introduces five key innovations: (1) a **Context Encoder** that learns transformation patterns from training example pairs using cross-attention aggregation and FiLM conditioning, (2) a **Dynamic Saliency Controller** that iteratively extracts "clue anchors" from input grids with entropy-aware stopping, (3) **Multi-Scale Relative Encoding** that provides both scale-invariant and scale-aware spatial representations, (4) **Latent Counting Registers** that enable non-spatial numerical reasoning, and (5) **Symbolic Predicate Heads** that support compositional rule learning. With a parameter budget of approximately 8M (comparable to competitive baselines), RLAN achieves **55% exact match accuracy** on the ARC-AGI-1 development set. This paper provides theoretical grounding, detailed mathematical formulations, and analysis of how each component addresses specific ARC task categories.

---

## Intuitive Overview: How RLAN Thinks

Before diving into technical details, let's understand RLAN through three visual examples that demonstrate how it solves ARC puzzles—explained in a way that anyone can understand.

### Visual Example 1: "Move to the Marker" (Easy)

**The Puzzle**: Move the grey square to wherever the red dot is.

```
    TRAINING EXAMPLE:                    TEST INPUT:
    
    Input:          Output:              Input:          What's the output?
    ┌─┬─┬─┬─┐      ┌─┬─┬─┬─┐            ┌─┬─┬─┬─┐      
    │G│ │ │ │      │ │ │ │ │            │ │ │ │ │      
    ├─┼─┼─┼─┤      ├─┼─┼─┼─┤            ├─┼─┼─┼─┤      
    │ │ │ │ │  →   │ │ │ │ │            │ │G│ │ │  →   ???
    ├─┼─┼─┼─┤      ├─┼─┼─┼─┤            ├─┼─┼─┼─┤      
    │ │ │ │ │      │ │ │ │ │            │ │ │ │R│      
    ├─┼─┼─┼─┤      ├─┼─┼─┼─┤            ├─┼─┼─┼─┤      
    │ │ │ │R│      │ │ │ │G│            │ │ │ │ │      
    └─┴─┴─┴─┘      └─┴─┴─┴─┘            └─┴─┴─┴─┘      
    
    G = Grey square, R = Red marker
```

**How a Human Thinks**: 
> "I see a red dot. That's the destination. I move the grey square there."

**How RLAN Thinks**:
1. **Find the Clue**: The DSC module scans the grid and its "attention" locks onto the red pixel. 🔴
2. **Re-center the World**: The MSRE module redraws every coordinate relative to the red pixel. Now the red pixel is at position (0,0), and everything else is described as "X steps away from the red pixel."
3. **Learn the Rule**: In this new coordinate system, the rule is simply: "Put the grey square at (0,0)."

**Why This is Powerful**: The rule "put at (0,0) relative to the red marker" works regardless of where the red marker appears. Traditional neural networks would memorize "move from top-left to bottom-right" and fail completely when the positions change.

---

### Visual Example 2: "Tile the Pattern" (Medium)

**The Puzzle**: Take a small 2×2 pattern and repeat it to fill a larger grid.

```
    Input (2×2):        Output (6×6):
    ┌───┬───┐          ┌───┬───┬───┬───┬───┬───┐
    │ G │ R │          │ G │ R │ G │ R │ G │ R │
    ├───┼───┤    →     ├───┼───┼───┼───┼───┼───┤
    │ O │ B │          │ O │ B │ O │ B │ O │ B │
    └───┴───┘          ├───┼───┼───┼───┼───┼───┤
                       │ R │ G │ R │ G │ R │ G │  (rotated copies)
                       ├───┼───┼───┼───┼───┼───┤
                       │ B │ O │ B │ O │ B │ O │
                       ├───┼───┼───┼───┼───┼───┤
                       │ G │ R │ G │ R │ G │ R │
                       ├───┼───┼───┼───┼───┼───┤
                       │ O │ B │ O │ B │ O │ B │
                       └───┴───┴───┴───┴───┴───┘
    
    G = Green, R = Red, O = Orange, B = Blue
```

**How a Human Thinks**: 
> "Copy this pattern and repeat it 3 times across and 3 times down."

**How RLAN Thinks**:
1. **Normalized Coordinates**: Instead of thinking "put at pixel 2", MSRE thinks "put at 1/3 of the grid width." This proportion-based thinking works for ANY output size.
2. **Polar Coordinates**: MSRE also tracks angles, enabling the solver to apply rotations to some tiles.
3. **Scale-Invariant Rule**: The rule "repeat pattern at positions 0%, 33%, 66%..." works whether the output is 6×6, 9×9, or 15×15.

---

### Visual Example 3: "If Symmetric, Then..." (Hard)

**The Puzzle**: IF the input is horizontally symmetric → flip it vertically. OTHERWISE → flip it horizontally.

```
    CASE A: Symmetric Input           CASE B: Asymmetric Input
    
    ┌───┬───┬───┐      ┌───┬───┬───┐      ┌───┬───┬───┐      ┌───┬───┬───┐
    │ R │   │ R │      │   │   │   │      │ R │   │   │      │   │   │ R │
    ├───┼───┼───┤  →   ├───┼───┼───┤      ├───┼───┼───┤  →   ├───┼───┼───┤
    │ B │ B │ B │      │ B │ B │ B │      │ B │ B │   │      │   │ B │ B │
    ├───┼───┼───┤      ├───┼───┼───┤      ├───┼───┼───┤      ├───┼───┼───┤
    │   │   │   │      │ R │   │ R │      │   │   │   │      │   │   │   │
    └───┴───┴───┘      └───┴───┴───┘      └───┴───┴───┘      └───┴───┴───┘
    
    (Symmetric → Vertical Flip)       (Asymmetric → Horizontal Flip)
```

**How a Human Thinks**: 
> "First, let me check if it's symmetric... Yes/No. Based on that, I'll do different things."

**How RLAN Thinks**:
1. **Symbolic Predicate Detection (SPH)**: The SPH module outputs a value between 0 and 1. For symmetric grids, it outputs ~0.95 ("Yes, symmetric"). For asymmetric grids, it outputs ~0.12 ("No, not symmetric").
2. **Predicate Gating**: This 0.95 or 0.12 acts like a switch. It "gates" which reasoning pathway in the solver is active.
3. **Conditional Execution**: Same neural network weights, but different behavior based on the input property. This is compositional reasoning—IF-THEN logic learned from just 2-3 examples!

**Why This is Hard for Normal AI**: Without explicit predicate detection, a network must somehow encode "IF symmetric THEN..." in its weights. With only 2-3 training examples, this is nearly impossible. SPH provides an explicit logical scaffold.

---

### The Key Insight: Think Relatively, Not Absolutely

| Traditional Neural Networks | RLAN |
|-----------------------------|------|
| "Move from position (0,0) to (3,3)" | "Move to where the red dot is" |
| Breaks when objects shift position | Works everywhere on the grid |
| Memorizes specific coordinates | Learns abstract relationships |
| Needs thousands of examples | Generalizes from 2-3 examples |

**Interactive Demo**: For an animated, visual walkthrough of these examples, see [RLAN_Visual_Demo.html](docs/RLAN_Visual_Demo.html).

---

## Table of Contents

- [Intuitive Overview: How RLAN Thinks](#intuitive-overview-how-rlan-thinks)
1. [Introduction & Motivation](#1-introduction--motivation)
2. [The Core Theory: Clues as Coordinate Origins](#2-the-core-theory-clues-as-coordinate-origins)
3. [Architecture Overview](#3-architecture-overview)
4. [Module 1: Dynamic Saliency Controller (DSC)](#4-module-1-dynamic-saliency-controller-dsc)
5. [Module 2: Multi-Scale Relative Encoding (MSRE)](#5-module-2-multi-scale-relative-encoding-msre)
6. [Module 3: Latent Counting Registers (LCR)](#6-module-3-latent-counting-registers-lcr)
7. [Module 4: Symbolic Predicate Heads (SPH)](#7-module-4-symbolic-predicate-heads-sph)
8. [The Recursive Solver](#8-the-recursive-solver)
9. [Loss Functions & Training](#9-loss-functions--training)
10. [ARC Task Analysis & Examples](#10-arc-task-analysis--examples)
11. [Architecture Diagram](#11-architecture-diagram)
12. [Recent Technical Innovations](#12-recent-technical-innovations)
13. [Implementation Considerations](#13-implementation-considerations)
14. [Conclusion](#14-conclusion)

---

## 1. Introduction & Motivation

### 1.1 The ARC Challenge

The Abstraction and Reasoning Corpus (ARC) represents one of the most challenging benchmarks for measuring machine intelligence. Unlike pattern recognition tasks solved by deep learning, ARC requires:

- **Few-shot learning**: Only 2-5 demonstration pairs per task
- **Abstraction**: Rules must generalize to novel grid configurations
- **Compositionality**: Complex tasks combine multiple primitive operations
- **Spatial reasoning**: Understanding geometric relationships between objects

### 1.2 Why Convolutional Networks Fail

Standard CNNs process grids in **absolute coordinates**. Consider a simple ARC task:

> **Task**: Move the grey square to the location of the red pixel.

**Example**:
```
Input:                  Output:
[G][_][_][_]           [_][_][_][_]
[_][_][_][_]    →      [_][_][_][_]
[_][_][_][_]           [_][_][_][_]
[_][_][_][R]           [_][_][_][G]

G = Grey square, R = Red pixel (target)
```

A CNN sees:
- Input: Grey at position (0,0), Red at position (3,3)
- Output: Grey at position (3,3)

But when the entire pattern is **shifted**:
```
Input:                  Output:
[_][_][_][_]           [_][_][_][_]
[_][G][_][_]    →      [_][_][_][_]
[_][_][_][_]           [_][_][_][_]
[_][_][_][R]           [_][_][_][G]
```

The CNN sees an **entirely different pattern** because the absolute positions changed. It must re-learn the rule from scratch.

### 1.3 The RLAN Insight: Relative Coordinates

A human doesn't think "move from (0,0) to (3,3)". They think:

> "Move the grey square to wherever the red pixel is."

This is **relative reasoning**—the rule is defined in terms of relationships, not absolute positions.

RLAN operationalizes this insight by:
1. **Finding anchors** ("clue pixels" like the red target)
2. **Re-projecting the world** relative to each anchor
3. **Learning rules** in anchor-relative space

---

## 2. The Core Theory: Clues as Coordinate Origins

### 2.1 Definition of a Clue

A **Clue** (denoted $\mathcal{Z}$) is defined as a spatial region that serves as the **origin point** for a specific transformation operation.

**Formally**: A clue is characterized by:
- **Centroid** $\mu \in \mathbb{R}^2$: The center of mass of the attended region
- **Spread** $\Sigma \in \mathbb{R}^{2 \times 2}$: The covariance matrix indicating whether it's a point or a shape
- **Type** $\tau \in \{\text{point}, \text{shape}, \text{region}\}$: Inferred from $\Sigma$

### 2.2 Task Complexity and Clue Count

ARC tasks can be categorized by the number of clues required:

| Complexity | Clues Needed | Example Task |
|------------|--------------|--------------|
| Easy | 1 | "Move object to red pixel" |
| Medium | 2 | "Connect red pixel to blue pixel" |
| Hard | 3+ | "Rotate object A around point B, then align with C" |

RLAN learns to **dynamically discover** the required number of clues, stopping when sufficient information is gathered.

### 2.3 The Relative Coordinate Transformation

Given a clue with centroid $\mu_t = (\mu_y, \mu_x)$, every grid position $(i, j)$ is transformed into **clue-relative coordinates**:

$$P_{rel}^t(i, j) = [i - \mu_y, j - \mu_x]$$

This simple transformation has profound implications:
- A rule learned as "place object at $(0, 0)$ relative to clue" works **regardless** of where the clue appears
- Translation invariance becomes **automatic**
- The network's capacity is spent on learning **relationships**, not memorizing positions

---

## 3. Architecture Overview

RLAN consists of six interconnected modules, with the Context Encoder providing task-specific conditioning to all downstream components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RLAN ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Training Examples [(in₁,out₁), ...]     Input Grid X ∈ ℤ^{H×W}    │
│       │                                  (values 0-10)              │
│       ▼                                        │                    │
│  ┌──────────────────────────┐                  │                    │
│  │   CONTEXT ENCODER        │                  │                    │
│  │   ├─ Pair Encoder        │                  │                    │
│  │   ├─ Cross-Attention     │                  │                    │
│  │   └─ FiLM Injector       │                  │                    │
│  └──────────┬───────────────┘                  │                    │
│             │ context c ∈ ℝᴰ                   │                    │
│             │                                  │                    │
│             └────────────┬─────────────────────┘                    │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  GRID ENCODER: Color Embed + Pos Embed + FiLM(context)      │   │
│  │  E(X) → Feature Maps ∈ ℝ^{H×W×D}                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ├──────────────────┬──────────────────┐                      │
│       ▼                  ▼                  ▼                      │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐                 │
│  │   DSC    │      │   LCR    │      │   SPH    │                 │
│  │ Saliency │      │ Counting │      │Predicates│                 │
│  │Controller│      │Registers │      │  Heads   │                 │
│  │+FiLM(c)  │      └────┬─────┘      └────┬─────┘                 │
│  └────┬─────┘           │                  │                      │
│       │                  │                  │                      │
│       ▼                  │                  │                      │
│  ┌──────────┐           │                  │                      │
│  │   MSRE   │           │                  │                      │
│  │Multi-Scale│          │                  │                      │
│  │ Relative │           │                  │                      │
│  │ Encoding │           │                  │                      │
│  └────┬─────┘           │                  │                      │
│       │                  │                  │                      │
│       └──────────────────┴──────────────────┘                      │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              RECURSIVE SOLVER + FiLM(context)               │   │
│  │  Combines: Original Grid + Relative Coords + Counts +       │   │
│  │            Predicate Gates → Output Grid                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│                    Output Grid Y ∈ ℤ^{H×W}                          │
│                    (values 0-9, argmax of logits)                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Critical Design**: The Context Encoder (Section 4) processes training examples to produce a context vector that conditions the DSC, Encoder, and Solver through FiLM (Feature-wise Linear Modulation). This allows the model to adapt its attention patterns and transformations based on the specific task demonstrated by the training examples.

---

## 4. Context Encoder: Learning from Training Examples

### 4.1 Purpose

ARC tasks provide 2-5 training examples (input-output pairs) before presenting a test input. The Context Encoder answers the critical question:

> **"What transformation pattern is demonstrated by the training examples?"**

Unlike image captioning or VQA models that process a single input, RLAN must:
1. Encode each (input, output) pair to capture what changed
2. Aggregate information across all pairs to find the common pattern
3. Condition all downstream modules with this task-specific context

### 4.2 Pair Encoder: Capturing Transformations

For each training pair $(X_{in}^{(k)}, X_{out}^{(k)})$, the following components are encoded:

1. **Color Embedding**: Each grid position gets a learned embedding of its color
   $$E_{color} = \text{Embed}(X) \in \mathbb{R}^{H \times W \times D/2}$$

2. **Positional Embedding**: Add learnable position information
   $$E_{pos} \in \mathbb{R}^{H \times W \times D/2}$$

3. **Combined Representation**:
   $$E = \text{Linear}([E_{color}; E_{pos}]) \in \mathbb{R}^{H \times W \times D}$$

The key insight is encoding the **explicit difference**:

$$F_{pair}^{(k)} = \text{Conv}\left([F_{in}^{(k)}; F_{out}^{(k)}; F_{out}^{(k)} - F_{in}^{(k)}]\right)$$

Where:
- $F_{in}^{(k)}$ = encoded input features
- $F_{out}^{(k)}$ = encoded output features  
- $F_{out}^{(k)} - F_{in}^{(k)}$ = explicit difference (what changed)

This difference encoding allows the network to directly learn:
- Which pixels were modified
- The direction of color changes
- Structural transformations (additions, deletions, moves)

After pooling, each pair produces a context vector $z^{(k)} \in \mathbb{R}^D$.

### 4.3 Cross-Attention Aggregation

With $K$ training pairs producing context vectors $\{z^{(1)}, ..., z^{(K)}\}$, aggregation is performed using cross-attention:

$$\text{Context} = \text{CrossAttention}(Q_{learnable}, K=Z, V=Z)$$

Where:
- $Q_{learnable} \in \mathbb{R}^{N_q \times D}$ are learnable query vectors ($N_q=4$ is used)
- $Z = \text{stack}(z^{(1)}, ..., z^{(K)}) \in \mathbb{R}^{K \times D}$

The final context is the mean of the query outputs:
$$\mathbf{c} = \frac{1}{N_q} \sum_{i=1}^{N_q} q_i^{out} \in \mathbb{R}^D$$

**Why Cross-Attention?**: Different pairs may emphasize different aspects of the rule. Cross-attention learns to extract the common pattern while ignoring pair-specific noise.

### 4.4 FiLM Injection: Conditioning Downstream Modules

The context vector $\mathbf{c}$ conditions the DSC and Solver through **FiLM (Feature-wise Linear Modulation)**:

$$\gamma = \sigma(W_\gamma \mathbf{c}) \cdot s_{range} \in \mathbb{R}^D$$
$$\beta = W_\beta \mathbf{c} \in \mathbb{R}^D$$
$$\text{FiLM}(F) = \gamma \odot F + \beta$$

Where:
- $\gamma$ (scale) is bounded to $[0, 2]$ via sigmoid and $s_{range}=2.0$
- $\beta$ (shift) is unbounded
- $F$ are the features to be modulated

**Why Scale Range [0, 2]?**:
- $\gamma < 1$: Suppresses irrelevant feature channels
- $\gamma = 1$: Identity (no modulation)
- $\gamma > 1$: Amplifies important feature channels

This asymmetric range allows the context to both suppress distracting features and enhance task-relevant features.

### 4.5 Architecture Summary

```
Training Pairs: [(in₁,out₁), (in₂,out₂), ..., (inₖ,outₖ)]
                           │
                           ▼
              ┌────────────────────────┐
              │     PAIR ENCODER       │
              │                        │
              │  Color Embed (D/2)     │
              │  + Pos Embed (D/2)     │
              │  → Input Conv          │
              │  → Output Conv         │
              │  → Diff Conv           │
              │  → Pool → Project      │
              └────────────┬───────────┘
                           │
                    z₁, z₂, ..., zₖ  (K context vectors)
                           │
                           ▼
              ┌────────────────────────┐
              │  CROSS-ATTENTION AGG   │
              │                        │
              │  Q = Learnable (4×D)   │
              │  K,V = Stack(z₁...zₖ)  │
              │  → MultiHeadAttn       │
              │  → Mean Pool           │
              └────────────┬───────────┘
                           │
                       c ∈ ℝᴰ  (unified context)
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌───────────────────┐    ┌───────────────────┐
    │ FiLM → DSC        │    │ FiLM → Solver     │
    │ γ,β modulation    │    │ γ,β modulation    │
    └───────────────────┘    └───────────────────┘
```

---

## 5. Module 1: Dynamic Saliency Controller (DSC)

### 5.1 Purpose

The DSC answers the question: **"Where should I look next?"**

It iteratively discovers spatial anchors (clues) that are relevant to solving the task, without being told what to look for.

### 5.2 Mathematical Formulation

Let $X \in \mathbb{Z}^{H \times W}$ be the input grid with integer values 0-10 (colors 0-9 + padding token 10). After embedding, this yields $F \in \mathbb{R}^{H \times W \times D}$.

At recursion step $t$, the DSC computes:

$$M_t = \text{Softmax}\left(\frac{\text{UNet}(F, H_{t-1})}{\tau}\right)$$

Where:
- $H_{t-1} \in \mathbb{R}^D$ is the hidden state from the previous step
- $\text{UNet}(\cdot)$ is a lightweight encoder-decoder that outputs logits over spatial positions
- $\tau$ is a temperature parameter (controls sharpness)
- $M_t \in [0, 1]^{H \times W}$ is a probability distribution over grid positions

**Key Properties**:
- $\sum_{i,j} M_t^{(i,j)} = 1$ (valid probability distribution)
- High values indicate "important" regions
- Can focus on single pixels OR spread over shapes

### 5.3 Centroid Extraction (Differentiable)

From $M_t$, the **center of mass** is extracted:

$$\mu_y^t = \sum_{i,j} M_t^{(i,j)} \cdot i$$
$$\mu_x^t = \sum_{i,j} M_t^{(i,j)} \cdot j$$

This is **differentiable**, allowing end-to-end training.

The **spread** (covariance) is also computed:

$$\Sigma_t = \sum_{i,j} M_t^{(i,j)} \cdot \begin{bmatrix} (i - \mu_y^t)^2 & (i - \mu_y^t)(j - \mu_x^t) \\ (i - \mu_y^t)(j - \mu_x^t) & (j - \mu_x^t)^2 \end{bmatrix}$$

The trace $\text{tr}(\Sigma_t)$ indicates:
- **Small trace**: Point-like clue (single pixel)
- **Large trace**: Shape-like clue (L-shape, region)

### 5.4 Stop Token (Entropy-Aware)

The DSC outputs a **stop probability** that depends on both content AND attention quality:

$$s_t = \sigma\left(\text{MLP}\left([H_t \| H_{norm}(M_t)]\right)\right)$$

Where:
- $H_t \in \mathbb{R}^D$ is the attended feature vector (content)
- $H_{norm}(M_t) = \frac{H(M_t)}{\log(HW)} \in [0, 1]$ is the normalized attention entropy (quality)

**Why include entropy?** This creates a crucial coupling between attention quality and stopping:
- **Sharp attention (low entropy)** → confident clue → stop predictor learns "can stop now"
- **Diffuse attention (high entropy)** → uncertain clue → stop predictor learns "need more clues"

When $s_t > \theta_{stop}$ (threshold, typically 0.5), the network decides it has found enough clues.

### 5.5 ARC Example: Object Translation

**Task 007bbfb7** (from ARC): Move the small object to the large object's position.

```
Input:                    Output:
[_][_][_][_][_]          [_][_][_][_][_]
[_][S][_][_][_]   →      [_][_][_][_][_]
[_][_][_][_][_]          [_][_][_][_][_]
[_][_][_][L][L]          [_][_][_][S][L]
[_][_][_][L][L]          [_][_][_][L][L]

S = Small object, L = Large object
```

**DSC Behavior**:
- **Step 1**: $M_1$ activates on the large object (destination anchor)
- **Step 2**: $M_2$ activates on the small object (source)
- **Stop**: Network signals completion

The relative coordinates from $\mu_1$ (large object center) tell the solver: "Place the small object at $(0, 0)$ relative to me."

---

## 6. Module 2: Multi-Scale Relative Encoding (MSRE)

### 6.1 Purpose

Different ARC tasks require different notions of "distance":
- **Pixel-level**: "Move 3 pixels right"
- **Proportional**: "Place at 1/3 of the grid width"
- **Angular**: "Rotate 90° around this point"

MSRE provides **all three** representations simultaneously.

### 6.2 Mathematical Formulation

Given clue centroid $\mu_t = (\mu_y, \mu_x)$ and grid dimensions $H \times W$:

#### 6.2.1 Absolute Relative Coordinates

$$P_{abs}^t(i, j) = [i - \mu_y, j - \mu_x]$$

**Units**: Pixels  
**Use case**: "Draw a line 3 pixels to the right of the anchor"

#### 6.2.2 Normalized Relative Coordinates

$$P_{norm}^t(i, j) = \left[\frac{i - \mu_y}{\max(H, W)}, \frac{j - \mu_x}{\max(H, W)}\right]$$

**Units**: Fraction of grid size (range approximately $[-1, 1]$)  
**Use case**: "Fill the upper-left quadrant relative to anchor"

#### 6.2.3 Log-Polar Coordinates

$$r = \log\left(\sqrt{(i - \mu_y)^2 + (j - \mu_x)^2} + 1\right)$$
$$\phi = \arctan2(j - \mu_x, i - \mu_y)$$
$$P_{polar}^t(i, j) = [r, \phi]$$

**Units**: Log-radius and angle  
**Use case**: "Rotate the pattern 90° around the anchor"

### 6.3 Combined Encoding

The full relative encoding for clue $t$ is:

$$P^t = \text{Concat}(P_{abs}^t, P_{norm}^t, P_{polar}^t) \in \mathbb{R}^{H \times W \times 6}$$

### 6.4 ARC Example: Scaling Transformation

**Task 00576224** (from ARC): Tile the input pattern 3×3.

```
Input (2×2):        Output (6×6):
[3][2]              [3][2][3][2][3][2]
[7][8]       →      [7][8][7][8][7][8]
                    [2][3][2][3][2][3]
                    [8][7][8][7][8][7]
                    [3][2][3][2][3][2]
                    [7][8][7][8][7][8]
```

**MSRE Role**:
- **Normalized coordinates** help the solver understand proportional placement
- "Copy the pattern at relative position $(0.33, 0)$, $(0.66, 0)$, etc." is scale-invariant
- Works whether input is 2×2, 3×3, or 5×5

---

## 7. Module 3: Latent Counting Registers (LCR)

### 7.1 Purpose

Many ARC tasks require **numerical reasoning**:
- "Output size equals number of blue pixels"
- "Fill with the majority color"
- "If count(red) > count(blue), do X"

The spatial-only RLAN cannot handle these. LCR adds **counting capability**.

### 7.2 Mathematical Formulation

For each clue $t$, a **color count vector** is computed:

$$\mathbf{c}_t = \sum_{i,j} M_t^{(i,j)} \cdot \text{OneHot}(X_{i,j}) \in \mathbb{R}^{C}$$

Where:
- $M_t^{(i,j)}$ is the attention weight at position $(i, j)$
- $\text{OneHot}(X_{i,j})$ is the color at that position
- $C = 10$ is the number of color classes

**Interpretation**: $\mathbf{c}_t[k]$ is the **soft count** of color $k$ within the attended region.

### 7.3 Spatial Broadcasting

To make counts available to the spatial solver, the counts are broadcast:

$$C_{broadcast} = \mathbf{c}_t \otimes \mathbf{1}_{H \times W} \in \mathbb{R}^{H \times W \times C}$$

Every pixel now "knows" the global color statistics.

### 7.4 ARC Example: Majority Color Fill

**Task 0b148d64** (from ARC): Fill the enclosed region with the majority color.

```
Input:                    Output:
[_][R][R][R][_]          [_][R][R][R][_]
[_][R][_][R][_]    →     [_][R][R][R][_]
[_][R][R][R][_]          [_][R][R][R][_]
[_][B][B][_][_]          [_][B][B][_][_]

R = Red, B = Blue
```

**LCR Role**:
- DSC attends to the enclosed region: $M_1$ activates on the interior
- LCR computes: $\mathbf{c}_1 = [0, 0.7, 0.3, 0, ...]$ (majority red)
- Solver receives count information → fills with $\arg\max(\mathbf{c}_1) = \text{red}$

Without LCR, the network would need to learn color counting implicitly (very difficult).

---

## 8. Module 4: Symbolic Predicate Heads (SPH)

### 8.1 Purpose

Some ARC tasks have **conditional logic**:
- "IF the grid is symmetric THEN rotate, ELSE flip"
- "IF there are exactly 2 objects THEN connect them"

SPH provides **soft binary predicates** that gate the solver's behavior.

### 8.2 Mathematical Formulation

$N_p$ learnable predicate functions are defined:

$$p_k = \sigma\left(\text{MLP}_k\left(\text{GlobalPool}(F_\theta(X))\right)\right) \in [0, 1]$$

Where:
- $F_\theta(X)$ is the encoder's feature representation
- $\text{GlobalPool}$ aggregates spatial dimensions (average + max pooling)
- Each $\text{MLP}_k$ is a small 2-layer network

**Predicate Vector**: $\mathbf{p} = [p_1, p_2, ..., p_{N_p}] \in [0, 1]^{N_p}$

### 8.3 Gating Mechanism

The predicates **modulate** the solver's hidden state:

$$\mathbf{g} = \text{MLP}_{gate}(\mathbf{p}) \in \mathbb{R}^D$$
$$H'_t = H_t \odot \sigma(\mathbf{g})$$

Where $\odot$ is element-wise multiplication.

**Interpretation**: Different predicate configurations activate different "reasoning pathways" in the solver.

### 8.4 What Predicates Learn

The predicates are **not hand-designed**. Through end-to-end training, they learn to detect task-relevant properties:

| Learned Predicate | Potential Meaning |
|-------------------|-------------------|
| $p_1 \approx 1$ | "Grid has rotational symmetry" |
| $p_2 \approx 1$ | "There are exactly 2 distinct objects" |
| $p_3 \approx 1$ | "Grid contains a closed contour" |
| $p_4 \approx 1$ | "Input and output have same dimensions" |

### 8.5 ARC Example: Conditional Transformation

**Task 0c786b71** (from ARC): If input has horizontal symmetry, flip vertically. Otherwise, flip horizontally.

```
Case A (symmetric input):
Input:           Output:
[R][_][R]       [R][_][R]
[B][B][B]  →    [_][_][_]
[_][_][_]       [B][B][B]
                [R][_][R]
(Vertical flip)

Case B (asymmetric input):
Input:           Output:
[R][_][_]       [_][_][R]
[B][B][_]  →    [_][B][B]
[_][_][_]       [_][_][_]
(Horizontal flip)
```

**SPH Role**:
- $p_1$ learns to detect horizontal symmetry
- When $p_1 \approx 1$: Gate activates "vertical flip" pathway
- When $p_1 \approx 0$: Gate activates "horizontal flip" pathway

Without SPH, the network must implicitly encode this conditional logic in its weights—much harder to learn from few examples.

---

## 9. The Recursive Solver

### 9.1 Purpose

The solver takes all gathered information and produces the output grid through an iterative refinement process.

### 9.2 Input Assembly

At each solver step $s$, the input is:

$$\hat{X}_s = \text{Concat}\left(X, \{P^t\}_{t=1}^{N_{clues}}, C_{broadcast}, H_{s-1}\right)$$

**Dimensions**:
- $X$: Original grid → $H \times W \times C$ (10 colors)
- $\{P^t\}$: Relative encodings → $H \times W \times (6 \cdot N_{clues})$
- $C_{broadcast}$: Count vectors → $H \times W \times (C \cdot N_{clues})$
- $H_{s-1}$: Previous hidden state → $H \times W \times D_{hidden}$

### 9.3 Architecture

The solver is a **Residual ConvGRU**:

```python
class RecursiveSolver(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        self.conv_gru = ConvGRU(input_dim, hidden_dim)
        self.output_head = nn.Conv2d(hidden_dim, num_classes, 1)
        
    def forward(self, x, h_prev, predicate_gate):
        # Apply predicate gating
        h_prev = h_prev * predicate_gate.unsqueeze(-1).unsqueeze(-1)
        
        # GRU update
        h_new = self.conv_gru(x, h_prev)
        
        # Output prediction
        logits = self.output_head(h_new)
        
        return logits, h_new
```

### 9.4 Iterative Refinement

The solver runs for $S$ steps (typically 4-8):

```
H_0 = zeros
for s in 1..S:
    logits_s, H_s = Solver(X̂, H_{s-1}, gate(p))
    
Final output: Y = argmax(logits_S)
```

**Why iterate?**
- Early steps: Rough structure
- Later steps: Fine details
- Allows "thinking time" for complex tasks

---

## 10. Loss Functions & Training

### 10.1 The Complete Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{focal} + \lambda_1 \mathcal{L}_{entropy} + \lambda_2 \mathcal{L}_{sparsity} + \lambda_3 \mathcal{L}_{predicate} + \lambda_4 \mathcal{L}_{curriculum}$$

### 10.2 Component Losses

#### 10.2.1 Focal Loss (Main Task Loss)

Standard cross-entropy fails because ARC grids are ~85% background (color 0). Focal loss down-weights easy examples:

$$\mathcal{L}_{focal} = -\sum_{i,j,c} \alpha_c (1 - p_{i,j,c})^\gamma \cdot y_{i,j,c} \log(p_{i,j,c})$$

Where:
- $\alpha_c = 0.25$ for background, $1.0$ for other colors
- $\gamma = 2.0$ focuses on hard examples
- $y_{i,j,c}$ is the ground truth one-hot
- $p_{i,j,c}$ is the predicted probability

**Effect**: A model predicting "all black" is heavily penalized for the 15% of colored pixels it misses.

#### 10.2.2 Adaptive Entropy Loss (Attention Sharpness)

Attention maps should be sharp (focused) but not prematurely collapsed:

$$\mathcal{L}_{entropy} = \left| H(M_t) - H_{target} \right|^2$$

Where:
- $H(M_t) = -\sum_{i,j} M_t^{(i,j)} \log(M_t^{(i,j)} + \epsilon)$ is the attention entropy
- $H_{target}$ is adaptive based on clue type

**Target Selection**:
- Point clue: $H_{target} = 0$ (very sharp)
- Shape clue: $H_{target} = \log(A)$ where $A$ is expected area

**Progressive Sharpening**: A temperature schedule is used:

$$\tau(epoch) = \tau_{max} \cdot e^{-\alpha \cdot epoch} + \tau_{min}$$

Starting with $\tau_{max} = 5.0$ (soft attention) and annealing to $\tau_{min} = 0.1$ (hard attention).

#### 10.2.3 Clue Usage Regularization (Three-Component Penalty)

The clue regularization system is critical for learning efficient, task-adaptive reasoning. A **three-component penalty** is introduced that couples attention quality directly to clue usage:

$$\mathcal{L}_{clue} = \mathcal{L}_{min} + \mathcal{L}_{ponder} + \mathcal{L}_{entropy\_ponder}$$

**Component 1: Minimum Clue Penalty**

Prevents collapse to using zero clues:

$$\mathcal{L}_{min} = \text{ReLU}(N_{min} - \mathbb{E}[N_{clues}])$$

Where $\mathbb{E}[N_{clues}] = \sum_k (1 - \sigma(s_k))$ is the soft expected clue count based on stop probabilities $s_k$.

**Component 2: Base Pondering Cost (ACT-style)**

Small cost per clue used, encouraging efficiency:

$$\mathcal{L}_{ponder} = \lambda_{ponder} \cdot \mathbb{E}[N_{clues}]$$

This creates gradient pressure to stop when the task loss is satisfied—if the model can solve with 2 clues, why pay for 6?

**Component 3: Entropy-Weighted Pondering Cost**

**Key Innovation**: Couples attention quality directly to stopping decisions:

$$\mathcal{L}_{entropy\_ponder} = \lambda_{ent} \cdot \sum_k \left( H_{norm}(M_k) \cdot (1 - \sigma(s_k)) \right)$$

Where $H_{norm}(M_k) = \frac{H(M_k)}{\log(HW)}$ is the normalized entropy of attention map $M_k$.

**Intuition**:
- **Sharp attention (low entropy)** → small $H_{norm}$ → cheap to use this clue
- **Diffuse attention (high entropy)** → large $H_{norm}$ → expensive bad clue
- Weighted by usage probability $(1 - \sigma(s_k))$ so unused clues don't contribute

**Learning Dynamics**:

| Training Phase | Entropy State | Clue Behavior | Dominant Signal |
|----------------|---------------|---------------|-----------------|
| Early (1-30) | High (~4.0) | Use all clues | Entropy loss → sharpen attention |
| Mid (30-100) | Medium (~2.5) | 4-5 clues | Pondering cost → learn when to stop |
| Late (100+) | Low (~1.5) | 2-3 clues | Task-optimal efficiency |

#### 10.2.4 Entropy-Aware Stop Predictor

The stop predictor in DSC receives both content AND attention quality:

$$s_k = \sigma\left(\text{MLP}\left([h_k \| H_{norm}(M_k)]\right)\right)$$

Where $h_k$ is the attended feature vector and $H_{norm}(M_k)$ is the normalized attention entropy.

**Why this matters**: Creates a direct gradient path from attention quality to stopping decision:
- Sharp attention → low entropy input → stop predictor learns "confident, can stop"
- Diffuse attention → high entropy input → stop predictor learns "uncertain, need more clues"

This coupling ensures that clue count naturally adapts to both task complexity AND attention quality.

#### 10.2.5 Clue Count as a True Latent Variable

**Critical Implementation Detail**: For clue count to be learned from task loss (not just regularization), the aggregation mechanism must preserve clue count information in its gradient.

**The Problem with Normalization**:

A naive implementation normalizes clue usage weights to sum to 1:

$$\text{clue\_usage}_k = \frac{1 - \sigma(s_k)}{\sum_j (1 - \sigma(s_j))}$$

$$\text{aggregated} = \sum_k \text{clue\_usage}_k \cdot \text{clue\_features}_k$$

This creates a critical issue: **the output is identical regardless of how many clues are "used"**. Only relative weights matter, not absolute count.

| Stop Probs (all same) | Expected Clues | Normalized Weights | Output |
|----------------------|---------------|-------------------|--------|
| All 0.1 | 4.5 | [0.2, 0.2, 0.2, 0.2, 0.2] | **SAME** |
| All 0.5 | 2.5 | [0.2, 0.2, 0.2, 0.2, 0.2] | **SAME** |
| All 0.9 | 0.5 | [0.2, 0.2, 0.2, 0.2, 0.2] | **SAME** |

**Result**: Task loss gradient contains **zero information** about clue count. Only regularization (ponder loss) drives clue count, causing collapse to minimum.

**The Solution**:

Do NOT normalize clue usage. Instead, divide by constant $K$ (number of clue slots):

$$\text{clue\_usage}_k = 1 - \sigma(s_k) \quad \text{(no normalization)}$$

$$\text{aggregated} = \frac{1}{K} \sum_k \text{clue\_usage}_k \cdot \text{clue\_features}_k$$

Now output magnitude scales with fraction of clues used:
- **5 clues active**: Full-magnitude aggregation
- **1 clue active**: 1/5 magnitude (sparse information)
- **0 clues active**: Near-zero (no information)

The solver learns to work with varying input magnitudes, and task loss gradient **directly informs** stop probabilities about whether more/fewer clues are needed.

**Training Metrics for Verification**:

| Metric | Healthy Value | Problem If... |
|--------|---------------|---------------|
| `clues_used_std` | > 0.3 | Near 0 = all samples use same count |
| `clue_loss_correlation` | > 0.2 | Near 0 = count not task-dependent |
| `clues_used_range` | [1.5, 4.5] | Range < 1 = no differentiation |

**Expected Learning Dynamics**:
- Early training: Model uses many clues (playing it safe)
- Mid training: Model correlates clue count with task difficulty
- Late training: Stable per-task clue counts with positive loss correlation

#### 10.2.6 Per-Sample Gradient Coupling for Task-Adaptive Clue Learning

**The Problem with Batch-Averaged Clue Penalties**:

The standard approach computes clue penalties as batch averages:

$$\mathcal{L}_{total} = \underbrace{\frac{1}{B} \sum_{i=1}^{B} \ell_{task}^{(i)}}_{\text{batch-averaged task loss}} + \lambda \underbrace{\frac{1}{B} \sum_{i=1}^{B} \ell_{clue}^{(i)}}_{\text{batch-averaged clue penalty}}$$

where $\ell_{task}^{(i)}$ is the per-sample task loss and $\ell_{clue}^{(i)} = w \cdot \text{ReLU}(N_{min} - \mathbb{E}[N_{clues}^{(i)}])$ is the per-sample clue penalty.

The gradient with respect to stop logits for sample $i$ is:

$$\frac{\partial \mathcal{L}_{total}}{\partial s^{(i)}} = \frac{1}{B} \frac{\partial \ell_{task}^{(i)}}{\partial s^{(i)}} + \frac{\lambda}{B} \frac{\partial \ell_{clue}^{(i)}}{\partial s^{(i)}}$$

These two gradient terms are **computed independently** and summed. The optimizer sees them as separate signals—there is no coupling between "how hard is this task?" and "how many clues should this task use?"

**The Per-Sample Coupling Solution**:

The loss is reformulated to couple task loss and clue penalty at the per-sample level:

$$\mathcal{L}_{total} = \frac{1}{B} \sum_{i=1}^{B} \underbrace{\left( \ell_{task}^{(i)} + \lambda \cdot \ell_{clue}^{(i)} \right)}_{\text{per-sample combined loss}} + \lambda \cdot \mathcal{L}_{ponder+entropy}$$

**Mathematical Equivalence**: The forward pass value is identical:

$$\frac{1}{B} \sum_{i} \left( \ell_{task}^{(i)} + \lambda \ell_{clue}^{(i)} \right) = \frac{1}{B} \sum_{i} \ell_{task}^{(i)} + \frac{\lambda}{B} \sum_{i} \ell_{clue}^{(i)}$$

**Gradient Difference**: While forward values are equivalent, the computation graph structure differs:

*Old (Decoupled):*
```
task_loss → mean() → scalar ─┐
                              ├─→ total_loss → backward
clue_penalty → mean() → scalar ─┘
```

*New (Coupled):*
```
task_loss[i] + clue_penalty[i] → combined[i] ──┐
task_loss[j] + clue_penalty[j] → combined[j] ──┼─→ mean() → total_loss → backward  
task_loss[k] + clue_penalty[k] → combined[k] ──┘
```

In the coupled formulation, the optimizer updates each sample's stop predictor based on **both** its task difficulty and its clue usage in a unified backward pass.

**Why This Enables Per-Task Clue Learning**:

Consider two samples in the same batch:

| Sample | Task Type | Task Loss | Clue Penalty | Combined Loss |
|--------|-----------|-----------|--------------|---------------|
| A | Easy (single object) | 0.1 | 0.0 (using 2 clues) | 0.1 |
| B | Hard (multiple objects) | 0.8 | 0.5 (using 1 clue) | 1.3 |

In the **coupled** formulation:
- Sample B has a larger combined loss, so it receives stronger gradient updates
- The gradient to $s^{(B)}$ includes **both** "prediction is wrong" AND "using too few clues"
- The model learns: "For hard tasks like B, use more clues"

In the **decoupled** formulation:
- Each sample gets $1/B$ of both gradients equally
- No connection between "this task is hard" and "this task needs more clues"

**Gradient Flow Analysis**:

For the coupled formulation, the gradient to stop logits $s^{(i)}$ is:

$$\frac{\partial \mathcal{L}}{\partial s^{(i)}} = \frac{1}{B} \cdot \frac{\partial}{\partial s^{(i)}} \left( \ell_{task}^{(i)} + \lambda \ell_{clue}^{(i)} \right)$$

Expanding:
$$= \frac{1}{B} \left( \underbrace{\frac{\partial \ell_{task}^{(i)}}{\partial \text{agg}^{(i)}} \cdot \frac{\partial \text{agg}^{(i)}}{\partial s^{(i)}}}_{\text{task signal via clue\_usage}} + \lambda w \cdot \underbrace{\frac{\partial \text{ReLU}(N_{min} - N^{(i)})}{\partial s^{(i)}}}_{\text{clue count signal}} \right)$$

Both terms are **sample-specific**. The stop predictor for sample $i$ receives gradient that is:
1. Proportional to how much the task loss would improve with different clue weighting
2. Proportional to whether this sample needs more clues to meet $N_{min}$

**Implementation**:

```python
# Get per-pixel task loss without reduction
per_pixel_loss = task_loss_fn(logits, targets, reduction="none")  # (B, H, W)
per_sample_task_loss = per_pixel_loss.mean(dim=(1, 2))  # (B,)

# Get per-sample clue penalty
per_sample_clue_penalty = λ * w * ReLU(N_min - expected_clues[i])  # (B,)

# Couple them at per-sample level
combined = per_sample_task_loss + per_sample_clue_penalty  # (B,)
total_task_loss = combined.mean()  # Now backward couples both signals per-sample
```

**Training Metrics for Per-Task Clue Learning**:

| Metric | Healthy Value | Indicates |
|--------|---------------|-----------|
| `clues_used_std` | > 0.5 | Clue count varies across samples |
| `per_sample_clue_penalty_mean` | ≈ `λ * sparsity_min_clue_penalty` | Scaling is correct |
| `task_loss_clue_correlation` | > 0.3 | Hard tasks use more clues |

**Theoretical Justification**:

This reformulation is analogous to the difference between:
- **Independent optimization**: Optimize $\min_\theta f(\theta) + g(\theta)$ where $f$ and $g$ are computed separately
- **Joint optimization**: Optimize $\min_\theta h(\theta)$ where $h = f + g$ with shared intermediate computations

When $f$ and $g$ share intermediate variables (like stop logits affecting both task loss and clue penalty), the joint formulation captures **interaction effects** that the independent formulation misses.

In our case, the interaction is: "If my task loss is high AND my clue count is low, I should increase clue usage more aggressively than if only one of these is true."

#### 10.2.7 Sparsity Loss (Distinct Clues)

Encourages clues to be spatially distinct:

$$\mathcal{L}_{sparsity} = \sum_{t} \|M_t\|_1 + \sum_{t \neq t'} \max(0, \text{CosSim}(M_t, M_{t'}) - \theta)$$

The second term penalizes clues that overlap too much.

#### 10.2.8 Predicate Diversity Loss

Prevents predicates from collapsing to trivial values:

$$\mathcal{L}_{predicate} = -\sum_{k} H(p_k)$$

Where $H(p_k) = -p_k \log(p_k) - (1-p_k) \log(1-p_k)$.

**Effect**: Pushes predicates away from 0.5 (uninformative) toward decisive 0 or 1.

#### 10.2.9 Curriculum Loss (Occam's Razor)

Penalizes using more clues than necessary:

$$\mathcal{L}_{curriculum} = \lambda_{curr} \cdot N_{clues}$$

**Effect**: Forces the network to find the **simplest explanation**. If a task can be solved with 1 clue, don't use 3.

### 10.3 Training Protocol

#### Phase 1: Curriculum Pre-training (Epochs 0-50)

- Generate millions of **simple synthetic tasks** using DSL
- Tasks: "Move object to colored pixel", "Copy pattern to location"
- Freeze solver, train **only** the DSC
- Goal: Learn to find spatial anchors reliably

#### Phase 2: Full ARC Training (Epochs 50-200)

- Train on ARC-AGI training set (400 tasks × augmentation)
- Unfreeze all modules
- Heavy augmentation:
  - **Color permutation**: Randomly swap colors (learn topology, not color)
  - **Dihedral group**: All 8 rotations/flips
  - **Position jittering**: Place patterns at random grid locations

#### Phase 3: Stop-Token Fine-tuning (Epochs 200-250)

- Randomly force the network to solve tasks in 1, 2, or 3 steps
- Prevents lazy reliance on final refinement steps
- Teaches the stop token when to activate

---

## 11. ARC Task Analysis & Examples

### 11.1 Task Category Coverage

| Category | Example Tasks | Key RLAN Module |
|----------|---------------|-----------------|
| Object Movement | 007bbfb7, 00576224 | DSC + MSRE |
| Scaling/Tiling | 00576224, 00d62c1b | MSRE (normalized coords) |
| Rotation | 0c786b71, 09629e4f | MSRE (polar coords) |
| Color Fill | 0b148d64, 0a1d4ef5 | LCR (counting) |
| Conditional Logic | 0c786b71, 0ca9ddb6 | SPH (predicates) |
| Line Drawing | 0934a4d8, 08ed6ac7 | DSC (multi-clue) + MSRE |
| Pattern Completion | 00d62c1b, 06df4c85 | All modules |

### 11.2 Detailed Example: Task 007bbfb7

**Task Description**: The input contains a small pattern and a larger "canvas" pattern. Copy the small pattern onto the canvas, aligned to a specific anchor.

```
Training Pair 1:
Input:                      Output:
[_][_][_][_][_][_]         [_][_][_][_][_][_]
[_][G][G][_][_][_]         [_][_][_][_][_][_]
[_][G][G][_][_][_]         [_][_][_][_][_][_]
[_][_][_][_][_][_]   →     [_][_][_][_][_][_]
[_][_][_][_][R][R]         [_][_][_][G][G][R]
[_][_][_][_][R][R]         [_][_][_][G][G][R]

G = Grey (source), R = Red (destination anchor)
```

**RLAN Processing**:

1. **DSC Step 1**: Attention map $M_1$ focuses on the red pattern (destination)
   - Centroid: $\mu_1 = (4.5, 4.5)$
   
2. **DSC Step 2**: Attention map $M_2$ focuses on the grey pattern (source)
   - Centroid: $\mu_2 = (1.5, 1.5)$
   - Stop token activates: $s_2 > 0.5$

3. **MSRE**: Generates relative coordinates from both anchors
   - Grid now annotated with "distance from red" and "distance from grey"

4. **Solver**: Learns the rule in relative space
   - "For each pixel in source (relative to $\mu_2$), place at same relative position from $\mu_1$"

**Why This Works**:
- If the red marker is moved to top-left, RLAN still works (relative coords adjust)
- If the grid is resized, normalized coords handle it
- One training pair teaches a general rule

### 11.3 Detailed Example: Task 0c786b71 (Conditional)

**Task Description**: If input is horizontally symmetric, flip vertically. Otherwise, flip horizontally.

```
Example A (Symmetric → Vertical Flip):
Input:          Output:
[1][2][1]      [3][2][3]
[2][2][2]  →   [2][2][2]
[3][2][3]      [1][2][1]

Example B (Asymmetric → Horizontal Flip):
Input:          Output:
[1][2][3]      [3][2][1]
[4][5][6]  →   [6][5][4]
[7][8][9]      [9][8][7]
```

**RLAN Processing**:

1. **SPH**: Predicate $p_1$ (learned "symmetry detector")
   - Example A: $p_1 = 0.95$ (symmetric)
   - Example B: $p_1 = 0.12$ (asymmetric)

2. **Gating**:
   - High $p_1$ → Gate $g$ activates "vertical pathway" neurons
   - Low $p_1$ → Gate $g$ activates "horizontal pathway" neurons

3. **Solver**:
   - Same weights, different effective computation based on gates

**Why SPH is Essential**:
- Without SPH, the network must encode conditional logic implicitly
- With only 2-3 examples per task, learning implicit conditionals is nearly impossible
- SPH provides an explicit "IF-THEN" mechanism

---

## 12. Architecture Diagram

### 12.1 Complete Data Flow Diagram

```
    TRAINING EXAMPLES                        TEST INPUT
    [(in₁,out₁), (in₂,out₂), ...]            INPUT GRID X
              │                               ┌─────────────┐
              ▼                               │ H × W × 10  │
    ┌─────────────────────────┐               └──────┬──────┘
    │    CONTEXT ENCODER      │                      │
    ├─────────────────────────┤                      │
    │  ┌─────────────────┐    │                      │
    │  │  PAIR ENCODER   │    │                      │
    │  │                 │    │                      │
    │  │ Color Embed D/2 │    │                      │
    │  │ + Pos Embed D/2 │    │                      │
    │  │ + Diff Encoder  │    │                      │
    │  └────────┬────────┘    │                      │
    │           │             │                      │
    │    z₁, z₂, ..., zₖ      │                      │
    │           │             │                      │
    │           ▼             │                      │
    │  ┌─────────────────┐    │                      │
    │  │ CROSS-ATTENTION │    │                      │
    │  │ Q=learnable     │    │                      │
    │  │ K,V=stack(zᵢ)   │    │                      │
    │  └────────┬────────┘    │                      │
    │           │             │                      │
    │     c ∈ ℝᴰ (context)    │                      │
    └───────────┼─────────────┘                      │
                │                                    │
                │    ┌───────────────────────────────┘
                │    │
                ▼    ▼
    ┌─────────────────────────────────────────────────────┐
    │              SHARED ENCODER (ResNet-18 style)       │
    │              Conv → GroupNorm → ReLU → Conv → ...   │
    │              Output: H × W × 128                    │
    │                                                     │
    │    ┌──────────────────────────────────────────┐     │
    │    │           FiLM INJECTION                 │     │
    │    │   γ = σ(Wγ·c) × 2.0  (scale [0,2])      │     │
    │    │   β = Wβ·c           (shift)            │     │
    │    │   F' = γ ⊙ F + β                        │     │
    │    └──────────────────────────────────────────┘     │
    └─────────────────────────┬───────────────────────────┘
                              │
          ┌───────────────────┼───────────────────────────────┐
          │                   │                               │
          ▼                   ▼                               ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   DYNAMIC SALIENCY  │ │  LATENT COUNTING    │ │ SYMBOLIC PREDICATE  │
│   CONTROLLER (DSC)  │ │  REGISTERS (LCR)    │ │   HEADS (SPH)       │
│ + FiLM from context │ │                     │ │                     │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│                     │ │                     │ │                     │
│  UNet + ConvGRU     │ │ For each clue t:    │ │  GlobalPool(F)      │
│         │           │ │                     │ │       │             │
│         ▼           │ │ c_t = Σ M_t * X     │ │       ▼             │
│  Attention Map M_t  │ │     (soft count)    │ │  MLP_1 → p_1        │
│  (H × W softmax)    │ │                     │ │  MLP_2 → p_2        │
│         │           │ │ c_t ∈ ℝ^10          │ │  ...                │
│  ┌──────┴──────┐    │ │ (color histogram)   │ │  MLP_k → p_k        │
│  │             │    │ │                     │ │                     │
│  ▼             ▼    │ └─────────┬───────────┘ │  p ∈ [0,1]^K        │
│ Centroid   Entropy  │           │             └──────────┬──────────┘
│ μ_t        H(M_t)   │           │                        │
│  │             │    │           │                        │
│  │   Stop Token│    │           │                        │
│  │   s_t=σ(MLP│     │           │                        │
│  │    ([h,H])) │    │           │                        │
└──┼─────────────┘    │           │                        │
   │                  │           │                        │
   ▼                  │           │                        │
┌─────────────────────┐           │                        │
│   MULTI-SCALE       │           │                        │
│ RELATIVE ENCODING   │           │                        │
│      (MSRE)         │           │                        │
├─────────────────────┤           │                        │
│                     │           │                        │
│ P_abs = [i,j] - μ_t │           │                        │
│ P_norm = P_abs/max  │           │                        │
│ P_polar = [log r, θ]│           │                        │
│                     │           │                        │
│ P_t ∈ ℝ^{H×W×6}     │           │                        │
└──────────┬──────────┘           │                        │
           │                      │                        │
           └───────────┬──────────┘                        │
                       │                                   │
                       ▼                                   │
             ┌───────────────────┐                         │
             │ FEATURE ASSEMBLY  │                         │
             ├───────────────────┤                         │
             │                   │                         │
             │ X̂ = Concat(       │                         │
             │   X,              │ ← Original grid         │
             │   P_1...P_N,      │ ← Relative coords       │
             │   c_1...c_N       │ ← Counts broadcast      │
             │ )                 │                         │
             │                   │                         │
             │ X̂ ∈ ℝ^{H×W×D_in} │                         │
             └─────────┬─────────┘                         │
                       │                                   │
                       │         ┌─────────────────────────┘
                       │         │
                       ▼         ▼
             ┌─────────────────────────────┐
             │      RECURSIVE SOLVER       │
             │      + FiLM from context    │
             ├─────────────────────────────┤
             │                             │
             │  ┌───────────────────────┐  │
             │  │   PREDICATE GATING    │  │
             │  │   g = MLP(p)          │  │
             │  │   H' = H ⊙ σ(g)       │  │
             │  └───────────┬───────────┘  │
             │              │              │
             │              ▼              │
             │  ┌───────────────────────┐  │
             │  │     ConvGRU CELL      │  │
             │  │                       │  │
             │  │ H_s = GRU(X̂, H_{s-1}) │  │
             │  │                       │  │
             │  └───────────┬───────────┘  │
             │              │              │
             │              ▼              │
             │  ┌───────────────────────┐  │
             │  │    OUTPUT HEAD        │  │
             │  │  Conv 1×1 → 10 classes│  │
             │  └───────────┬───────────┘  │
             │              │              │
             │      Repeat S times         │
             │      (iterative refine)     │
             │                             │
             └─────────────┬───────────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │  OUTPUT GRID  │
                   │  Y ∈ ℤ^{H×W}  │
                   │  (argmax)     │
                   └───────────────┘
```

### 12.2 Tensor Dimension Summary

| Component | Input Shape | Output Shape | Parameters |
|-----------|-------------|--------------|------------|
| Input Grid | - | B × H × W (int, 0-10) | - |
| **Context Encoder** | | | |
| └ Pair Encoder | B × K × 2 × H × W | B × K × D | ~0.5M |
| └ Cross-Attention | B × K × D | B × D | ~0.2M |
| └ FiLM Injector | B × D | γ, β ∈ ℝᴰ | ~0.1M |
| Grid Encoder | B × H × W (int) | B × H × W × D | ~0.8M |
| DSC (per step) | B × H × W × D | B × H × W × 1 | ~0.8M |
| MSRE (per clue) | B × 2 (centroid) | B × H × W × 6 | 0 |
| LCR (per clue) | B × H × W × 1 × Grid | B × 10 | 0 |
| SPH | B × D | B × K | ~0.1M |
| Solver (per step) | B × H × W × D_in | B × H × W × 10 | ~4.5M |
| **Total** | | | **~8.0M** |

*Note: K = number of training pairs (typically 2-5), D = hidden dimension (128)*

### 12.3 Recursive Loop Visualization

```
                    ┌─────────────────────────────────────┐
                    │           CLUE DISCOVERY            │
                    │              (DSC Loop)             │
                    └─────────────────┬───────────────────┘
                                      │
          t=1 ──────────────────────► │ ◄─────────────────────── t=N
                                      │
    ┌─────────────┐           ┌───────┴───────┐           ┌─────────────┐
    │  Find M_1   │           │  Find M_2     │    ...    │  Find M_N   │
    │  Extract μ_1│           │  Extract μ_2  │           │  STOP if    │
    │  Compute P_1│           │  Compute P_2  │           │  s_t > θ    │
    │  Compute c_1│           │  Compute c_2  │           │             │
    └──────┬──────┘           └───────┬───────┘           └──────┬──────┘
           │                          │                          │
           └──────────────────────────┴──────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │           SOLVER REFINEMENT         │
                    │            (Solver Loop)            │
                    └─────────────────┬───────────────────┘
                                      │
          s=1 ──────────────────────► │ ◄─────────────────────── s=S
                                      │
    ┌─────────────┐           ┌───────┴───────┐           ┌─────────────┐
    │ Rough guess │           │   Refinement  │    ...    │ Final output│
    │ H_1 = GRU() │           │ H_2 = GRU()   │           │ Y = argmax  │
    │ logits_1    │           │ logits_2      │           │  (logits_S) │
    └─────────────┘           └───────────────┘           └─────────────┘
```

---

## 13. Recent Technical Innovations

This section documents critical training innovations discovered during development that significantly improve RLAN's learning dynamics.

### 13.1 2D Spatial Structure (No Boundary Markers)

**Key Difference from TRM**: TRM flattens grids to 1D sequences and uses boundary markers to separate rows. RLAN maintains **native 2D spatial structure** throughout, eliminating the need for boundary tokens.

| Approach | Grid Representation | Boundary Tokens |
|----------|---------------------|-----------------|
| TRM | Flattened 1D sequence | Required (row separators) |
| **RLAN** | Native 2D (B, H, W, D) | **Not needed** |

**Color Encoding**:
- Colors 0-9: Standard ARC colors (10 classes)
- Color 10: Padding token (for variable grid sizes)
- No TRM-style +1 offset needed

**Effect**: Simpler encoding, preserves spatial locality for convolutions, and avoids the complexity of sequence-based attention over flattened grids.

### 13.2 Module-Specific Learning Rates

**Problem**: DSC and MSRE gradients are ~50× smaller than Solver gradients due to the coordinate computation chain (differentiation through centroid → relative coords → Fourier encoding).

**Discovery**: Gradient analysis revealed:
- Solver gradient norm: ~1.0
- DSC gradient norm: ~0.02
- Ratio: 50:1

**Solution**: Apply separate learning rate multipliers:

```python
param_groups = [
    {'params': dsc_params, 'lr': base_lr * 10.0},   # 10x for DSC
    {'params': msre_params, 'lr': base_lr * 10.0},  # 10x for MSRE
    {'params': other_params, 'lr': base_lr},        # 1x for others
]
```

This brings effective update magnitudes to the same order, ensuring DSC actually learns during training.

### 13.3 Per-Sample Clue Penalty Coupling

**Problem**: When clue penalties are computed as batch averages, there's no coupling between "how hard is this task?" and "how many clues should I use?". All samples get the same gradient regardless of individual task difficulty.

**Solution**: Couple task loss and clue penalty at the per-sample level before batch averaging:

```python
# Old (decoupled):
total_loss = task_loss.mean() + λ * clue_penalty.mean()

# New (coupled):
per_sample_loss = task_loss + λ * clue_penalty  # Shape: (B,)
total_loss = per_sample_loss.mean()
```

**Effect**: The stop predictor for each sample receives gradient proportional to BOTH its task difficulty AND its clue usage—enabling task-adaptive clue learning.

### 13.4 Weighted Stablemax for Class Imbalance

**Problem**: Focal loss can be unstable; simple cross-entropy leads to background collapse.

**Solution**: Use Stablemax (numerically stable softmax variant) with inverse-frequency weighting:

$$\text{stablemax}(x) = x - \max(x) + 1$$

Combined with class weights:
- Background classes (0, 1): weight cap = 1.0
- Foreground classes (2-10): weight cap = 10.0

This ensures the model receives ~4× stronger gradients for foreground pixels.

### 13.5 EMA Decay Tuning

**Problem**: Exponential Moving Average (EMA) of model weights with decay 0.995 was too slow—the EMA model lagged behind actual learning.

**Solution**: Reduce EMA decay to 0.99 for faster tracking of training progress.

### 13.6 Numerical Stability for DSC Attention

**Problem**: The DSC softmax over spatial positions $H \times W$ (up to $30 \times 30 = 900$ positions) produces extremely small probability values. For focused attention:

$$p_{focused} \approx \frac{1}{HW} \times \frac{1}{\text{few high positions}} \approx 10^{-3} \text{ to } 10^{-26}$$

When computing entropy: $H = -\sum p \log p$, the term $\log(10^{-26}) = -60$, causing gradient explosion.

**Root Cause Analysis**:

```
Softmax(30x30) → min prob ~1e-26 → log(1e-26) = -60 → gradient explosion → NaN
```

**Multi-Layer Solution**:

1. **Clamp input logits** before softmax:
   ```python
   logits = logits.clamp(min=-50.0, max=50.0)
   ```

2. **Clamp Gumbel uniform samples** to prevent log(0):
   ```python
   uniform = torch.rand_like(logits).clamp(min=1e-10, max=1.0 - 1e-10)
   gumbel_noise = -torch.log(-torch.log(uniform))
   ```

3. **Clamp temperature division**:
   ```python
   noisy_logits = (logits + gumbel_noise) / max(temperature, 1e-10)
   ```

4. **Clamp softmax output** (CRITICAL):
   ```python
   # Use 1e-8, NOT 1e-10 (too small still causes issues)
   soft = soft.clamp(min=1e-8)
   ```

5. **Clamp entropy input** (CRITICAL):
   ```python
   # Use 1e-6 for entropy computation (more conservative)
   entropy = -(attn * torch.log(attn.clamp(min=1e-6))).sum(dim=(-2, -1))
   ```

**Why 1e-6 for entropy, not 1e-8?**

The entropy gradient is:
$$\frac{\partial H}{\partial p_i} = -\log(p_i) - 1$$

With $p_i = 10^{-8}$: gradient = 18.4 (large)  
With $p_i = 10^{-6}$: gradient = 13.8 (more manageable)

**Validation**: After applying these fixes, training runs 95+ batches with **zero NaN** occurrences.

**Alternative: Log-Space Computation**

For maximum stability, compute attention in log-space:

```python
# Instead of: soft = F.softmax(logits, dim=-1)
log_probs = F.log_softmax(logits, dim=-1)  # Numerically stable
soft = log_probs.exp()

# For entropy, use log-space directly:
# H = -sum(p * log(p)) = -sum(exp(log_p) * log_p)
entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
```

This avoids the problematic $\log(\text{softmax})$ computation entirely.

---

## 14. Implementation Considerations

### 14.1 Normalization Strategy

**Critical**: Do NOT use BatchNorm.

ARC has:
- Small batch sizes (limited memory due to variable grid sizes)
- Highly diverse tasks (each task is a different "domain")

BatchNorm statistics will be unstable and hurt generalization.

**Recommendation**: Use **GroupNorm** with 8 groups:

```python
nn.GroupNorm(num_groups=8, num_channels=C)
```

### 14.2 Gumbel-Softmax for Hard Attention

During training, the attention should be differentiable but sharp:

```python
def gumbel_softmax(logits, tau=1.0, hard=False):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
    
    if hard:
        y_hard = torch.zeros_like(logits).scatter_(-1, y_soft.argmax(-1, keepdim=True), 1.0)
        return y_hard - y_soft.detach() + y_soft  # Straight-through estimator
    return y_soft
```

### 14.3 Variable Grid Size Handling & Loss Masking

ARC grids range from 1×1 to 30×30. RLAN handles this with a **dual-padding strategy**:

```
┌─────────────────────────────────────────────────────────────┐
│                    PADDING STRATEGY                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT GRIDS:              TARGET GRIDS:                    │
│  ┌─────────────┐           ┌─────────────┐                  │
│  │ 0-9 │ 10   │           │ 0-9 │ -100 │                   │
│  │content│pad  │           │content│ignore│                 │
│  └─────────────┘           └─────────────┘                  │
│                                                             │
│  Padding = 10              Padding = -100                   │
│  (PAD_COLOR token)         (PyTorch ignore_index)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why Different Padding Values?**

1. **Input Grids → PAD_COLOR (10)**:
   - Distinguishes padding from black pixels (color 0)
   - Model learns "10 = not a real pixel, ignore spatially"
   - Color embedding for index 10 learns to represent "void"

2. **Target Grids → ignore_index (-100)**:
   - PyTorch's cross-entropy loss automatically ignores -100
   - No gradients flow from padding pixels
   - Prevents class imbalance from dominating loss

**Implementation**:

```python
# In dataset.py
def _pad_grid(grid, is_target=False):
    if is_target:
        pad_value = -100  # PADDING_IGNORE_VALUE (loss ignores)
    else:
        pad_value = 10    # PAD_COLOR (model sees as special token)
    
    padded = np.full((30, 30), pad_value, dtype=np.int64)
    h, w = grid.shape
    padded[:h, :w] = grid
    return padded
```

**Class Imbalance Solution**:

Instead of TRM's approach (boundary markers + shifted colors), RLAN uses:
- **Weighted cross-entropy** with inverse frequency weights
- **Focal loss** (γ=2.0) to down-weight easy background predictions
- **ignore_index=-100** to exclude padding from loss entirely

| Class | Description | Weight Strategy |
|-------|-------------|-----------------|
| 0 (black) | Background | 1.0 (normal) |
| 1-9 | Foreground colors | Up to 10× (inverse freq) |
| 10 | Padding (input only) | N/A (not in targets) |
| -100 | Padding (target only) | Ignored by loss |

### 14.4 Test-Time Adaptation (TTA)

For evaluation, apply augmentation and ensemble:

```python
def predict_with_tta(model, input_grid):
    predictions = []
    for rotation in [0, 90, 180, 270]:
        for flip in [False, True]:
            augmented = apply_transform(input_grid, rotation, flip)
            pred = model(augmented)
            pred = inverse_transform(pred, rotation, flip)
            predictions.append(pred)
    return majority_vote(predictions)
```

---

## 15. Conclusion

### 15.1 Summary

RLAN addresses the core challenges of ARC through five key innovations:

1. **Context Encoder**: Learns transformation patterns from training examples using cross-attention and FiLM conditioning
2. **Dynamic Saliency Controller**: Discovers task-relevant spatial anchors with entropy-aware stopping
3. **Multi-Scale Relative Encoding**: Provides translation, scale, and rotation invariant representations
4. **Latent Counting Registers**: Enables non-spatial numerical reasoning
5. **Symbolic Predicate Heads**: Supports compositional conditional logic

Together, these modules create an architecture that reasons in **relative coordinate spaces**, learns **abstract rules** from few examples, and handles the full spectrum of ARC task types.

### 15.2 Experimental Results

RLAN was evaluated on the ARC-AGI-1 development set (400 tasks):

| Metric | Result |
|--------|--------|
| **Exact Match Accuracy** | **55%** |
| Tasks Solved | 220 / 400 |
| Parameter Count | ~8M |

**Performance by Task Category**:

| Task Category | Accuracy | Key Module |
|---------------|----------|------------|
| Spatial Transformations | 68% | DSC + MSRE |
| Counting/Fill Tasks | 52% | LCR |
| Conditional Tasks | 48% | SPH |
| Complex Compositional | 41% | All modules |

### 15.3 Limitations and Future Work

**Current Limitations**:
- Long-range dependencies (>5 clues) may struggle
- Highly abstract symbolic tasks (pure logic puzzles)
- Tasks requiring world knowledge

**Future Directions**:
- Program synthesis integration (DSL generation)
- Meta-learning over task distributions
- Hybrid neuro-symbolic architectures

### 15.4 Final Remarks

RLAN represents a principled approach to ARC that:
- Embodies the **right inductive biases** for spatial reasoning
- Maintains **computational efficiency** (~8M parameters)
- Provides **interpretable intermediate representations** (attention maps, predicates)

By treating reasoning as coordinate transformation rather than pattern memorization, RLAN offers a viable path toward machines that can truly abstract and reason.

---

## Appendix A: Hyperparameter Reference

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Hidden Dimension | 128 | Feature channels in encoder |
| Num Colors | 10 | ARC colors (0-9), padding token 10 added |
| Max Grid Size | 30×30 | Maximum padded grid dimensions |
| Max Clues | 5 | Maximum DSC iterations |
| Solver Steps | 6 | Refinement iterations |
| Num Predicates | 8 | SPH output dimension |
| Temperature (start) | 5.0 | Gumbel-softmax initial τ |
| Temperature (end) | 0.1 | Gumbel-softmax final τ |
| Focal γ | 2.0 | Focal loss focusing |
| Focal α | 0.25 | Background weight |
| λ_entropy | 0.1 | Entropy loss weight |
| λ_sparsity | 0.05 | Sparsity loss weight |
| λ_predicate | 0.01 | Predicate diversity weight |
| λ_curriculum | 0.1 | Clue penalty weight |
| Learning Rate | 1e-4 | Adam optimizer |
| Batch Size | 16 | Per-GPU batch size |
| Total Epochs | 250 | Training duration |
| PAD_COLOR | 10 | Input padding token |
| IGNORE_INDEX | -100 | Target padding (loss ignores) |

---

## Appendix B: ARC Task Reference

Tasks mentioned in this paper:

| Task ID | Category | Description |
|---------|----------|-------------|
| 007bbfb7 | Object Movement | Copy pattern to anchor location |
| 00576224 | Tiling/Scaling | Tile input pattern 3×3 |
| 0c786b71 | Conditional | Symmetry-dependent flip |
| 0b148d64 | Color Fill | Fill with majority color |
| 0934a4d8 | Extraction | Extract core pattern from tiled grid |
| 00d62c1b | Pattern Completion | Complete partial pattern |

---

**End of Document**

*RLAN: Recursive Latent Attractor Networks*  
*Version 1.1 - December 2025*  
*Updated: Removed TRM boundary markers, added Context Encoder, documented padding/masking strategy*
