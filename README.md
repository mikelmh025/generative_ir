# GenIR: Generative Visual Feedback for Mental Image Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-2506.06220-b31b1b.svg)](https://arxiv.org/abs/2506.06220)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://visual-generative-ir.github.io)

> **Note**: This work is currently under review. Code and data will be released soon.

## Overview

GenIR introduces a novel approach to **Mental Image Retrieval (MIR)** - a realistic search scenario where users iteratively refine their queries based on mental images to retrieve intended images from a database. Unlike traditional one-shot text-to-image retrieval, our method addresses the multi-round, interactive nature of real-world human search behavior.

### Key Contributions

- **New Task Definition**: We introduce Mental Image Retrieval (MIR), bridging the gap between benchmark performance and real-world search applications
- **Generative Visual Feedback**: Our GenIR paradigm uses diffusion-based image generation to provide clear, interpretable visual feedback at each interaction round
- **Automated Dataset Generation**: We develop a fully automated pipeline to create high-quality multi-round MIR datasets
- **Superior Performance**: GenIR significantly outperforms existing interactive retrieval methods in MIR scenarios

## Method

### The Problem
Traditional vision-language models excel at text-to-image retrieval benchmarks but struggle with real-world search scenarios where:
- Users search based on **mental images** (ranging from vague recollections to vivid mental representations)
- Search is an **iterative, multi-round process** rather than one-shot
- Users need **clear, actionable feedback** to refine their queries effectively

### Our Solution
GenIR leverages **diffusion-based image generation** to:
1. **Reify AI understanding** at each interaction round through synthetic visual representations
2. **Provide interpretable feedback** that users can easily understand and act upon
3. **Enable intuitive query refinement** through visual rather than abstract verbal feedback

## Architecture

```
[Mental Image] → [Query] → [GenIR System] → [Visual Feedback + Retrieved Images]
                    ↑                              ↓
                [Refined Query] ← [User Feedback] ← [Next Round]
```

## Dataset

Our automated pipeline generates high-quality multi-round MIR datasets that capture:
- Diverse mental image scenarios
- Natural query refinement patterns
- Realistic user interaction flows

## Results

GenIR demonstrates significant improvements over existing interactive retrieval methods in Mental Image Retrieval scenarios. Detailed experimental results and comparisons will be available upon publication.

## Installation

> **Coming Soon**: Installation instructions will be provided when the code is released.

```bash
# Installation commands will be available here
```

## Usage

> **Coming Soon**: Usage examples and API documentation will be provided when the code is released.

```python
# Usage examples will be available here
```

## Contact

For questions or discussions, please contact:
- Diji Yang: dyang39@ucsc.edu
- Minghao Liu: mliu40@ucsc.edu


## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{yang2025genir,
  title={GenIR: Generative Visual Feedback for Mental Image Retrieval},
  author={Yang, Diji and Liu, Minghao and Lo, Chung-Hsiang and Zhang, Yi and Davis, James},
  journal={arXiv preprint arXiv:2506.06220},
  year={2025}
}
```
