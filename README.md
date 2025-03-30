# AdjointDiffusion

\textit{AdjointDiffusion} is a new method for structural optimization using diffusion models. 
It is a **physics-guided and fabrication-aware structural optimization** leveraging diffusion models augmented with adjoint gradient. By combining powerful generative models with adjoint sensitivity analysis, this approach can more efficiently discover complex, high-performance designs than the traditional methods.

The codes are provided following the paper named [Physics-guided and fabrication-aware
structural optimization using diffusion models](https://arxiv.org)

---

## Table of Contents

1. [TL;DR](#TL;DR)  
2. [Intuitive Explanation of Diffusion Models](#intuitive-explanation-of-diffusion-models)
3. [Installation](#installation)  
4. [Quick Start](#quick-start)  
5. [Usage](#usage)  
   - [Dataset Generation](#dataset-generation)
   - [Training](#training)
   - [Sampling](#sampling)
   - [Baseline Algorithms](#baseline-algorithms)
6. [Experiment Logging with Weights & Biases](#experiment-logging-with-weights--biases)

7. [Code Organization](#code-organization)
8. [Citation](#citation)

## TL;DR

| Integrating adjoint sensitivity analysis with diffusion models can generate high-performance and interesting structures! |
| - **Adjoint Sensitivity Integration**: Seamlessly incorporates adjoint gradients into the diffusion process. |
| - **Fabrication Constraints**: Accounts for manufacturability, ensuring real-world feasibility. |
| - **Extensibility**: Includes baseline algorithms (e.g., genetic algorithm) with sample scripts for data generation and training. |
| - **Experiment Tracking & Visualization**: Integrates with [Weights & Biases](https://wandb.ai/home) and TensorBoard. |


---

## Intuitive Explanation of Diffusion Models

Imagine an ink drop falling into water — it slowly spreads and dissolves. Diffusion models mimic this process in reverse: they start from noise and slowly form meaningful structures. By guiding this "reverse diffusion" with gradients from an adjoint method, we ensure the final designs are optimized and fabrication-ready.

---



## Installation

### 1. Clone the repository
```bash
git clone https://github.com/dongjin-seo2020/AdjointDiffusion.git
cd AdjointDiffusion
```

### 2. Set up a Python environment (recommended)

#### Using pip
```bash
pip install -r requirements.txt
```

#### Using conda
```bash
conda create -n adjoint_diffusion python=3.9
conda activate adjoint_diffusion
conda install --file requirements.txt
```


---

## Quick Start

1. Generate a dataset:
```bash
python dataset_generation.py
```

2. Train a diffusion model:
```bash
./01-train.sh
```

alternative way: run 02-train.ipynb


3. Sample and optimize structures:
```bash
./01-sample.sh
```

alternative way: run 02-sample.ipynb

4. View outputs
- Every output (performance, structure) is logged in [wandb](#experiment-logging-with-weights--biases).
- Checkpoints (and logs) are saved in `./experiments/<run_name>`



### 5. Baseline Algorithms
### Baseline Algorithms
We provide baseline algorithms in the `./baseline_algorithms` directory. These include \textit{nlopt} methods like MMA for comparison.


## Experiment Logging with Weights & Biases

We use [wandb](https://wandb.ai/home) for logging and visualization.

1. Sign up at [wandb.ai](https://wandb.ai)
2. Log in:
```bash
wandb login
```
3. Run any training/sampling script and it will automatically log data to wandb.

---


## Citation

If you use this code, please cite the following paper:

```bibtex
@article{YourCitation,
  title   = {Physics-guided and fabrication-aware structural optimization using diffusion models},
  author  = {Dongjin Seo†, Soobin Um†, Sangbin Lee, Jong Chul Ye*, Haejun Chung*},
  journal = {arXiv},
  year    = {2025},
  url     = {https://arxiv.org/}
}
```

---

**Happy Diffusing & Optimizing!**
