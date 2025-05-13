# AdjointDiffusion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python: 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/) 


![logo](images/logo.png)

**AdjointDiffusion** is a new method for structural optimization using diffusion models. 
It is a **physics-guided and fabrication-aware structural optimization** using diffusion models augmented with adjoint gradient. By combining powerful generative models with adjoint sensitivity analysis, this approach can more efficiently discover complex, high-performance designs than the traditional methods.

The codes are provided following the paper named [Physics-guided and fabrication-aware
structural optimization using diffusion models](https://arxiv.org/abs/2504.17077)

---

## Table of Contents

1. [TL;DR](#TL;DR)  
2. [Intuitive Explanation of Diffusion Models](#intuitive-explanation-of-diffusion-models)
3. [Installation](#installation)  
4. [Quick Start](#quick-start)  
5. [Customize Your Simulation](#customize-your-simulation)  
6. [Experiment Logging with Weights & Biases](#experiment-logging-with-weights--biases)
7. [Results](#results)
8. [Code Organization](#code-organization)
9. [Citation](#citation)

## TL;DR

✨ **Integrating adjoint sensitivity analysis with diffusion models can generate high-performance and interesting structures!**

Key features:
- **Adjoint Sensitivity Integration**: Seamlessly incorporates adjoint gradients into the diffusion process.
- **Fabrication Constraints**: Accounts for manufacturability, ensuring real-world feasibility.
- **Extensibility**: Users can use their own datasets or simulations.
- **Experiment Tracking & Visualization**: Integrates with [Weights & Biases](https://wandb.ai/home).

---

## Intuitive Explanation of Diffusion Models

Imagine an ink drop falling into water — it slowly spreads and dissolves. Diffusion models mimic this process in reverse: they start from noise and slowly form meaningful structures. By guiding this "reverse diffusion" with gradients from an adjoint method, we ensure the final designs are optimized and fabrication-ready.

---



## Installation

This setup ensures compatibility between Meep and PyTorch. If you find any alternatives, feel free to contribute improvements via pull requests!

### 1. Clone the repository
```bash
git clone https://github.com/dongjin-seo2020/AdjointDiffusion.git
cd AdjointDiffusion
```

### 2. Set up a Python environment (recommended)


#### Using conda
To create and activate the recommended environment with necessary dependencies:

```bash
conda create -n adjoint_diffusion -c conda-forge pymeep pymeep-extras python=3.9 
conda activate adjoint_diffusion
```


### 3. Installation of other libraries

Install `torch` following the command (recommended):
```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
```

Note: It has been observed that newer NVIDIA GPUs (e.g., RTX 5090) may not be compatible with this specific version of PyTorch. If you encounter issues, please refer to the official PyTorch installation guide to find a version compatible with your hardware:
https://pytorch.org/get-started/locally/

Install the required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```


### 4. Error Handling


If you encounter permission-related issues when trying to run the training script, make sure it is executable by running:

```bash
chmod +x 01-train.sh
```

The, you can execute it with:

```bash
./01-train.sh
```

If you encounter errors while installing `mpi4py`, try the following steps:
```bash
apt --fix-broken install
apt install mpich
pip install mpi4py
```
Make sure you have root access when using `apt`.



---

## Quick Start

1. **Generate a dataset:**

```bash
python dataset_generation.py
```

- The data will be saved at `datasets/<n>/sigma<k>/struct/`, where `n` is the structure dimension (e.g., `n=64` generates 64×64 binary structures) and `k` is the variance of the Gaussian filter (a larger `k` increases the minimum feature size).

2. **Update the training and sampling scripts** to specify the appropriate output directories.

- For example, for `train.sh`, you should specify the variables as:
```sh
DATA_DIR=/path/to/datasets
LOG_DIR=path/to/experiments
GPU_ID=0
```

- Or, you can set environment variables (Linux/macOS):
```bash
export DATA_DIR=/path/to/datasets
export LOG_DIR=/path/to/experiments
export GPU_ID=0
```

- For detailed usage examples, including training and sampling with actual settings, see:
  - [training_examples.ipynb](./training_examples.ipynb)
  - [sampling_example.ipynb](./sampling_example.ipynb)

3. **Train a diffusion model:**

```bash
./01-train.sh
```

- Alternatively: run `02-train.ipynb`
- Note: Set `--class_cond` to `False` if your dataset contains only a single structural condition (i.e., no class conditioning needed).
If you have multiple structural conditions (e.g., different fabrication constraints), set it to `True` to enable class-conditional training.



4. **Sample and optimize structures:**

```bash
./01-sample.sh
```

- Alternatively: run `02-sample.ipynb`


5. **View outputs**
- Every output (performance, structure) is logged in [wandb](#experiment-logging-with-weights--biases).
- Logs and generated structures are saved in `./logs/<run_name>`


6. **Baseline Algorithms**

We provide baseline algorithms in the `./baseline_algorithms` directory. These include **nlopt** methods like MMA for comparison.


---

## Customize Your Simulation

If you'd like to **integrate a custom physical simulation** into the reverse diffusion process, follow these steps:

1. **Implement Your simulation class** in `guided_diffusion/simulation.py`.

   Create a class that defines how to compute the figure of merit (FoM) and its corresponding adjoint gradient. For example:
   ```python
   class YourSimClass:
       def __init__(self, ...):
           ...
       def compute_fom(self, structure):
           ...
       def compute_adjoint(self, structure):
           ...
    ```


2. **Update the import** in `guided_diffusion/gaussian_diffusion.py`:

   Replace the existing simulation import with your custom class:
   ```python
   from guided_diffusion.simulation import YourSimClass
   ```

3. **Plug your simulation into the sampling loop** in `guided_diffusion/gaussian_diffusion.py`
In `guided_diffusion/gaussian_diffusion.py`, locate where `simulation_()` is called (typically inside the `p_sample()` function), and replace it with your custom simulation logic. Make sure your class is initialized properly and passed via `my_kwargs`.
For example, for `my_kwargs`:
```python
my_kwargs = {
    "sim_guided": True,
    "simulation_": YourSimClass(...),
    "eta": 0.05,
    "inter_rate": 25,
    "stoptime": 0.1,
    "guidance_type": "dps",  # or "dds"
    "exp_name": "experiment1",
    ...
}
```

Now, your custom simulation will be used during the reverse diffusion process.

---


## Experiment Logging with Weights & Biases

We use [wandb](https://wandb.ai/home) for logging and visualization.

1. Sign up at [wandb.ai](https://wandb.ai)
2. Log in:
```bash
wandb login
```
3. Run any training/sampling script and it will automatically log data to wandb.

---



## Results

We visualize the performance of AdjointDiffusion across different tasks and configurations.


### Optimization Convergence and Comparisons - Problem Setup 1 (Waveguide)

![Performance Plot 1](images/Result1.png)




### Comparison of Generated Structures - Problem Setup 1 (Waveguide)

![Bar Plots](images/Result1-2.png)


### Optimization Convergence and Comparisons 2 - Problem Setup 2 (Multi-wavelength Splitter)

![Performance Plot 2](images/Result2.png)

### Comparison of Generated Structures - Problem Setup 2 (Multi-wavelength Splitter)

![Bar Plots](images/Result2-2.png)


### Result: Waveguide

![Waveguide](images/Result-waveguide.png)


### Result: Multi-wavelength Splitter

![Color Router](images/Result-colorrouter.png)

---

## Code Organization

```
AdjointDiffusion/
├── dataset_generation.py       # Dataset generation script
├── image_train.py              # Main training script
├── image_sample.py             # Main sampling script
├── requirements.txt            # Python dependencies
├── guided_diffusion/           # Backend of diffusion models
└── baseline_algorithms/        # Baseline algorithms (nlopt, Gradient Ascent)
```

---

## Citation

If you use this code, please cite the following paper:

```bibtex
@article{YourCitation,
  title   = {Physics-guided and fabrication-aware structural optimization using diffusion models},
  author  = {Dongjin Seo†, Soobin Um†, Sangbin Lee, Jong Chul Ye*, Haejun Chung*},
  journal = {arXiv},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.17077}
}
```

---


## Acknowledgements

Parts of this repository are adapted from [OpenAI's guided-diffusion](https://github.com/openai/guided-diffusion), which is licensed under the MIT License.

We thank the OpenAI team for their contribution. Significant modifications have been made to enable adjoint sensitivity integration and fabrication-aware optimization.

---
**Happy Diffusing & Optimizing!**
