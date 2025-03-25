# AdjointDiffusion

\textit{AdjointDiffusion} is a new method for structural optimization using diffusion models. It uses adjoint gradient to guide the diffusion process, which is more efficient than the traditional method.

The codes are provided following the paper named [Physics-guided and fabrication-aware
structural optimization using diffusion models](https://arxiv.org)


## TL;DR
|---|
| Integrating adjoint sensitivity analysis with diffusion models can generate interesting structures! |
|---|

## HOWTO


### 0. Environment Setup


### 1. Installation



### 2. Training



### 3. Sampling






## Okay... What is exactly happening here?
Though the code may look complex, what happens here is not so complicated. 
Let's say we want to optimize a structure which satisfies the fabrication constraints.
What should we take into account? We need two things: optimization and fabrication constraints.
First, we need to define the Figure of Merit (FoM), or optimization function.
Second, we 


## I still don't get it... Give me an intuitive explanation!
Sure. If you want an intuitive explanation of diffusion models, think about an ink drop on water.
The ink drop will spread out on the water surface, and the color will be diluted.








## example
~~~
python main.py --wavelength=900 --angle=60 --eps_greedy_period=1000000
~~~
The configuration of default conditions is written in `./config/config.json` file.


## installation
If you install it without any version control of environments, type 
~~~
pip install -r requirements.txt
~~~

or for Anaconda,
~~~
conda install -r requirements.txt
~~~

If you do not own MATLAB but use RETICOLO as a simulation tool, you will additionally need a MATLAB engine. Please refer to the site:
https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

## optimized structures
The optimized structures are saved as .np files in `./structures` folder.


## results & visualization
The logs of each experiment will be saved in `./experiments` folder.
You can use Tensorboard to visualize the results.
~~~
tensorboard --log_dir=experiments
~~~


## genetic algorithm
To utilize the genetic algorithm code in `./solvers` folder, you need to install MATLAB add-on named 'global optimization toolbox'. Also, please erase the `img = img/2.0 + 0.5;` part in the code `Eval_Eff_1D.m` to use genetic algorithm, since the output of genetic algorithm is a binary sequence composed of 0 and 1.

## citation
If you refer to the code, please cite [our paper](arxiv.).

