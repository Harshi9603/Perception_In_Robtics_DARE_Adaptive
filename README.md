<h1 align="center"> DARE: Diffusion Policy to Autonomous Robotics Exploration </h1>

<div align="center">

[![ICRA 2025](https://img.shields.io/badge/ICRA%202025-Paper-blue?style=flat&logo=ieee)](https://ieeexplore.ieee.org/abstract/document/11128196)
[![arXiv](https://img.shields.io/badge/arXiv-2512.02535-red?style=flat&logo=arxiv)](https://arxiv.org/abs/2410.16687)
[![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

<img src="assets/dare_main.gif" width="50%"/>

</div>

---

## Introduction
Autonomous robot exploration requires efficient path planning to map unknown environments. While conventional methods are often limited to optimizing based on current beliefs, **DARE (Diffusion Policy for Autonomous Robot Exploration)** leverages the power of generative AI to reason about unknown areas by drawing on learned experiences.

DARE is a novel approach that utilizes **diffusion models** trained on expert demonstrations to explicitly generate long-horizon exploration paths. By combining an attention-based encoder with a diffusion policy, DARE learns to recognize potential structures in unknown regions from partial beliefs, enabling it to plan paths that consider these unobserved areas.

**Key Features:**
*   **Generative Path Planning:** Uses diffusion models to explicitly generate efficient exploration paths.
*   **Expert Demonstrations:** Trained on ground truth optimal demonstrations to learn superior exploration patterns.
*   **Structure Reasoning:** Capable of reasoning about potential structures in unknown areas based on partial beliefs.
*   **Robust Performance:** Achieves state-of-the-art performance with strong generalizability in both simulation and real-world scenarios.

<div align="center">
<img src="assets/workflow.png" width="125%"/>
</div>

---

## Usage
### Requirements
Install the following dependencies in a conda environment as shown below:
```bash
git clone https://github.com/marmotlab/DARE.git && cd DARE
conda create -n env_dare python=3.12.9 -y
conda activate env_dare
pip install -e .
```


### Dataset Collection
Modify `dataset_parameter.py` to fit your dataset needs then run dataset collection script:
```bash
python dataset_driver.py
```

Dataset will be saved to directory `diffusion_exploration/dataset/name_of_test`.
It will include a `data.zarr` directory which contains the dataset and a `gifs` directory.

### Policy Training
Copy desired training config file from `diffusion_exploration/diffusion_policy/config`.
Modify desired task config file from `diffusion_exploration/diffusion_policy/config/task`.

**Note:** You probably should modify the `zarr_path` to change dataset location

You can run the training script which requires two arguements:
1. `--config-dir` which is the directory to find the config file
2. `--config-name` which is the name of the config file

```bash
python train.py --config-dir=. --config-name=train_exploration_transformer_node_discrete.yaml
```

This will create a directory `diffusion_exploration/data/date/time/name_of_run`

### Evaluation
Modify `test_parameter.py` to fit your test needs then run evaluation script:
```bash
python test_driver.py
```

Test results will be printed on terminal and saved as a CSV
`inference_gifs` directory will be created in `diffusion_exploration/data/date/time/name_of_run`.

---

## Class Project: K-sampling trajectory re-ranking:

Metrics from the infromative graph:

1. Utility sum along implied node path: total utility collected by the nodes visited by the predicted path.
2. Unique-node count: how many distinct graph nodes the predicted path visits.
3. Revisit penalty: how often the predicted path returns to nodes it already visited.
4. Invalid-edge count: how many predicted transitions are not valid graph edges.
5. Terminal-node utility: utility of the last implied node in the predicted path.
6. Terminal-node degree / connectivity: number of valid neighboring next nodes from the last implied node.
7. Guidepost overlap / alignment: how much the predicted path follows guidepost-marked nodes.

These are used to calculate a score based on k-sampled trajectories from the diffusion model. Each metric has a coefficient which is tuned using Optuna Tree-structured Parzen Estimator algorithm (TPESampler). As well, the number of samples k is tuned. 

The current implimentation runs this on the already trained model, trying to improve just inference. To run this:

```bash
python k_sampling_reranking_driver.py
```

where the number of test trials can be set in this function. The results from the optuna optimization are printed and saved into a file in the folder `/optune`.


## Class Project: Adaptive Step Scheduler for Diffusion Inference

### Overview

We introduce an **Adaptive Step Scheduler** that dynamically adjusts the number
of DDIM denoising steps at inference time based on the measured complexity of
the current environment. This is applied on top of the existing k-sampling
re-ranking pipeline and requires **no retraining**.

The key insight is that DARE's fixed-step diffusion loop wastes compute in
wide-open environments while potentially under-sampling in cluttered ones.
By allocating steps proportional to complexity, we reduce average inference
latency while maintaining trajectory quality.

---

### Method

A scalar **complexity score C ∈ [0, 1]** is computed from two signals
extracted directly from the robot's belief map at each timestep:

| Signal | Description |
|---|---|
| **Obstacle density** | Fraction of occupied cells in a local window around the robot |
| **Frontier distance** | Normalised Euclidean distance to the nearest exploration frontier |

These are combined linearly:

```
C = w_density × density + w_distance × dist_norm
```

The score is then mapped to a denoising step count:

```
T = round( T_min + (T_max - T_min) × C )
```

An optional exponential moving average (EMA) smooths the step count across
consecutive inference calls to prevent abrupt changes.

Default configuration (`AdaptiveSchedulerConfig`):

| Parameter | Value | Meaning |
|---|---|---|
| `t_min` | 20 | Steps in the easiest environments |
| `t_max` | 100 | Steps in the hardest environments |
| `w_density` | 0.6 | Weight for obstacle density |
| `w_distance` | 0.4 | Weight for frontier distance |
| `ema_alpha` | 0.3 | EMA smoothing coefficient |

---

### Integration

The scheduler modifies only the inference loop in `test_worker.py`. The
original DARE diffusion loop iterates over all timesteps:

```python
# Original DARE
for t in self.policy.noise_scheduler.timesteps:
    ...
```

The adaptive version slices the timestep sequence before each forward pass:

```python
# Adaptive – only num_steps timesteps are used
self.policy.noise_scheduler.timesteps = original_timesteps[:num_steps]
action_dict = self.policy.predict_action(obs_dict)
self.policy.noise_scheduler.timesteps = original_timesteps   # restore
```

No changes are made to the model weights, training pipeline, or observation
pre-processing.

---

### New Files

| File | Role |
|---|---|
| `diffusion_exploration/utils/adaptive_scheduler.py` | Core scheduler class + module-level singleton |
| `test_worker.py` | Modified inference loop with scheduler integration |
| `test_adaptive_scheduler.py` | Unit tests (14 tests, no GPU required) |
| `run_dare_adaptive.sh` | SLURM script for baseline vs. adaptive comparison |

---

### Metrics Logged

The scheduler appends the following fields to the existing per-episode CSV:

| Column | Description |
|---|---|
| `adaptive_steps_mean` | Mean denoising steps per inference call |
| `adaptive_steps_min` | Minimum steps used in the episode |
| `adaptive_steps_max` | Maximum steps used in the episode |
| `complexity_mean` | Mean complexity score |
| `steps_saved_pct` | Percentage of steps saved vs. always using T_max |

---
### Example Behaviour
| Environment type | Complexity C | Steps T |
|---|---|---|
| Wide open space | ~0.05 | ~23 |
| Mixed / moderate | ~0.45 | ~56 |
| Dense / cluttered | ~0.85 | ~88 |

---
### Running the Tests
```bash
python test_adaptive_scheduler.py
```
All 14 unit tests run without a GPU or the DARE environment installed.
---
### Running on SOL (SLURM)
```bash
sbatch run_dare_adaptive.sh
```
This runs both a baseline (fixed steps) and the adaptive version sequentially
on the same GPU node and saves logs for comparison.
