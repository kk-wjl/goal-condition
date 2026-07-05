# goal-condition

Goal-conditioned generative modeling experiments for motion data, currently centered on retargeted LAFAN1 trajectories.

The repo currently contains three main experiment families:

- `sliding window` flow matching
- `full chunk` flow matching
- `hierarchy` with a high-level waypoint CVAE and a low-level local-goal flow model


## Setup

Python requirement:

- `python >= 3.13`

Setup with `uv`:

```bash
uv sync
```
Run commands with `uv run ...`, for example:

```bash
uv run python main.py
```


## Data

The LAFAN1 loader expects retargeted motion CSVs under `data/`.

Supported layouts:

- `data/LAFAN1_Retargeting_Dataset/<robot>/*.csv`
- `data/<robot>/*.csv`



## Experiments

### 1. Sliding Window Flow Matching

Script:

```bash
uv run python scripts/FM_lafan1_sliding_window.py --help
```

- cuts short windows from longer motion chunks
- conditions each window on a sparse future root goal
- trains a single low-level flow model

Outputs:

- `scripts/outputs/FM_lafan1_sliding_window/`

### 2. Full-Chunk Flow Matching

Script:

```bash
uv run python scripts/FM_lafan1_full_chunk.py --help
```

- generates one full chunk at a time
- pins the first `prefix_frames`
- conditions on the chunk's final root goal
- supports both UNet and DiT backbones

Outputs:

- `scripts/outputs/FM_lafan1_full_chunk/`

### 3. Hierarchical Goal-Conditioned Modeling

Script:

```bash
uv run python scripts/FM_lafan1_hierarchy.py --help
```

- `train_high`: train the high-level waypoint CVAE
- `train_low`: train the low-level local-goal flow model
- `rollout`: load both checkpoints and perform hierarchical rollout

High level:

- input: normalized prefix + final goal
- output: intermediate root waypoints

Low level:

- input: short prefix + one local goal
- output: one local trajectory segment

Rollout:

- sample intermediate waypoints from the high-level prior
- append the final goal
- generate trajectory segments sequentially with the low-level flow
- report rollout metrics such as final-goal error and velocity finite-difference MSE

Outputs:

- `scripts/outputs/FM_lafan1_hierarchy/high_level/`
- `scripts/outputs/FM_lafan1_hierarchy/low_level/`
- `scripts/outputs/FM_lafan1_hierarchy/rollout/`


## Commands

Train the hierarchy high level:

```bash
uv run python scripts/FM_lafan1_hierarchy.py --mode train_high
```

Train the hierarchy low level:

```bash
uv run python scripts/FM_lafan1_hierarchy.py --mode train_low
```

Run hierarchy rollout with saved checkpoints:

```bash
uv run python scripts/FM_lafan1_hierarchy.py --mode rollout

```
Test waypoints generator with 2D visualization:

```bash
PYTHONPATH=. uv run python tests/test_waypointcvae.py
```

Test rollout generator:

```bash
PYTHONPATH=. uv run python tests/test_hierarchy_rollout.py
```

Run static checks:

```bash
uv run pyright
uv run pytest -q
```

## References

- [Goal-Conditioned Imitation Learning using Score-based Diffusion Policies](https://arxiv.org/abs/2304.02532)
- [Learning Latent Plans from Play](https://arxiv.org/abs/1903.01973)
- [Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation](https://arxiv.org/abs/2403.03890)
- [gen-modeling](https://github.com/btx0424/gen-modeling)
