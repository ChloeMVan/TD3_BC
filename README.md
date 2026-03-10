# A Minimalist Approach to Offline Reinforcement Learning

TD3+BC is a simple approach to offline RL where only two changes are made to TD3: (1) a weighted behavior cloning loss is added to the policy update and (2) the states are normalized. Unlike competing methods there are no changes to architecture or underlying hyperparameters. The paper can be found [here](https://arxiv.org/abs/2106.06860).

### Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

**NumPy:** Use `numpy<2` (pinned in requirements.txt). NumPy 2.x can cause crashes with PyTorch and mujoco-py.

**Cython:** Use `cython<3` (pinned in requirements.txt). Cython 3.x has stricter type checking that breaks mujoco-py. If you get `CompileError` in `cymj.pyx`, run `pip install "cython<3"`.

**MuJoCo (required for D4RL):** D4RL uses mujoco-py, which expects MuJoCo 210 at `~/.mujoco/mujoco210`. Download the **macOS** build from [roboti.us](https://www.roboti.us/index.html) or [GitHub releases](https://github.com/roboti-us/mujoco/releases) — **not** the Linux build. The Linux `.so` files are ELF format and will fail on macOS with "unknown file type"; macOS needs `.dylib` (Mach-O). Extract and place the `mujoco210` folder in `~/.mujoco/`. See the [mujoco-py README](https://github.com/openai/mujoco-py#install-mujoco) for details.

**GCC for mujoco-py (macOS):** mujoco-py compiles C extensions and needs GCC. `gcc@9` does not work on Apple Silicon or macOS newer than Monterey. Use a newer GCC instead:

```bash
brew install gcc@13   # or: brew install gcc
export CC=$(brew --prefix gcc@13)/bin/gcc-13   # or: export CC=$(brew --prefix gcc)/bin/gcc-14
```

**GLFW for mujoco-py (macOS):** mujoco-py links against GLFW. Install it and ensure the linker can find it:

```bash
brew install glfw
export LIBRARY_PATH="$(brew --prefix glfw)/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$(brew --prefix glfw)/lib:$LD_LIBRARY_PATH"
```

Then run your script in the same shell (or add the `export` lines to your `~/.zshrc`).

### Usage
Paper results were collected with [MuJoCo 1.50](http://www.mujoco.org/) (and [mujoco-py 1.50.1.1](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.17.0](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Networks are trained using [PyTorch 1.4.0](https://github.com/pytorch/pytorch) and Python 3.6.

The paper results can be reproduced by running:
```
./run_experiments.sh
```

### Offline-to-Online (TD3-BC → TD3 transfer)

This repo includes a two-stage pipeline: train TD3-BC offline on D4RL, then transfer the weights to vanilla TD3 and fine-tune online.

**1. Offline:** Train TD3-BC on a D4RL dataset and save checkpoint + state normalization:

```bash
python train_offline.py --env halfcheetah-medium-v2 --seed 0 --max_timesteps 1000000 --save_model
```

Checkpoints are written to `./models/TD3_BC_<env>_<seed>/` (policy weights, `normalization.npz`, `metadata.json`).

**2. Online:** Load the checkpoint into TD3 (vanilla, from the `TD3/` folder) and train in the environment:

```bash
python train_online.py --checkpoint_dir ./models/TD3_BC_halfcheetah-medium-v2_0 --seed 0 --max_timesteps 1000000 --save_model
```

Use `--env` to override the environment; otherwise it is read from the checkpoint’s `metadata.json`. State normalization from offline training is applied so the policy sees the same input distribution. Results are saved to `./results/`.

### Bibtex
```
@inproceedings{fujimoto2021minimalist,
	title={A Minimalist Approach to Offline Reinforcement Learning},
	author={Scott Fujimoto and Shixiang Shane Gu},
	booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
	year={2021},
}
```

---
*This is not an official Google product. 
