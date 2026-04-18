# Magic Tower RL

DQN agent that learns to clear the deterministic magic-tower game described in [`instructions.md`](instructions.md).

## Environment

All commands assume the existing conda environment **`te`**.

```bash
conda activate te
pip install -r requirements.txt
```

## Usage

```bash
conda activate te

# 1. Train (early-stops once eval win-rate stays at 100% for 10 consecutive evaluations)
python train.py

# 2. Greedy rollout, prints the winning action sequence and writes viz/trajectory.json
python evaluate.py

# 3. Plot reward / loss / win-rate curves -> logs/training_curves.png
python plot_training.py

# 4. Open the interactive web replay
python -m http.server 8000
# then visit http://localhost:8000/viz/
```

## Files

- `magic_tower_env.py` - Gymnasium environment for the fixed CONFIG.
- `dqn_agent.py` - Q-network, replay buffer, target net, epsilon-greedy.
- `train.py` - Training loop, periodic greedy evaluation, early stop, checkpoint.
- `evaluate.py` - Loads best checkpoint, runs a greedy episode, exports trajectory.
- `plot_training.py` - Renders training curves from `logs/metrics.csv`.
- `viz/` - Static HTML + JS replay viewer driven by `viz/trajectory.json`.

## Reward design

- Kill a monster: `+1`
- Win (kill all `N` monsters with HP > 0): `+50`
- Death: `-50`
- Buy defense / heal (when affordable): `0`
- Invalid action (insufficient gold): `-0.1`
