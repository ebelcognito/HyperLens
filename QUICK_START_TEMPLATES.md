# Quick Start: Code Templates

## 1. Minimal Test Script (Start Here)

```python
# test_ray_optics_basic.py
import json
import subprocess

def run_simulation(scene_dict):
    """Send scene to ray-optics, get result"""
    scene_json = json.dumps(scene_dict)
    
    result = subprocess.run(
        ['node', 'ray-optics/integrations/runner.js'],
        input=scene_json.encode(),
        capture_output=True,
        timeout=5
    )
    
    return json.loads(result.stdout)

# Test 1: Ray hits detector directly
scene_direct = {
    "version": 5,
    "objs": [
        {
            "type": "SingleRay",
            "p1": {"x": 0, "y": 150},
            "p2": {"x": 50, "y": 150}  # Ray going right
        },
        {
            "type": "Detector",
            "p1": {"x": 300, "y": 140},
            "p2": {"x": 300, "y": 160}
        }
    ]
}

result = run_simulation(scene_direct)
print(f"Direct hit power: {result['detectors'][0]['power']}")

# Test 2: Glass blocks ray
scene_blocked = {
    "version": 5,
    "objs": [
        {
            "type": "SingleRay",
            "p1": {"x": 0, "y": 150},
            "p2": {"x": 50, "y": 150}
        },
        {
            "type": "Glass",
            "path": [
                {"x": 100, "y": 100, "arc": False},
                {"x": 120, "y": 100, "arc": False},
                {"x": 120, "y": 200, "arc": False},
                {"x": 100, "y": 200, "arc": False}
            ],
            "refIndex": 1.5,
            "partialReflect": True
        },
        {
            "type": "Detector",
            "p1": {"x": 300, "y": 140},
            "p2": {"x": 300, "y": 160}
        }
    ]
}

result = run_simulation(scene_blocked)
print(f"With glass power: {result['detectors'][0]['power']}")
```

**Expected output:**
```
Direct hit power: ~0.95-1.0
With glass power: < 0.5 (glass bends ray away)
```

---

## 2. Simple 1D Environment (Stage 1)

```python
# envs/simple_steering_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import subprocess

class SimpleSteeringEnv(gym.Env):
    """
    Simplest possible environment:
    - 1D action: move glass X position only
    - Fixed 50x50 glass square
    - Goal: maximize detector power
    """
    
    def __init__(self):
        super().__init__()
        
        # Action: delta X position
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: [quad_x, detector_power, progress]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([400, 1, 1]), 
            dtype=np.float32
        )
        
        # Fixed elements
        self.source_pos = {"x": 50, "y": 150}
        self.detector_pos = {"x": 350, "y": 150}
        
        # Controllable
        self.quad_x = 200  # Center X position
        self.quad_size = 50
        self.refIndex = 1.5
        
        # Episode tracking
        self.max_steps = 20
        self.step_count = 0
        self.current_power = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Random initial position
        self.quad_x = np.random.uniform(100, 300)
        self.step_count = 0
        self.current_power = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Apply action (scale by 10 pixels per step)
        self.quad_x += action[0] * 10
        self.quad_x = np.clip(self.quad_x, 50, 350)
        
        # Run simulation
        result = self._simulate()
        self.current_power = result['detectors'][0]['power']
        
        # Reward is just detector power
        reward = self.current_power
        
        # Episode ends after max_steps
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, done, False, {}
    
    def _get_obs(self):
        return np.array([
            self.quad_x / 400,  # Normalize
            self.current_power,
            self.step_count / self.max_steps
        ], dtype=np.float32)
    
    def _simulate(self):
        """Build scene and run ray-optics simulation"""
        half = self.quad_size / 2
        
        scene = {
            "version": 5,
            "objs": [
                # Source
                {
                    "type": "SingleRay",
                    "p1": self.source_pos,
                    "p2": {"x": self.source_pos["x"] + 50, "y": self.source_pos["y"]}
                },
                # Glass quad (fixed size, variable X position)
                {
                    "type": "Glass",
                    "path": [
                        {"x": self.quad_x - half, "y": 125, "arc": False},
                        {"x": self.quad_x + half, "y": 125, "arc": False},
                        {"x": self.quad_x + half, "y": 175, "arc": False},
                        {"x": self.quad_x - half, "y": 175, "arc": False}
                    ],
                    "refIndex": self.refIndex,
                    "partialReflect": True
                },
                # Detector
                {
                    "type": "Detector",
                    "p1": {"x": self.detector_pos["x"], "y": self.detector_pos["y"] - 10},
                    "p2": {"x": self.detector_pos["x"], "y": self.detector_pos["y"] + 10}
                }
            ]
        }
        
        # Call Node.js
        result = subprocess.run(
            ['node', 'ray-optics/integrations/runner.js'],
            input=json.dumps(scene).encode(),
            capture_output=True,
            timeout=5
        )
        
        return json.loads(result.stdout)
```

---

## 3. Training Script (Stage 1)

```python
# main_stage1.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.simple_steering_env import SimpleSteeringEnv
import numpy as np

def test_random_baseline(env, n_episodes=10):
    """Measure random policy performance"""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    
    print(f"Random baseline: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    return np.mean(rewards)

# Create environment
env = SimpleSteeringEnv()

# Baseline
random_baseline = test_random_baseline(env)

# Setup callbacks
eval_callback = EvalCallback(
    env, 
    eval_freq=1000,
    best_model_save_path='./results/models/',
    log_path='./results/logs/'
)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path='./results/checkpoints/'
)

# Create PPO agent
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log='./results/tensorboard/'
)

# Train
print("Starting training...")
model.learn(
    total_timesteps=20_000,
    callback=[eval_callback, checkpoint_callback]
)

# Evaluate
print("\nEvaluating trained policy...")
obs, _ = env.reset()
total_reward = 0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Trained policy: {total_reward:.3f}")
print(f"Improvement over random: {(total_reward - random_baseline) / random_baseline * 100:.1f}%")
```

**Expected output:**
```
Random baseline: 0.234 ± 0.156
Starting training...
...
Trained policy: 0.456
Improvement over random: 94.9%
```

---

## 4. Monitoring Script

```python
# utils/monitor.py
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_training_progress(log_dir='./results/tensorboard/'):
    """Plot key metrics from TensorBoard logs"""
    
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Extract metrics
    rewards = ea.Scalars('rollout/ep_rew_mean')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    steps = [r.step for r in rewards]
    values = [r.value for r in rewards]
    axes[0, 0].plot(steps, values)
    axes[0, 0].set_title('Episode Reward Mean')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].grid(True)
    
    # More plots...
    
    plt.tight_layout()
    plt.savefig('./results/training_progress.png')
    print("Saved training plot to ./results/training_progress.png")
```

---

## 5. Debug Checklist Script

```python
# tests/debug_environment.py
from envs.simple_steering_env import SimpleSteeringEnv
import numpy as np

def run_diagnostics():
    env = SimpleSteeringEnv()
    
    print("=" * 50)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 50)
    
    # Test 1: Reset works
    obs, _ = env.reset()
    assert obs.shape == (3,), "Observation shape wrong"
    print("OK: Reset works")
    
    # Test 2: Step works
    action = np.array([0.5])
    obs, reward, done, truncated, info = env.step(action)
    assert isinstance(reward, float), "Reward not float"
    assert 0 <= reward <= 1, f"Reward out of range: {reward}"
    print(f"OK: Step works (reward={reward:.3f})")
    
    # Test 3: Different actions give different rewards
    env.reset()
    rewards = []
    for action_val in [-1.0, 0.0, 1.0]:
        env.reset()
        _, r, _, _, _ = env.step(np.array([action_val]))
        rewards.append(r)
    
    if len(set(rewards)) > 1:
        print(f"OK: Actions affect reward: {rewards}")
    else:
        print(f"Warning: All actions give same reward: {rewards}")
    
    # Test 4: Episode ends
    env.reset()
    for _ in range(25):
        _, _, done, _, _ = env.step(env.action_space.sample())
        if done:
            break
    assert done, "Episode never ends"
    print("OK: Episodes terminate correctly")
    
    # Test 5: Performance range
    env.reset()
    powers = []
    for x in np.linspace(100, 300, 10):
        env.quad_x = x
        result = env._simulate()
        powers.append(result['detectors'][0]['power'])
    
    print(f"OK: Power range: [{min(powers):.3f}, {max(powers):.3f}]")
    
    if max(powers) - min(powers) < 0.1:
        print("Warning: Low power variance, geometry might not matter")
    
    print("\n" + "=" * 50)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    run_diagnostics()
```

---

## Usage Order

1. **First**: Run `test_ray_optics_basic.py` to verify ray-optics works
2. **Second**: Run `tests/debug_environment.py` to verify environment is sane
3. **Third**: Run `main_stage1.py` for initial training
4. **Fourth**: Monitor with TensorBoard: `tensorboard --logdir results/tensorboard/`
5. **Fifth**: Once working, move to Stage 2 (full quadrilateral control)

---

## Critical Success Metrics (Stage 1)

After 20k steps, you should see:
- `ep_rew_mean` > 0.3 (random baseline ~0.2)
- Reward increasing trend (not flat)
- Training time < 15 minutes
- No NaN losses

If any fail, check debug checklist in main workflow document.
