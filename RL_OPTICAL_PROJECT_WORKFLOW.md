# RL Optical Design Project: Systematic Development Workflow

## Overview
This document outlines a staged approach to building an RL agent for optical system design using the ray-optics library, with emphasis on avoiding wasted time and maintaining sensible progress through careful metric tracking.

---

## Project Structure

```
optical_rl/
├── main.py                    # Training orchestration, logging, hyperparameter management
├── envs/
│   ├── __init__.py
│   ├── base_optical_env.py    # Base Gymnasium environment
│   ├── simple_steering_env.py # Stage 1: Minimal environment
│   └── quad_steering_env.py   # Stage 2+: Full quadrilateral environment
├── utils/
│   ├── __init__.py
│   ├── ray_optics_bridge.py  # Interface to Node.js runner
│   ├── scene_builder.py       # JSON scene construction
│   ├── metrics.py             # Logging and metric tracking
│   └── visualization.py       # Plot training progress, render scenes
├── configs/
│   ├── stage1_baseline.yaml   # Hyperparameters for each stage
│   ├── stage2_initial_rl.yaml
│   └── stage3_tuned.yaml
├── tests/
│   ├── test_environment.py    # Unit tests for environment
│   └── test_ray_optics.py     # Integration tests with ray-optics
└── results/
    ├── logs/                  # TensorBoard logs
    ├── models/                # Saved model checkpoints
    └── visualizations/        # Generated plots and scene renders
```

---

## Stage 0: Sanity Checks & Validation (Week 1)

**Goal:** Verify that all components work independently before integration.

### Tasks

#### 0.1: Test Ray-Optics Integration
```python
# test_ray_optics.py
def test_basic_scene():
    """Verify Node.js subprocess communication works"""
    scene = {
        "version": 5,
        "objs": [
            {"type": "SingleRay", "p1": {"x": 0, "y": 0}, "p2": {"x": 50, "y": 0}},
            {"type": "Detector", "p1": {"x": 100, "y": -10}, "p2": {"x": 100, "y": 10}}
        ]
    }
    result = run_simulation(scene)
    assert result['detectors'][0]['power'] > 0
    print(f"OK: Basic simulation works: power={result['detectors'][0]['power']}")
```

#### 0.2: Test Manual Scene Variations
```python
def test_refraction_sanity():
    """Verify that adding glass actually changes detector reading"""
    # Scene without glass
    result_no_glass = run_simulation(scene_no_glass)
    
    # Scene with glass blocking path
    result_with_glass = run_simulation(scene_with_glass)
    
    # With glass should bend ray away from detector
    assert result_no_glass['power'] != result_with_glass['power']
    print(f"OK: Glass affects detector: {result_no_glass['power']} -> {result_with_glass['power']}")
```

#### 0.3: Test Geometry Controllability
```python
def test_position_control():
    """Test that moving glass quadrilateral changes outcomes"""
    powers = []
    for x_pos in [50, 100, 150, 200]:
        scene = build_scene_with_quad_at_x(x_pos)
        result = run_simulation(scene)
        powers.append(result['detectors'][0]['power'])
    
    assert len(set(powers)) > 1  # At least one position gives different result
    print(f"OK: Position control works: powers={powers}")
```

### Key Metrics to Check
- Subprocess latency: < 100ms per simulation
- Detector power range: 0.0 to 1.0 (reasonable values)
- Geometry changes produce different outcomes
- Material changes (refIndex) affect results

### Success Criteria
- All tests pass
- Can manually create scenes that:
  - Block the ray (power ≈ 0)
  - Let ray through (power ≈ 1.0)
  - Partially redirect ray (power between 0-1)

---

## Stage 1: Minimal Viable Environment (Week 2)

**Goal:** Create the simplest possible RL environment that can learn *something*.

### 1.1: Ultra-Simple Environment Design

**Simplifications:**
- Fixed quadrilateral shape (rigid body)
- Only control: **X position** (1D action space)
- Fixed material properties (refIndex=1.5)
- Source at (0, 150), Detector at (300, 150)
- Quadrilateral: 50x50 square, only moves horizontally

```python
class SimpleSteeringEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))  # Only X position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        # State: [quad_x_position, detector_power, step_count/max_steps]
        
    def step(self, action):
        # Move quadrilateral horizontally
        self.quad_x += action[0] * 10  # ±10 pixels per step
        self.quad_x = np.clip(self.quad_x, 50, 250)
        
        # Run simulation
        result = self.simulate()
        
        # Simple reward: just detector power
        reward = result['detectors'][0]['power']
        
        self.step_count += 1
        done = self.step_count >= 20  # Short episodes!
        
        obs = np.array([self.quad_x, reward, self.step_count/20])
        return obs, reward, done, {}
```

### 1.2: Baseline Tests

**Before any RL, test with fixed policies:**

```python
def test_random_policy(env, n_episodes=10):
    """Random actions should give ~0.1-0.3 average reward"""
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    
    print(f"Random policy: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
    return np.mean(rewards)

def test_optimal_policy(env, optimal_x=150):
    """Manually place quad at optimal position"""
    env.reset()
    env.quad_x = optimal_x  # Cheat: directly set position
    result = env.simulate()
    print(f"Optimal placement reward: {result['detectors'][0]['power']:.3f}")
```

### 1.3: Initial RL Training

```python
# Stage 1 hyperparameters (conservative, fast iteration)
config = {
    'algorithm': 'PPO',
    'learning_rate': 3e-4,
    'n_steps': 2048,       # Steps per rollout
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'total_timesteps': 20_000,  # Fast: ~10 minutes
}

model = PPO('MlpPolicy', env, **config)
model.learn(total_timesteps=config['total_timesteps'])
```

### Key Metrics to Track

**Stage 1 Success Indicators:**
```python
metrics = {
    'episode_reward_mean': [],      # Should increase over time
    'episode_length_mean': [],      # Should stay ~20
    'value_loss': [],               # Should decrease then stabilize
    'policy_loss': [],              # Should decrease
    'approx_kl': [],                # Should stay < 0.02 (policy not changing too fast)
    'explained_variance': [],       # Should be > 0.5 (value function is useful)
}
```

### Warning Signs
- `episode_reward_mean` not improving after 10k steps: Problem with environment or reward
- `explained_variance` < 0: Value function not learning
- `approx_kl` > 0.1: Learning rate too high
- Episode rewards have huge variance: Need reward normalization

### Success Criteria
- Final policy better than random (mean reward > random baseline + 20%)
- Training stable (no NaN losses)
- Agent learns to move quad toward optimal X position
- Training time < 15 minutes for 20k steps

---

## Stage 2: Full Quadrilateral Control (Week 3-4)

**Goal:** Add complexity gradually, ensure learning still works.

### 2.1: Incremental Complexity Addition

**Add one dimension at a time, verify learning:**

1. **2D position** (X, Y) - keep fixed shape
2. **2D position + rotation** - rigid body
3. **4 corner points** (8D) - deformable quad
4. **4 corners + 1 material property** (refIndex)
5. **4 corners + 2 material properties** (refIndex + cauchyB)

**After each addition:**
- Run training for 50k steps
- Check if learning still works
- If not, rollback and debug

### 2.2: Observation Space Design

```python
class QuadSteeringEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,))
        # [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, d_refIndex, d_cauchyB]
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,))
        # [x1,y1,x2,y2,x3,y3,x4,y4, refIndex, cauchyB, partialReflect,
        #  detector_power, source_x, source_y, step/max_steps]
```

### 2.3: Improved Reward Function

**Start simple, add complexity only if needed:**

```python
def calculate_reward(self, result, previous_power):
    detector_power = result['detectors'][0]['power']
    
    # Stage 2A: Just detector power
    reward = detector_power * 10  # Scale to ~1.0 range
    
    # Stage 2B: Add progress bonus (only if 2A works)
    # improvement = detector_power - previous_power
    # reward += improvement * 5  # Reward improvement
    
    # Stage 2C: Add penalties (only if needed)
    # if self._is_self_intersecting():
    #     reward -= 10
    
    return reward
```

### 2.4: Hyperparameter Tuning Strategy

**Grid search over critical parameters:**

```python
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'n_steps': [1024, 2048, 4096],
    'action_scale': [1.0, 5.0, 10.0],  # How much points move per action
}

# Run 3-5 trials per config, each for 50k steps
# Track: final mean reward, training stability, time to converge
```

### Key Metrics to Track

**Expanded tracking:**
```python
metrics = {
    # Performance
    'episode_reward_mean': [],
    'episode_reward_max': [],
    'best_detector_power': [],       # Best power achieved across all episodes
    
    # Geometry statistics
    'mean_quad_area': [],            # Is quad collapsing to a line?
    'mean_quad_center_x': [],        # Is quad moving toward target zone?
    'self_intersection_rate': [],    # How often quad self-intersects
    
    # Material statistics
    'mean_refIndex': [],             # Is refIndex reasonable?
    'refIndex_std': [],              # Is agent exploring materials?
    
    # Learning health
    'value_loss': [],
    'policy_loss': [],
    'entropy': [],                   # Is agent exploring? Should be > 0.1
    'explained_variance': [],
    
    # Simulation stats
    'mean_ray_count': [],            # Complexity of solutions
    'mean_truncation': [],           # Numerical accuracy
    'simulation_time_ms': [],        # Performance bottleneck?
}
```

### Warning Signs & Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Reward not improving | Action scale too small/large | Tune `action_scale` |
| Quad collapses to line | No shape penalty | Add area regularization |
| High self-intersection rate | No collision penalty | Add intersection penalty |
| RefIndex goes to extremes | No material penalty | Clip or add regularization |
| Learning then degrading | Learning rate too high | Reduce LR by 10x |
| Very slow training | Too many steps/episode | Reduce max_steps to 50 |
| High variance in rewards | Need normalization | Use reward normalization |

### Success Criteria
- Agent achieves > 70% of theoretical maximum detector power
- Quadrilateral maintains reasonable shape (area > min threshold)
- Material properties stay in physical range (1.0 < n < 2.5)
- Training converges in < 200k steps
- No catastrophic forgetting (performance doesn't degrade)

---

## Stage 3: Systematic Hyperparameter Optimization (Week 5)

**Goal:** Find robust hyperparameters through systematic search.

### 3.1: Automated Hyperparameter Search

```python
import optuna

def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    action_scale = trial.suggest_uniform('action_scale', 1.0, 20.0)
    
    # Train for limited time
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, 
                n_steps=n_steps, batch_size=batch_size)
    model.learn(total_timesteps=50_000)
    
    # Evaluate
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)
    
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

### 3.2: Reward Function Coefficient Tuning

**Multi-objective optimization:**

```python
def calculate_reward_v2(self, result):
    power = result['detectors'][0]['power']
    ray_count = result['processedRayCount']
    area = self._calculate_quad_area()
    
    # Coefficients to tune
    w_power = self.config['w_power']        # Start: 10.0
    w_efficiency = self.config['w_efficiency']  # Start: 0.001
    w_area = self.config['w_area']          # Start: 0.1
    
    reward = (w_power * power - 
              w_efficiency * ray_count / 100 +
              w_area * np.log(area + 1))
    
    return reward

# Grid search over w_power, w_efficiency, w_area
# Use Pareto frontier analysis for multi-objective trade-offs
```

### 3.3: Curriculum Learning

**Gradually increase difficulty:**

```python
class CurriculumEnv(QuadSteeringEnv):
    def __init__(self, difficulty=1):
        super().__init__()
        self.difficulty = difficulty
        
    def reset(self):
        if self.difficulty == 1:
            # Easy: Source and detector close, large detector
            self.source_pos = {"x": 50, "y": 150}
            self.detector_pos = {"x": 200, "y": 150}
            self.detector_size = 40
            
        elif self.difficulty == 2:
            # Medium: Further apart
            self.source_pos = {"x": 50, "y": 150}
            self.detector_pos = {"x": 350, "y": 150}
            self.detector_size = 20
            
        elif self.difficulty == 3:
            # Hard: Different heights, small detector
            self.source_pos = {"x": 50, "y": 100}
            self.detector_pos = {"x": 350, "y": 200}
            self.detector_size = 10
        
        return super().reset()

# Training schedule:
# - Train on difficulty 1 until > 80% success
# - Switch to difficulty 2 until > 70% success
# - Switch to difficulty 3
```

### Success Criteria
- Found hyperparameters that work across 5 random seeds
- Training time < 300k steps for convergence
- Agent solves easy tasks > 90%, medium tasks > 70%

---

## Stage 4: Advanced Features (Week 6+)

### 4.1: Multi-Wavelength Training
- Add RGB sources (3 rays)
- Reward for focusing all colors to same point
- Penalize chromatic aberration

### 4.2: Multiple Obstacles
- Add 2-3 quadrilaterals
- Coordinate multi-element systems
- Explore lens combinations

### 4.3: Adaptive Complexity
- Start with simple shapes, allow agent to add complexity
- Action masking for invalid geometries
- Hierarchical RL (meta-controller + element-controller)

---

## Monitoring & Debugging Guidelines

### Essential Logging

```python
class MetricsLogger:
    def log_episode(self, episode_data):
        # Log to TensorBoard
        writer.add_scalar('reward/episode', episode_data['total_reward'])
        writer.add_scalar('geometry/quad_area', episode_data['final_area'])
        writer.add_scalar('physics/detector_power', episode_data['final_power'])
        
        # Save checkpoints
        if episode_data['total_reward'] > self.best_reward:
            model.save(f'models/best_model_ep{episode}')
        
        # Save visualization every 100 episodes
        if episode % 100 == 0:
            self.visualize_scene(episode_data['final_scene'])
```

### Debugging Checklist

When training fails or plateaus:

1. **Check environment sanity**
   - [ ] Can random policy get non-zero reward?
   - [ ] Does optimal manual solution exist?
   - [ ] Is action space bounded correctly?

2. **Check reward signal**
   - [ ] Is reward too sparse? (Add dense rewards)
   - [ ] Is reward too noisy? (Add smoothing)
   - [ ] Does reward correlate with desired behavior?

3. **Check learning algorithm**
   - [ ] Is value function learning? (check explained_variance)
   - [ ] Is policy exploring? (check entropy)
   - [ ] Are gradients exploding? (check gradient norms)

4. **Check hyperparameters**
   - [ ] Is learning rate appropriate? (try 10x smaller/larger)
   - [ ] Is action scale appropriate? (check action distributions)
   - [ ] Are episodes too long? (try shorter episodes first)

---

## Timeline & Milestones

### Week 1: Foundation
- [ ] Complete Stage 0 sanity checks
- [ ] Verify ray-optics integration
- [ ] Establish baseline metrics

### Week 2: First Success
- [ ] Simple 1D environment working
- [ ] Agent better than random
- [ ] Metrics tracking in place

### Week 3-4: Full Environment
- [ ] Quadrilateral control working
- [ ] Reasonable solutions found
- [ ] Hyperparameter tuning started

### Week 5: Optimization
- [ ] Robust hyperparameters found
- [ ] Multiple seeds successful
- [ ] Publication-ready results

### Week 6+: Advanced
- [ ] Multi-wavelength working
- [ ] Complex scenes handled
- [ ] Generalization tested

---

## Key Principles

1. **Start Simple**: Always begin with the simplest possible version
2. **Validate Early**: Test each component independently before integration
3. **Measure Everything**: Track metrics to diagnose problems quickly
4. **Iterate Fast**: Keep episodes short (20-50 steps) for rapid experimentation
5. **One Change at a Time**: Add complexity incrementally, verify learning still works
6. **Baselines Matter**: Always compare to random and optimal policies
7. **Save Checkpoints**: Save models frequently, especially when performance improves
8. **Visualize**: Plot training curves and render scenes to understand behavior

---

## Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| Training takes days | Reduce episode length, fewer total steps, faster simulation |
| Can't beat random baseline | Simplify environment, check reward scaling |
| Learning then forgetting | Lower learning rate, increase batch size |
| Agent finds trivial solution | Add constraints, penalize degenerate cases |
| Unstable training | Normalize observations, clip gradients, reduce LR |
| No exploration | Increase entropy coefficient, add noise to actions |
| Overfitting to one config | Train with randomized source/detector positions |

---

## Success Metrics Summary

**Minimum Viable Product (MVP):**
- Agent achieves 50% of optimal detector power
- Training converges in < 100k steps
- Solutions are physically reasonable (valid geometry, realistic materials)

**Research-Grade Results:**
- Agent achieves 80%+ of optimal power
- Generalizes to new source/detector configurations
- Discovers non-trivial optical designs (e.g., lens-like focusing)
- Outperforms hand-designed baselines

**Publication-Grade Results:**
- Robust across multiple random seeds (std < 10%)
- Scales to complex multi-element systems
- Ablation studies show which components matter
- Interpretable learned policies (can explain why agent made decisions)
