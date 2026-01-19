# Template Reinforcement Learning

Génère un template de projet RL avec Gymnasium et Stable-Baselines3.

## Structure

```
$ARGUMENTS ou "rl-project"/
├── envs/
│   ├── __init__.py
│   └── custom_env.py
├── agents/
│   ├── __init__.py
│   └── agent.py
├── training/
│   ├── __init__.py
│   └── train.py
├── configs/
│   └── config.yaml
├── logs/
├── models/
├── requirements.txt
└── README.md
```

## Fichiers à générer

### requirements.txt
```
gymnasium
stable-baselines3[extra]
tensorboard
wandb
numpy
matplotlib
pyyaml
```

### envs/custom_env.py
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    """Environnement personnalisé compatible Gymnasium."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Définir les espaces d'observation et d'action
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-1, high=1, size=(4,)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # Logique de l'environnement
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Mise à jour de l'état
        self.state = self.state + self.np_random.uniform(-0.1, 0.1, size=(4,)).astype(np.float32)

        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"State: {self.state}")

    def close(self):
        pass
```

### training/train.py
```python
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import yaml

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train(config_path: str = "configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Créer l'environnement
    env_id = config["env"]["id"]
    n_envs = config["env"].get("n_envs", 4)

    env = SubprocVecEnv([make_env(env_id, i, config["seed"]) for i in range(n_envs)])

    # Callbacks
    eval_env = DummyVecEnv([make_env(env_id, 0, config["seed"])])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/checkpoints/",
    )

    # Créer l'agent
    algo = config["algorithm"]["name"]
    model_class = {"PPO": PPO, "SAC": SAC, "DQN": DQN}[algo]

    model = model_class(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        **config["algorithm"].get("params", {}),
    )

    # Entraînement
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],
    )

    model.save("./models/final_model")
    env.close()

if __name__ == "__main__":
    train()
```

### configs/config.yaml
```yaml
seed: 42

env:
  id: CartPole-v1
  n_envs: 4

algorithm:
  name: PPO
  params:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01

training:
  total_timesteps: 1000000
```

Crée tous ces fichiers.
