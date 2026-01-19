# Initialiser un projet ML/PyTorch

Crée la structure de projet ML standard avec les fichiers de base.

## Structure à créer

```
$ARGUMENTS ou "ml-project"/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
├── configs/
│   └── config.yaml
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## Fichiers à générer

### requirements.txt
```
torch>=2.0.0
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
pyyaml
tqdm
tensorboard
wandb
pytest
```

### src/models/model.py
```python
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
```

### src/training/trainer.py
```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(dataloader)
```

### configs/config.yaml
```yaml
seed: 42
device: cuda

data:
  batch_size: 32
  num_workers: 4

model:
  name: BaseModel

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001

logging:
  wandb: false
  tensorboard: true
```

Crée tous ces fichiers avec le contenu approprié.
