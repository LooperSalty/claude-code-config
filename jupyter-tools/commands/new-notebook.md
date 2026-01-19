# Créer un nouveau Notebook

Génère un notebook Jupyter avec une structure de base pour l'analyse de données ou le ML.

## Templates disponibles

### 1. Data Analysis

```python
# Cellule 1 - Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
%matplotlib inline
```

```python
# Cellule 2 - Chargement des données
df = pd.read_csv('data.csv')
print(f"Shape: {df.shape}")
df.head()
```

```python
# Cellule 3 - Exploration
df.info()
df.describe()
```

```python
# Cellule 4 - Valeurs manquantes
missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)
```

```python
# Cellule 5 - Visualisation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# À personnaliser selon les données
plt.tight_layout()
plt.show()
```

### 2. Machine Learning

```python
# Cellule 1 - Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Configuration
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

np.random.seed(SEED)
torch.manual_seed(SEED)
```

```python
# Cellule 2 - Chargement et préparation
# Charger les données
df = pd.read_csv('data.csv')

# Séparer features et target
X = df.drop('target', axis=1).values
y = df['target'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

```python
# Cellule 3 - DataLoaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

```python
# Cellule 4 - Modèle
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

model = Model(X_train.shape[1], 128, len(np.unique(y))).to(DEVICE)
print(model)
```

```python
# Cellule 5 - Entraînement
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
```

```python
# Cellule 6 - Évaluation
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.numpy())

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
```

```python
# Cellule 7 - Visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(train_losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1], cmap='Blues')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')

plt.tight_layout()
plt.show()
```

Quel type de notebook veux-tu créer ? (data-analysis / ml / custom)
