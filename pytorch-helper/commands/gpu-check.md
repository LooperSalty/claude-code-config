# GPU Check

Vérifie la disponibilité et l'état du GPU pour PyTorch.

## Commande à exécuter

```python
import torch

print("=== Configuration PyTorch ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n=== GPU {i}: {props.name} ===")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
else:
    print("Aucun GPU CUDA détecté. Utilisation du CPU.")

# Test rapide
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1000, 1000, device=device)
y = torch.matmul(x, x)
print(f"\nTest matmul sur {device}: OK")
```

Exécute ce code Python et affiche les résultats.
