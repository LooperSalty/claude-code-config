# Debug Tensor

Analyse et debug les tensors PyTorch dans le code.

## Instructions

Quand l'utilisateur demande de debugger un tensor ou a des problèmes de shape:

1. **Analyser le code** pour trouver les opérations sur tensors
2. **Identifier les shapes** à chaque étape
3. **Détecter les erreurs** courantes:
   - Shape mismatch
   - Device mismatch (CPU vs CUDA)
   - Type mismatch (float vs long)
   - Broadcasting issues

## Template de debug

```python
def debug_tensor(t: torch.Tensor, name: str = "tensor") -> None:
    """Affiche les infos de debug d'un tensor."""
    print(f"=== {name} ===")
    print(f"  Shape: {t.shape}")
    print(f"  Dtype: {t.dtype}")
    print(f"  Device: {t.device}")
    print(f"  Requires grad: {t.requires_grad}")
    print(f"  Min: {t.min().item():.4f}, Max: {t.max().item():.4f}")
    print(f"  Mean: {t.mean().item():.4f}, Std: {t.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(t).any().item()}")
    print(f"  Has Inf: {torch.isinf(t).any().item()}")
```

## Erreurs courantes à vérifier

| Erreur | Cause | Solution |
|--------|-------|----------|
| `RuntimeError: size mismatch` | Shapes incompatibles | Vérifier les dimensions avec .shape |
| `RuntimeError: expected cuda` | Device mismatch | Utiliser .to(device) |
| `RuntimeError: expected Float` | Type incorrect | Utiliser .float() ou .long() |
| `Loss is NaN` | Gradient explosion | Réduire lr, ajouter gradient clipping |

Propose des solutions concrètes basées sur l'erreur rencontrée.
