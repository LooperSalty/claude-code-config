# Nettoyer un Notebook

Supprime les outputs et métadonnées des notebooks pour un commit git propre.

## Avec nbstripout (recommandé)

```bash
# Installation
pip install nbstripout

# Nettoyer un notebook
nbstripout notebook.ipynb

# Configurer git pour nettoyer automatiquement
nbstripout --install
```

## Avec jupyter nbconvert

```bash
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

## Script Python personnalisé

```python
import json
import sys

def clean_notebook(path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        # Supprimer les outputs
        if 'outputs' in cell:
            cell['outputs'] = []
        # Supprimer le compteur d'exécution
        if 'execution_count' in cell:
            cell['execution_count'] = None
        # Nettoyer les métadonnées de cellule
        if 'metadata' in cell:
            cell['metadata'] = {}

    # Nettoyer les métadonnées du notebook
    if 'metadata' in nb:
        # Garder seulement kernelspec et language_info
        nb['metadata'] = {
            k: v for k, v in nb['metadata'].items()
            if k in ['kernelspec', 'language_info']
        }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

    print(f"Cleaned: {path}")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        clean_notebook(path)
```

## Git hook pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout
```

Quel notebook veux-tu nettoyer ?
