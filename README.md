# Configuration Claude Code

Fichiers de configuration et plugins personnalisés pour [Claude Code](https://claude.ai/claude-code).

## Contenu

| Élément | Description |
|---------|-------------|
| `settings.json` | Permissions (accès complet) |
| `CLAUDE.md` | Instructions en français, conventions |
| `pytorch-helper/` | Plugin ML/PyTorch |
| `threejs-snippets/` | Plugin Three.js/R3F |
| `jupyter-tools/` | Plugin Jupyter notebooks |

## Installation

### 1. Configuration globale

```bash
cp settings.json ~/.claude/settings.json
```

### 2. Plugins

```bash
cp -r pytorch-helper ~/.claude/plugins/
cp -r threejs-snippets ~/.claude/plugins/
cp -r jupyter-tools ~/.claude/plugins/
```

### 3. Instructions projet

```bash
cp CLAUDE.md /chemin/vers/ton/projet/
```

## Plugins personnalisés

### pytorch-helper

| Commande | Description |
|----------|-------------|
| `/pytorch-helper:init-project` | Créer structure projet ML |
| `/pytorch-helper:debug-tensor` | Debug shapes et tensors |
| `/pytorch-helper:gpu-check` | Vérifier config GPU/CUDA |
| `/pytorch-helper:rl-template` | Template Reinforcement Learning |

### threejs-snippets

| Commande | Description |
|----------|-------------|
| `/threejs-snippets:init-r3f` | Projet React Three Fiber |
| `/threejs-snippets:shader` | Templates shaders GLSL |
| `/threejs-snippets:physics` | Ajouter physique Rapier/Cannon |
| `/threejs-snippets:model` | Charger modèles GLTF/GLB |

### jupyter-tools

| Commande | Description |
|----------|-------------|
| `/jupyter-tools:new-notebook` | Créer notebook ML/Data |
| `/jupyter-tools:convert` | Convertir notebook (HTML, PDF, .py) |
| `/jupyter-tools:clean` | Nettoyer outputs pour git |

## Plugins officiels recommandés

```bash
# LSP (autocomplétion, diagnostics)
/plugin install pyright-lsp
/plugin install typescript-lsp
/plugin install rust-analyzer-lsp

# Workflow
/plugin install github
/plugin install commit-commands
```

## Optimisé pour

- **ML/IA**: PyTorch, Gymnasium, stable-baselines3
- **Web 3D**: Three.js, React Three Fiber, GLSL
- **Game Dev**: Godot, Pygame, Bevy
- **Web**: React, Next.js, FastAPI

## Licence

MIT
