# Configuration Claude Code

Fichiers de configuration personnalisés pour [Claude Code](https://claude.ai/claude-code).

## Fichiers

| Fichier | Description | Emplacement |
|---------|-------------|-------------|
| `settings.json` | Paramètres et permissions | `~/.claude/settings.json` |
| `CLAUDE.md` | Instructions et préférences | Racine du projet |

## Installation

### Configuration globale (tous les projets)

```bash
# Copier settings.json
cp settings.json ~/.claude/settings.json
```

### Configuration par projet

```bash
# Copier CLAUDE.md à la racine de ton projet
cp CLAUDE.md /chemin/vers/ton/projet/CLAUDE.md
```

## Personnalisation

### settings.json

Modifier les permissions selon tes besoins:

```json
{
  "permissions": {
    "allow": ["Bash(commande autorisée)"],
    "deny": ["Bash(commande interdite)"]
  }
}
```

### CLAUDE.md

Adapter les instructions selon:
- Tes langages préférés
- Tes frameworks
- Tes conventions de code
- Ta langue de communication

## Contenu

Cette configuration est optimisée pour:
- **ML/IA**: PyTorch, scikit-learn, Gymnasium
- **Web 3D**: Three.js, React Three Fiber, Babylon.js
- **Game Dev**: Godot, Pygame, Bevy
- **Web**: React, Next.js, FastAPI

## Licence

MIT
