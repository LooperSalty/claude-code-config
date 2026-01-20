# Configuration Claude Code

Fichiers de configuration, plugins et serveurs MCP personnalisés pour [Claude Code](https://claude.ai/claude-code).

## Contenu

| Élément | Description |
|---------|-------------|
| `settings.json` | Permissions (accès complet) |
| `CLAUDE.md` | Instructions en français, conventions |
| `mcp-servers.json` | Configuration serveurs MCP |
| `pytorch-helper/` | Plugin ML/PyTorch |
| `threejs-snippets/` | Plugin Three.js/R3F |
| `jupyter-tools/` | Plugin Jupyter notebooks |
| `frontend-design/` | Plugin frontend design de qualité |

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
cp -r frontend-design ~/.claude/plugins/
```

### 3. Serveurs MCP

```bash
# Context7 - Documentation à jour pour libs/frameworks
claude mcp add context7 -- npx -y @upstash/context7-mcp@latest

# Playwright - Automatisation navigateur
claude mcp add playwright -- npx @playwright/mcp@latest

# Figma - Accès designs et composants
claude mcp add figma -- npx -y figma-developer-mcp --stdio
```

### 4. Instructions projet

```bash
cp CLAUDE.md /chemin/vers/ton/projet/
```

## Serveurs MCP

### Context7

Documentation à jour pour n'importe quelle librairie.

```bash
# Installation
claude mcp add context7 -- npx -y @upstash/context7-mcp@latest
```

**Utilisation** : Ajoute `use context7` dans ton prompt pour obtenir la doc à jour.

```
use context7 Comment utiliser useQuery dans React Query v5 ?
```

### Playwright

Automatisation de navigateur contrôlée par Claude.

```bash
# Installation
claude mcp add playwright -- npx @playwright/mcp@latest
```

**Utilisation** :
- Tests E2E automatisés
- Scraping de pages web
- Screenshots et captures
- Authentification automatique

```
Utilise playwright pour ouvrir https://example.com et prendre un screenshot
```

### Figma

Accès aux designs Figma directement dans Claude.

```bash
# Installation
claude mcp add figma -- npx -y figma-developer-mcp --stdio

# Configuration token (requis)
# 1. Créer un token sur https://www.figma.com/developers/api#access-tokens
# 2. Ajouter dans mcp-servers.json :
#    "env": { "FIGMA_ACCESS_TOKEN": "ton_token_ici" }
```

**Utilisation** :
- Extraire composants et styles
- Générer code depuis designs
- Analyser structure UI

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

### frontend-design

Plugin officiel Claude Code pour créer des interfaces frontend modernes et uniques.

| Commande | Description |
|----------|-------------|
| `/frontend-design` | Créer des composants UI de qualité professionnelle |

**Caractéristiques** :
- Génère des designs distinctifs et créatifs
- Évite l'esthétique générique des IA
- Code production-ready avec React/Tailwind
- Composants web modernes et responsifs

**Installation** :
```bash
/plugin marketplace add anthropics/claude-code
/plugin install frontend-design@claude-code
```

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

## Vérifier les MCP installés

```bash
# Liste des serveurs MCP
claude mcp list

# Dans une session interactive
/mcp
```

## Optimisé pour

- **ML/IA**: PyTorch, Gymnasium, stable-baselines3
- **Web 3D**: Three.js, React Three Fiber, GLSL
- **Game Dev**: Godot, Pygame, Bevy
- **Web**: React, Next.js, FastAPI
- **Automatisation**: Playwright, tests E2E
- **Design**: Figma, UI/UX

## Ressources

- [Context7 Documentation](https://context7.com/docs)
- [Playwright MCP](https://github.com/microsoft/playwright-mcp)
- [Figma MCP](https://github.com/anthropics/figma-developer-mcp)
- [Claude Code Docs](https://docs.anthropic.com/claude-code)

## Note sur Mobbin

Mobbin n'a pas encore de serveur MCP officiel. En alternative, utilise :
- **Figma MCP** pour accéder aux designs
- **Playwright MCP** pour capturer des screenshots de Mobbin
- **Web to MCP** (extension Chrome) pour envoyer des composants à Claude

## Licence

MIT
