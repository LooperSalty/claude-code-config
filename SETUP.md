# SETUP — Reconstruire l'environnement de dev (nouvelle machine)

> Guide maître pour Claude **et** pour ANAKIN. Décrit **tout** ce qu'il faut installer et
> configurer pour retrouver l'environnement de dev complet sur une machine vierge.
>
> **Cible** : Windows 11. Shell de référence : **PowerShell**.
> **Raccourci** : exécuter [`bootstrap.ps1`](bootstrap.ps1) automatise les sections 1 à 4.

---

## 0. Profil

- **Utilisateur** : Paul RUIZ (ANAKIN / LooperSalty) — France
- **GitHub** : [github.com/LooperSalty](https://github.com/LooperSalty) (plan Pro). ⚠️ Repos **privés par défaut** (`gh repo create --private`).
- **Domaines** : Game Dev (Roblox/Luau, Unreal, Bevy, C++/OpenGL, Unity), IA/ML (PyTorch, multi-agents), Web (TS/React/Next/Three.js), Backend (Node, FastAPI, Laravel/Symfony), Systems (Rust, C), Mobile (Flutter).

---

## 1. Gestionnaire de paquets & runtimes (WinGet)

Tout s'installe via **WinGet**. Lancer PowerShell puis :

```powershell
# --- Runtimes & langages ---
winget install --id OpenJS.NodeJS.LTS          # Node.js 24 LTS (+ npm)
winget install --id Python.Python.3.12         # Python 3.12 (runtime principal)
winget install --id Astral.UV                  # uv / uvx (gestionnaire Python rapide, MCP)
winget install --id Rustlang.Rustup            # Rust (rustup -> cargo, rustc)
winget install --id Microsoft.OpenJDK.17       # JDK 17 (Java/Spring)
winget install --id Microsoft.DotNet.SDK.10    # .NET SDK 10 (C#)

# --- Outils ---
winget install --id Git.Git                    # Git
winget install --id GitHub.cli                 # gh (GitHub CLI)
winget install --id Microsoft.VisualStudioCode # VS Code (EDITOR par defaut)
winget install --id Docker.DockerDesktop       # Docker Desktop
winget install --id BurntSushi.ripgrep.MSVC    # ripgrep (rg)

# --- Game dev ---
winget install --id Rojo.Rojo                  # Rojo (sync filesystem <-> Roblox Studio)
winget install --id Rojo.Rokit                 # Rokit (toolchain manager Roblox)
winget install --id BlenderFoundation.Blender  # Blender (+ MCP)
winget install --id EpicGames.EpicGamesLauncher # Epic Launcher (-> Unreal Engine 5.8)
winget install --id Microsoft.VisualStudio.Community # VS Community (outils C++ pour Unreal/plugins)
```

> **Bun** (`bun 1.3.x`) est installé via npm global (section 3), pas WinGet.
> **Roblox Studio** : à installer depuis [create.roblox.com](https://create.roblox.com) — doit être lancé ≥1 fois pour générer les fichiers/clés de registre nécessaires aux MCP.

### Versions de référence (snapshot)

| Outil | Version | | Outil | Version |
|-------|---------|---|-------|---------|
| Node  | 24.x LTS | | Rust | 1.95 |
| npm   | 11.x | | Java (OpenJDK) | 17 |
| Python | 3.12 (+ 3.14 dispo) | | .NET SDK | 10 |
| uv    | 0.10 | | Git | 2.5x |
| bun   | 1.3 | | gh | 2.8x |
| Rojo  | 7.5 | | Docker | 29.x |

---

## 2. Claude Code

```powershell
# Installation native recommandee (npm deprecie)
irm https://claude.ai/install.ps1 | iex
```

Puis restaurer la config globale (`~/.claude`) — voir section 5.

---

## 3. Paquets npm globaux

```powershell
npm install -g `
  @google/gemini-cli `   # Gemini CLI (orchestration multi-agents)
  @playwright/mcp `      # MCP Playwright
  bun `                  # Runtime Bun
  eas-cli `              # Expo Application Services (mobile/Flutter-React Native)
  firebase-tools `       # Firebase
  supabase `             # Supabase CLI
  vercel                 # Vercel CLI
```

## 4. Outils uv (Python global)

```powershell
uv tool install free-claude-code   # fcc-* (free-claude-code)
uv tool install kimi-cli           # kimi / kimi-cli
```

> Les serveurs MCP en `uvx`/`uv` (blender-mcp, unreal) sont gérés à la volée — pas besoin de pré-installer.

---

## 5. Config globale Claude Code (`~/.claude`)

Ces fichiers sont versionnés ici sous [`claude-home/`](claude-home/) et [`settings.json`](settings.json).

```powershell
$dst = "$env:USERPROFILE\.claude"
New-Item -ItemType Directory -Force $dst, "$dst\rules", "$dst\agents", "$dst\hooks" | Out-Null

Copy-Item settings.json              "$dst\settings.json" -Force
Copy-Item claude-home\CLAUDE.md      "$dst\CLAUDE.md" -Force
Copy-Item claude-home\rules\*.md     "$dst\rules\" -Force
Copy-Item claude-home\agents\*.md    "$dst\agents\" -Force
Copy-Item claude-home\hooks\*.ps1    "$dst\hooks\" -Force
```

### Contenu de `claude-home/`

| Élément | Rôle |
|---------|------|
| `CLAUDE.md` | Instructions globales : **autonomie totale**, français, style direct. |
| `rules/` (8) | `agents`, `coding-style`, `git-workflow`, `hooks`, `patterns`, `performance`, `security`, `testing`. |
| `agents/` (13) | architect, planner, code-reviewer, security-reviewer, tdd-guide, build-error-resolver, e2e-runner, refactor-cleaner, doc-updater, database-reviewer, python-reviewer, go-reviewer, go-build-resolver. |
| `hooks/session-end-to-obsidian.ps1` | Log de fin de session vers l'Obsidian Vault (voir section 8). |

### `settings.json` — points clés

- `permissions.defaultMode: "bypassPermissions"` + allowlist `Bash(*)`/MCP → **autonomie totale** (préférence ANAKIN).
- `effortLevel: "xhigh"`, `autoUpdatesChannel: "latest"`, `tui: "fullscreen"`, `voiceEnabled: true`.
- Hook `SessionEnd` → `session-end-to-obsidian.ps1`.
- `enabledPlugins` + `extraKnownMarketplaces` → voir section 7.

---

## 6. Serveurs MCP

Inventaire complet dans [`mcp-servers.json`](mcp-servers.json) (secrets = placeholders `${VAR}`).
Renseigner d'abord les secrets — voir [`.env.example`](.env.example) :

```powershell
[Environment]::SetEnvironmentVariable("FIGMA_ACCESS_TOKEN","<token>","User")
[Environment]::SetEnvironmentVariable("FAL_KEY","<cle>","User")
[Environment]::SetEnvironmentVariable("ROBLOX_OPEN_CLOUD_API_KEY","<cle>","User")
[Environment]::SetEnvironmentVariable("GITHUB_PERSONAL_ACCESS_TOKEN","<pat>","User")
```

### 6.1 MCP globaux (npx / uvx — zéro install préalable)

```powershell
claude mcp add context7   -- npx -y @upstash/context7-mcp@latest
claude mcp add playwright -- npx @playwright/mcp@latest --browser chrome
claude mcp add figma      --env FIGMA_ACCESS_TOKEN=$env:FIGMA_ACCESS_TOKEN -- npx -y figma-developer-mcp --stdio
claude mcp add fal-ai     --env FAL_KEY=$env:FAL_KEY -- npx -y fal-ai-mcp-server
claude mcp add robloxstudio --env ROBLOX_OPEN_CLOUD_API_KEY=$env:ROBLOX_OPEN_CLOUD_API_KEY -- npx -y robloxstudio-mcp@latest
claude mcp add blender    -- uvx blender-mcp
claude mcp add desktop-computer-use -- npx -y computer-use-mcp
```

### 6.2 MCP nécessitant un repo/build local

Ces serveurs pointent vers du code local — il faut cloner/builder d'abord.

**`unreal`** — Unreal Engine 5.8 (fork chongdashu/unreal-mcp)
```powershell
git clone https://github.com/chongdashu/unreal-mcp.git "$env:USERPROFILE\unreal-mcp"
cd "$env:USERPROFILE\unreal-mcp\Python"; uv sync
claude mcp add unreal -- uv --directory "$env:USERPROFILE/unreal-mcp/Python" run unreal_mcp_server.py
```
> Nécessite aussi le plugin C++ `UnrealMCP` compilé dans le projet UE (bridge TCP `127.0.0.1:55557`). Détails et patches de migration 5.5→5.8 dans `~/.claude/.../memory/unreal-dev.md`.

**`windows-desktop`** — automatisation desktop Windows (repo perso)
```powershell
git clone git@github.com:LooperSalty/mcp-desktop-windows.git "$env:USERPROFILE\Documents\Projets\mcp_desktop_windows"
pip install -r "$env:USERPROFILE\Documents\Projets\mcp_desktop_windows\requirements.txt"
claude mcp add windows-desktop -- python "$env:USERPROFILE/Documents/Projets/mcp_desktop_windows/server.py"
```

**`roblox-studio-official`** — MCP officiel Roblox (scope projet `roblox-tools`)
> Le binaire `rbx-studio-mcp.exe` est fourni par Roblox Studio (plugin officiel "MCP"). Le déposer dans `~/roblox-tools/` puis l'enregistrer en scope projet. Voir `memory/roblox-dev.md`.

### 6.3 MCP scopés par projet

| Projet | Serveur | Note |
|--------|---------|------|
| `roblox-tools` | `roblox-studio-official` | binaire `.exe` officiel |
| `Desktop/CRUMBLE` | `github` | `@modelcontextprotocol/server-github` + PAT |
| `Projets/synapse` | `blender` | idem global |
| `Projets/Ascension` | `ascension-rpg-test` | build local `mcp-server/dist/index.js` (`npm i && npm run build`) |

### 6.4 Roblox — prérequis Studio

- Installer Roblox Studio, le lancer ≥1 fois.
- Plugin MCP : `MCPPlugin.rbxmx` → `%LOCALAPPDATA%\Roblox\Plugins\`.
- Dans Studio : activer **Allow HTTP Requests** (Game Settings → Security).
- Rojo : `rojo serve` (sync live) — template dans `~/roblox-projects/templates/`.

### 6.5 Blender — prérequis

- Installer l'addon `blender-mcp` dans Blender (Edit → Preferences → Add-ons), puis activer le serveur dans le panneau latéral (onglet BlenderMCP).

---

## 7. Plugins & marketplaces

Marketplaces à ajouter (déjà déclarés dans `settings.json > extraKnownMarketplaces`) :

```
anthropics/skills              -> anthropic-agent-skills
nextlevelbuilder/ui-ux-pro-max-skill -> ui-ux-pro-max-skill
ruvnet/ruflo                   -> ruflo
```

Plugins activés (`enabledPlugins`) — officiels `claude-plugins-official` sauf indication :
`frontend-design`, `context7`, `superpowers`, `code-review`, `github`, `figma`, `supabase`,
`security-guidance`, `claude-code-setup`, `vercel`, `document-skills` (anthropic-agent-skills),
`ui-ux-pro-max` (ui-ux-pro-max-skill).

```powershell
# Après restauration de settings.json, les plugins se réinstallent au lancement.
# Sinon, manuellement par ex. :
claude  # puis dans la session :
# /plugin marketplace add anthropics/claude-code
# /plugin install frontend-design@claude-code
```

### Plugins custom de ce repo

`pytorch-helper/`, `threejs-snippets/`, `jupyter-tools/`, `frontend-design/` → voir [README.md](README.md).

```powershell
Copy-Item -Recurse pytorch-helper,threejs-snippets,jupyter-tools "$env:USERPROFILE\.claude\plugins\" -Force
```

### Skills

~76 skills installées. Sources (voir `memory/skills-inventory.md`) :
- `github.com/anthropics/skills` (officiel)
- `github.com/sickn33/antigravity-awesome-skills` (1370+)
- `github.com/travisvn/awesome-claude-skills`
- `skills.pawgrammer.com`, `claudemarketplaces.com`

---

## 8. Dépendances externes notables

- **Obsidian** : le hook `session-end-to-obsidian.ps1` écrit dans `C:\Users\ANAKIN\Documents\Obsidian Vault\Sessions\`. Installer Obsidian et créer ce vault (ou adapter le chemin dans le `.ps1`).
- **Visual Studio Community + outils C++** : requis pour compiler le plugin Unreal `UnrealMCP`.
- **Unreal Engine 5.8** (via Epic Launcher) pour le MCP `unreal`.

---

## 9. Secrets à régénérer / rotation

Voir [`.env.example`](.env.example). Tokens requis : `FIGMA_ACCESS_TOKEN`, `FAL_KEY`,
`ROBLOX_OPEN_CLOUD_API_KEY`, `GITHUB_PERSONAL_ACCESS_TOKEN`.

> ⚠️ **Sécurité** : ne jamais committer de vraie clé. Toujours passer par variables
> d'environnement / `.env` (gitignore). Voir [SECURITY.md](SECURITY.md).

---

## 10. Checklist de vérification

```powershell
node -v; npm -v; python --version; uv --version; cargo --version; gh --version; rojo --version
claude --version
claude mcp list            # tous les MCP attendus presents ?
gh auth status             # connecte a GitHub ?
```
