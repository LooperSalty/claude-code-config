#requires -Version 5.1
<#
.SYNOPSIS
    Bootstrap de l'environnement de dev ANAKIN sur une machine Windows vierge.
.DESCRIPTION
    Installe runtimes/outils (WinGet), Claude Code, paquets npm globaux, outils uv,
    restaure la config globale ~/.claude, puis enregistre les serveurs MCP.
    Idempotent : peut etre relance sans casser l'existant.
.PARAMETER SkipWinget
    Saute l'installation des paquets WinGet.
.PARAMETER SkipMcp
    Saute l'enregistrement des serveurs MCP.
.EXAMPLE
    pwsh -File bootstrap.ps1
.NOTES
    Voir SETUP.md pour le detail et les etapes manuelles (Unreal, Roblox, Obsidian).
#>
param(
    [switch]$SkipWinget,
    [switch]$SkipMcp
)

$ErrorActionPreference = 'Continue'
$RepoDir = $PSScriptRoot
function Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [!] $m" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# 1. WinGet : runtimes & outils
# ---------------------------------------------------------------------------
if (-not $SkipWinget) {
    Step "WinGet - runtimes & outils"
    $pkgs = @(
        'OpenJS.NodeJS.LTS','Python.Python.3.12','Astral.UV','Rustlang.Rustup',
        'Microsoft.OpenJDK.17','Microsoft.DotNet.SDK.10',
        'Git.Git','GitHub.cli','Microsoft.VisualStudioCode','Docker.DockerDesktop',
        'BurntSushi.ripgrep.MSVC',
        'Rojo.Rojo','Rojo.Rokit','BlenderFoundation.Blender',
        'EpicGames.EpicGamesLauncher','Microsoft.VisualStudio.Community'
    )
    foreach ($p in $pkgs) {
        Write-Host "  installing $p ..."
        winget install --id $p --accept-source-agreements --accept-package-agreements -e --silent 2>$null
    }
    Ok "WinGet termine (relancer le shell pour rafraichir le PATH)"
} else { Warn "WinGet saute (-SkipWinget)" }

# ---------------------------------------------------------------------------
# 2. Claude Code
# ---------------------------------------------------------------------------
Step "Claude Code"
if (Get-Command claude -ErrorAction SilentlyContinue) {
    Ok "Claude Code deja installe ($(claude --version))"
} else {
    irm https://claude.ai/install.ps1 | iex
    Ok "Claude Code installe"
}

# ---------------------------------------------------------------------------
# 3. npm globaux
# ---------------------------------------------------------------------------
Step "Paquets npm globaux"
if (Get-Command npm -ErrorAction SilentlyContinue) {
    npm install -g `@google/gemini-cli `@playwright/mcp bun eas-cli firebase-tools supabase vercel
    Ok "npm globaux installes"
} else { Warn "npm introuvable - relancer apres rafraichissement du PATH" }

# ---------------------------------------------------------------------------
# 4. Outils uv
# ---------------------------------------------------------------------------
Step "Outils uv"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv tool install free-claude-code 2>$null
    uv tool install kimi-cli 2>$null
    Ok "uv tools installes"
} else { Warn "uv introuvable - relancer apres rafraichissement du PATH" }

# ---------------------------------------------------------------------------
# 5. Restauration config globale ~/.claude
# ---------------------------------------------------------------------------
Step "Config globale ~/.claude"
$dst = "$env:USERPROFILE\.claude"
New-Item -ItemType Directory -Force $dst,"$dst\rules","$dst\agents","$dst\hooks" | Out-Null
Copy-Item "$RepoDir\settings.json"          "$dst\settings.json" -Force
Copy-Item "$RepoDir\claude-home\CLAUDE.md"  "$dst\CLAUDE.md" -Force
Copy-Item "$RepoDir\claude-home\rules\*.md"  "$dst\rules\"  -Force
Copy-Item "$RepoDir\claude-home\agents\*.md" "$dst\agents\" -Force
Copy-Item "$RepoDir\claude-home\hooks\*.ps1" "$dst\hooks\"  -Force
Ok "Config restauree dans $dst"

# Plugins custom du repo
$pdir = "$dst\plugins"
New-Item -ItemType Directory -Force $pdir | Out-Null
foreach ($pl in 'pytorch-helper','threejs-snippets','jupyter-tools') {
    if (Test-Path "$RepoDir\$pl") { Copy-Item -Recurse -Force "$RepoDir\$pl" $pdir }
}
Ok "Plugins custom copies"

# ---------------------------------------------------------------------------
# 6. Secrets (env utilisateur)
# ---------------------------------------------------------------------------
Step "Secrets"
$envFile = "$RepoDir\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -match '^\s*[A-Z_]+\s*=' -and $_ -notmatch '^\s*#' } | ForEach-Object {
        $k,$v = $_ -split '=',2
        $k = $k.Trim(); $v = $v.Trim()
        if ($v) { [Environment]::SetEnvironmentVariable($k,$v,'User'); $env:$k = $v; Ok "env: $k" }
    }
} else {
    Warn "Pas de .env - copier .env.example en .env et renseigner les tokens (Figma, fal, Roblox, GitHub)"
}

# ---------------------------------------------------------------------------
# 7. Serveurs MCP globaux (npx/uvx)
# ---------------------------------------------------------------------------
if (-not $SkipMcp -and (Get-Command claude -ErrorAction SilentlyContinue)) {
    Step "Serveurs MCP globaux"
    claude mcp add context7   -- npx -y `@upstash/context7-mcp`@latest 2>$null
    claude mcp add playwright -- npx `@playwright/mcp`@latest --browser chrome 2>$null
    claude mcp add blender    -- uvx blender-mcp 2>$null
    claude mcp add desktop-computer-use -- npx -y computer-use-mcp 2>$null
    if ($env:FIGMA_ACCESS_TOKEN) { claude mcp add figma --env FIGMA_ACCESS_TOKEN=$env:FIGMA_ACCESS_TOKEN -- npx -y figma-developer-mcp --stdio 2>$null }
    if ($env:FAL_KEY)            { claude mcp add fal-ai --env FAL_KEY=$env:FAL_KEY -- npx -y fal-ai-mcp-server 2>$null }
    if ($env:ROBLOX_OPEN_CLOUD_API_KEY) { claude mcp add robloxstudio --env ROBLOX_OPEN_CLOUD_API_KEY=$env:ROBLOX_OPEN_CLOUD_API_KEY -- npx -y robloxstudio-mcp`@latest 2>$null }
    Ok "MCP globaux enregistres (voir SETUP.md 6.2 pour unreal/windows-desktop/roblox-officiel)"
} else { Warn "MCP saute" }

# ---------------------------------------------------------------------------
Step "Termine"
Write-Host @"
Etapes MANUELLES restantes (voir SETUP.md) :
  - Roblox Studio : installer, lancer 1x, copier MCPPlugin.rbxmx, activer HTTP
  - Unreal 5.8    : Epic Launcher, cloner unreal-mcp, compiler plugin C++ (section 6.2)
  - windows-desktop : cloner LooperSalty/mcp-desktop-windows + pip install (section 6.2)
  - Obsidian      : creer le vault pour le hook de fin de session (section 8)
  - gh auth login ; verifier 'claude mcp list'
"@ -ForegroundColor Cyan
