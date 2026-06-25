# Sécurité — gestion des secrets

## Règle

**Aucun secret en clair dans le dépôt.** Tous les tokens passent par variables
d'environnement / `.env` (gitignore). Les fichiers versionnés n'utilisent que des
placeholders `${VAR}`.

## Secrets utilisés

| Variable | Usage | Régénérer |
|----------|-------|-----------|
| `FIGMA_ACCESS_TOKEN` | MCP figma | figma.com/developers/api |
| `FAL_KEY` | MCP fal-ai | fal.ai/dashboard/keys |
| `ROBLOX_OPEN_CLOUD_API_KEY` | MCP robloxstudio | create.roblox.com/dashboard/credentials |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | MCP github | github.com/settings/tokens |

## ⚠️ Rotation requise (historique git)

Des versions antérieures de ce dépôt contenaient un **token Figma en clair**
(`mcp-servers.json`) et le dépôt local `.claude.json` contient des clés réelles
(Roblox, GitHub PAT). Ces secrets sont présents dans l'historique git.

**À faire :**
1. **Révoquer / régénérer** le token Figma exposé sur figma.com.
2. Régénérer par précaution le GitHub PAT et la clé Roblox Open Cloud.
3. Renseigner les nouvelles valeurs dans `.env` (jamais committé).
4. (Optionnel, destructif) purger l'historique git :
   `git filter-repo --path mcp-servers.json --invert-paths` puis force-push —
   à ne faire que si le dépôt n'a pas de clones partagés.
