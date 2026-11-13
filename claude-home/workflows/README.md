# Atelier — workflows de création de projet

Sept workflows Claude Code (outil **Workflow**) qui construisent un projet à partir d'un brief ou d'un cahier des
charges : logiciel, application web, site, jeu. Ils appliquent les règles de `claude-home/rules/` et s'appuient sur
les agents et skills installés (plugins `ecc`, `eco-router`, `superpowers`, `agent-skills`, skills three.js, etc.).

## Utilisation

```
/atelier chemin/vers/cahier-des-charges.docx        # piloté phase par phase (recommandé)
/atelier-auto { "brief": "...", "profondeur": "standard" }   # tout d'une traite
/atelier-revue                                       # une phase seule, sur un projet déjà cadré
```

| Phase | Workflow | Produit | Agents (standard) |
|-------|----------|---------|-------------------|
| 1 | `atelier-cadrage` | `docs/atelier/BRIEF.md`, `SPEC.md`, `etat.json` | 4 analystes, rédacteur, critique |
| 2 | `atelier-architecture` | `ARCHITECTURE.md`, `docs/adr/` | 3 éclaireurs, 3 architectes, 3 jurés, rédacteur, vérificateur adverse |
| 3 | `atelier-plan` | `PLAN.md`, `plan.json` | planificateur, 3 critiques, validation mécanique |
| 4 | `atelier-build` | le code, une vague par appel | 1 agent par tâche (TDD), relecteur par tâche, contrôleur, dépanneur |
| 5 | `atelier-revue` | `REVUE.md`, corrections | 1 relecteur par lentille, contre-experts, correcteurs |
| 6 | `atelier-recette` | `RECETTE.md`, `README.md`, verdict | recetteurs (voie vivante et voie statique), correcteurs |

Chaque phase se termine par un commit conventionnel et la mise à jour de `docs/atelier/etat.json`, ce qui permet
de reprendre après une interruption.

## Réglages (arguments communs)

- `profondeur` : `eco` (prototype), `standard` (défaut), `max` (exhaustif : plus de jurés, double contre-expertise,
  revue par zone, second tour de revue).
- `agents` : remplace l'agent d'un rôle, par exemple `{ "ouvrier": { "model": "opus" } }` ou
  `{ "relecteur": { "agentType": "agent-skills:code-reviewer" } }`.
- `etat` : l'état renvoyé par la phase précédente (sinon relu sur disque).

Rôles par défaut : `eclaireur` → `eco-router:scout` (Haiku) ; `ouvrier` → Sonnet ; `expert` → modèle de session ;
`planificateur`, `architecte`, `relecteur`, `securite`, `depanneur`, `e2e`, `documentaliste` → agents `ecc:*`.
Si un type d'agent manque sur la machine, le script se replie sur un agent générique et le signale.

## Profils

`atelier-cadrage` classe le projet (`webapp`, `website`, `game`, `software`) et relève ses traits (`3d`, `donnees`,
`auth`, `api`, `ia`, `temps-reel`, `animation`, `mobile`, `python`, `rust`, `go`, `roblox`). Le profil qui en résulte
fixe les skills proposés aux agents, les lentilles de revue et les vérifications de recette. Pour ajouter un type ou
un trait : tables `PROFILS` et `TRAITS` dans `atelier-cadrage.js`, puis les clés correspondantes dans `LENTILLES`
(`atelier-revue.js`) et `RECETTES` (`atelier-recette.js`) — les tests refusent une clé orpheline.

## Modifier les scripts

Le moteur de workflows exige des scripts autonomes (pas d'import). Le code partagé vit donc dans
`tools/atelier/commun.fragment.js` et se recopie entre les balises `<atelier:commun>` :

```powershell
node tools/atelier/sync-commun.mjs                 # recopie le bloc commun dans les six scripts de phase
node --test tools/atelier/atelier.test.mjs         # 18 tests à blanc : agents simulés, aucun token dépensé
node tools/atelier/dryrun.mjs claude-home/workflows/atelier-plan.js '{"etat":{...}}'   # trace d'un script
Copy-Item claude-home\workflows\*.js $env:USERPROFILE\.claude\workflows\ -Force        # installation
```

Dans une session déjà ouverte, `/reload-skills` recharge les workflows modifiés.

Contraintes du moteur à respecter : `export const meta = {...}` en première instruction et littéral pur ; pas de
`Date.now()`, `Math.random()`, `new Date()` sans argument ; pas d'accès au système de fichiers depuis le script
(l'état passe par les agents et par `docs/atelier/`).
