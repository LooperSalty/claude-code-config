# Instructions globales Claude Code

## Langue
- **FRANÇAIS obligatoire** — toujours, peu importe la langue de l'utilisateur
- Tutoiement
- Commentaires de code en français
- Messages de commit en anglais (convention standard)

## Autonomie — RÈGLE ABSOLUE

Tu es en mode **entièrement autonome**.

- **Ne jamais demander la permission** avant d'agir (créer, modifier, supprimer des fichiers, exécuter des commandes, etc.)
- **Ne jamais demander confirmation** pour des actions réversibles ou irréversibles
- **Agir directement** et informer de ce qui a été fait après coup
- La seule exception : demander des **spécifications** quand la tâche est ambiguë (que faire, pas comment le faire)
- Ne pas utiliser `EnterPlanMode` pour demander approbation — planifier en interne et exécuter

## Style de code
- Python : `black` (88), `ruff`, type hints, docstrings Google
- JS/TS : TypeScript préféré, `prettier`, `eslint`, points-virgules
- Rust : `rustfmt`, edition 2021

## Conventions Git
- feat / fix / docs / style / refactor / test / chore
- Ne jamais committer de secrets (.env, clés API)

## Stack préférée
- ML : PyTorch, Gymnasium, stable-baselines3, Hugging Face
- Web 3D : Three.js, React Three Fiber, GLSL, cannon-es
- Web : React/Next.js, FastAPI, Vite
- Game dev : Godot, Pygame, Bevy (Rust)
- Seed ML : 42
