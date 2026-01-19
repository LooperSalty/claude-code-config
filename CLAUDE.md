# Instructions Claude Code

> **IMPORTANT** : Toujours communiquer en français, peu importe la langue utilisée par l'utilisateur.

## Langue

- **Langue obligatoire : FRANÇAIS** (jamais en anglais)
- Utiliser le tutoiement
- Commentaires de code en français
- Messages de commit en anglais (convention standard)

## Style de code

### Python
- Formatter: `black` (ligne max: 88)
- Linter: `ruff`
- Type hints: toujours les utiliser
- Docstrings: style Google

### JavaScript / TypeScript
- Préférer **TypeScript** à JavaScript
- Formatter: `prettier`
- Linter: `eslint`
- Utiliser des points-virgules

### Rust
- Formatter: `rustfmt`
- Edition: 2021

## Conventions Git

Préfixes de commit:
- `feat:` - Nouvelle fonctionnalité
- `fix:` - Correction de bug
- `docs:` - Documentation
- `style:` - Formatage (pas de changement de code)
- `refactor:` - Refactoring
- `test:` - Ajout/modification de tests
- `chore:` - Maintenance

## Domaines d'expertise

### Machine Learning / IA
- PyTorch (préféré à TensorFlow)
- scikit-learn
- Gymnasium pour le RL
- stable-baselines3
- OpenCV pour la vision
- Transformers (Hugging Face)

### Web 3D
- Three.js
- React Three Fiber (@react-three/fiber)
- Babylon.js
- GLSL / WGSL pour les shaders
- cannon-es pour la physique

### Développement de jeux
- Godot
- Pygame
- Bevy (Rust)
- Pattern ECS (Entity Component System)

### Frameworks Web
- React / Next.js
- FastAPI / Flask (Python)
- Express / NestJS (Node.js)

## Bonnes pratiques

- Toujours créer un `.gitignore` approprié
- Ne jamais committer de secrets (.env, clés API, etc.)
- Préférer les tests unitaires
- Code propre et lisible > commentaires excessifs
- DRY (Don't Repeat Yourself)
- KISS (Keep It Simple, Stupid)

## Structure de projet préférée

### Projet ML
```
project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
├── tests/
├── configs/
├── requirements.txt
└── README.md
```

### Projet Web 3D
```
project/
├── src/
│   ├── components/
│   ├── scenes/
│   ├── shaders/
│   └── assets/
├── public/
├── package.json
└── tsconfig.json
```

## Notes

- Seed pour reproductibilité ML: 42
- Préférer CUDA si disponible, fallback CPU sinon
- Utiliser Vite pour les projets web modernes
