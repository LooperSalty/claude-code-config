---
description: Atelier — crée un projet de A à Z (logiciel, app web, site, jeu) avec des agents IA — cadrage → architecture → plan → build par vagues → revue → recette
argument-hint: <brief ou chemin du cahier des charges> [type=webapp|website|game|software] [profondeur=eco|standard|max]
---

# Atelier — pilotage d'un projet par workflows d'agents

Demande : $ARGUMENTS

Tu pilotes la création d'un projet avec les six workflows `atelier-*` (outil **Workflow**). Cette commande vaut
autorisation d'utiliser l'outil Workflow. Tu restes dans la boucle entre chaque phase : tu lis le résultat, tu
décides, tu relances. Aucune question à l'utilisateur sauf brief introuvable : les ambiguïtés deviennent des
hypothèses écrites dans la SPEC.

## Avant de lancer

1. Le répertoire courant est la racine du projet. Repère le brief (texte donné, ou fichier : docx, pdf, md).
2. Relève en une commande ce que les agents ne peuvent pas deviner : versions de node / python, docker utilisable
   ou non, ports pris. Passe ces faits dans `env`.
3. Choisis la profondeur : `standard` par défaut ; `max` si ultracode est actif ou si l'utilisateur veut de
   l'exhaustif ; `eco` pour un prototype jetable.

## Les six phases

Passe à chaque phase l'objet `etat` renvoyé par la précédente (sinon elle le relit dans `docs/atelier/etat.json`).

| # | Workflow | Arguments | Ce que tu vérifies avant de continuer |
|---|----------|-----------|----------------------------------------|
| 1 | `atelier-cadrage` | `{ brief \| briefPath, type?, env?, profondeur }` | `etat.type` et `etat.traits` cohérents avec le brief ; lis `hypotheses` et `questionsOuvertes` |
| 2 | `atelier-architecture` | `{ etat, contraintes?, profondeur }` | `verification` sans problème bloquant ou majeur ; `etat.commandes` complètes |
| 3 | `atelier-plan` | `{ etat, profondeur }` | `erreurs` vide (validation mécanique) ; sinon relance le plan |
| 4 | `atelier-build` | `{ etat, plan, profondeur }` — une vague par appel | `ok: true` ; boucle tant que `restantes` n'est pas vide |
| 5 | `atelier-revue` | `{ etat, profondeur }` | `restants` : aucun défaut critique ou haut non corrigé |
| 6 | `atelier-recette` | `{ etat, profondeur }` | `verdict` : SHIP, NEEDS WORK ou BLOCKED |

Règles de pilotage :

- **Build rouge** (`ok: false`) : lis `vagues[].controle.echecs`, diagnostique (skill `superpowers:systematic-debugging`),
  corrige toi-même ou relance la vague avec `vagues: [id]`. Ne passe jamais à la vague suivante sur du rouge.
- **Plan invalide** : ne lance pas le build tant que `erreurs` n'est pas vide.
- **Recette NEEDS WORK** : traite les échecs listés, puis relance `atelier-recette` avec
  `verifications: [les clés en échec]`.
- **Reprise après interruption** : relance la phase sans `etat` ; elle repart de `docs/atelier/etat.json` et
  `plan.json`. `atelier-build` reprend à la première vague non livrée.
- Workflows introuvables par leur nom (installation faite pendant la session) : passe
  `scriptPath: ~/.claude/workflows/atelier-<phase>.js`.
- Tout d'une traite, sans point de contrôle : `atelier-auto` avec `{ brief, profondeur }`.

## Ce que les workflows appliquent déjà

Les règles de `~/.claude/rules/` sont câblées dans les scripts : planner / architect / code-reviewer /
security-reviewer / build-error-resolver / e2e-runner / doc-updater (plugin ecc), TDD strict par tâche, relecture
immédiate du code écrit, jury multi-angles pour l'architecture, contrôles de sécurité avant chaque commit,
commits conventionnels, modèles par palier (Haiku pour la veille, Sonnet pour la routine, modèle de session pour
le reste). Les skills du profil (three.js, frontend-design, postgres, etc.) sont proposés aux agents selon le type
de projet et ses traits.

## Compte rendu final

Termine par un rapport court : verdict, ce qui est livré (phases, vagues, commits), résultats des tests et de la
recette, défauts restants, hypothèses prises et questions ouvertes pour le client, commande pour lancer le projet.
