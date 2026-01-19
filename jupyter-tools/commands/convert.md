# Convertir un Notebook

Convertit un notebook Jupyter vers différents formats.

## Commandes disponibles

### Vers Python script
```bash
jupyter nbconvert --to script notebook.ipynb
```

### Vers HTML
```bash
jupyter nbconvert --to html notebook.ipynb
```

### Vers PDF (nécessite LaTeX)
```bash
jupyter nbconvert --to pdf notebook.ipynb
```

### Vers Markdown
```bash
jupyter nbconvert --to markdown notebook.ipynb
```

### Script Python propre (sans numéros de cellule)
```bash
jupyter nbconvert --to script --no-prompt notebook.ipynb
```

## Conversion batch

```bash
# Tous les notebooks du dossier
jupyter nbconvert --to html *.ipynb

# Récursif
find . -name "*.ipynb" -exec jupyter nbconvert --to script {} \;
```

## Exécuter et convertir

```bash
# Exécute le notebook puis convertit en HTML
jupyter nbconvert --execute --to html notebook.ipynb
```

Quel format de conversion veux-tu utiliser ?
