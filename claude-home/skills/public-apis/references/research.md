# Boîte à outils de recherche : API sans clé

Chaque modèle ci-dessous a été appelé et a répondu le 2026-10-25. `{q}` = requête encodée en URL (espace → `+` ou `%20`, accents en `%XX`).

Appel type (Bash) : `curl -s -A "claude-code-research/1.0" "<url>"`, puis extraire seulement les champs utiles :
`curl -s -A "claude-code-research/1.0" "<url>" | python -c "import sys,json; d=json.load(sys.stdin); print(...)"`
Toujours limiter le nombre de résultats (`limit`, `rows`, `per-page`, `srlimit`…) et citer l'URL appelée comme source.

## Savoir général et faits

| Besoin | URL | Champs utiles |
|---|---|---|
| Résumé d'un article Wikipédia | `https://{lang}.wikipedia.org/api/rest_v1/page/summary/{Titre_avec_underscores}` | `extract`, `description`, `content_urls` |
| Chercher dans Wikipédia | `https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit=5` | `query.search[].title`, `snippet` |
| Texte intégral d'un article | `https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={Titre}&format=json` | `query.pages.*.extract` (long : garder la partie utile) |
| Popularité d'un article (vues/jour) | `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{Titre}/daily/{AAAAMMJJ}/{AAAAMMJJ}` | `items[].views` |
| Entité Wikidata (id Q…) | `https://www.wikidata.org/w/api.php?action=wbsearchentities&search={q}&language=fr&format=json&limit=3` | `search[].id`, `description` |
| Fait précis, structuré (SPARQL) | `https://query.wikidata.org/sparql?format=json&query={SPARQL encodé}` (ex. population de Paris : `SELECT ?pop WHERE {wd:Q90 wdt:P1082 ?pop}`) | `results.bindings` |
| Définition anglaise | `https://api.dictionaryapi.dev/api/v2/entries/en/{mot}` ou `https://en.wiktionary.org/api/rest_v1/page/definition/{mot}` | `meanings[].definitions` |

## Science et publications

| Besoin | URL | Remarques |
|---|---|---|
| Articles toutes disciplines | `https://api.openalex.org/works?search={q}&per-page=5` (+ `&filter=publication_year:2024-2026` pour le récent) | Le meilleur point de départ. Garder le tri par pertinence : `sort=cited_by_count:desc` remonte des articles hors sujet. Sans clé, budget quotidien partagé |
| Prépublications (IA, physique, maths) | `https://export.arxiv.org/api/query?search_query=all:{q}&max_results=5` ou `?id_list=1706.03762` | Réponse XML (Atom) |
| Métadonnées d'un DOI (éditeurs) | `https://api.crossref.org/works/{doi}` ; recherche `https://api.crossref.org/works?query={q}&rows=5` | Un DOI arXiv (10.48550/…) est chez DataCite : `https://api.datacite.org/dois/{doi}` |
| Biomédical | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={q}&retmode=json&retmax=5`, puis `esummary.fcgi?db=pubmed&id={ids}&retmode=json` | Ou `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&pageSize=5` |
| Citations, auteurs | `https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=5&fields=title,year,citationCount,url` | Souvent 429 sans clé : prendre OpenAlex d'abord |
| Espèces vivantes | `https://api.gbif.org/v1/species/search?q={nom}&limit=3` ; observations `https://api.gbif.org/v1/occurrence/search?scientificName={nom}&country=FR&limit=5` | |
| Séismes récents | `https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&limit=10&orderby=time` (+ `&minmagnitude=5`) | |

## Développement et technique

| Besoin | URL | Remarques |
|---|---|---|
| Dépôts GitHub | `https://api.github.com/search/repositories?q={q}&sort=stars&per_page=5` | 60 requêtes/h sans jeton ; `gh api ...` si connecté |
| Dernière version d'un projet GitHub | `https://api.github.com/repos/{owner}/{repo}/releases/latest` | `tag_name`, `published_at` |
| Paquet npm | `https://registry.npmjs.org/{paquet}/latest` ; recherche `https://registry.npmjs.org/-/v1/search?text={q}&size=5` | `version`, `downloads` |
| Paquet PyPI | `https://pypi.org/pypi/{paquet}/json` | `info.version`, `releases` |
| Crate Rust | `https://crates.io/api/v1/crates?q={q}&per_page=5` | |
| Fin de support (Node, Python, Ubuntu…) | `https://endoflife.date/api/{produit}.json` ; liste `https://endoflife.date/api/all.json` | `cycle`, `eol`, `latest` |
| Failles connues d'une version | `curl -s -X POST https://api.osv.dev/v1/query -H 'Content-Type: application/json' -d '{"package":{"name":"lodash","ecosystem":"npm"},"version":"4.17.15"}'` | `vulns[].id`, `summary` |
| Modèles Hugging Face | `https://huggingface.co/api/models?search={q}&limit=5&sort=downloads` | |
| Questions Stack Overflow | `https://api.stackexchange.com/2.3/search/advanced?q={q}&site=stackoverflow&order=desc&sort=votes&pagesize=5` | |
| Discussions tech (Hacker News) | `https://hn.algolia.com/api/v1/search?query={q}&hitsPerPage=5` ; récentes : `search_by_date?query={q}&tags=story` | |

## Web, archives et données ouvertes

| Besoin | URL | Remarques |
|---|---|---|
| Version archivée d'une page | `https://archive.org/wayback/available?url={url}&timestamp={AAAAMMJJ}` | Historique : `https://web.archive.org/cdx/search/cdx?url={url}&limit=10&output=json` |
| Jeux de données publics FR | `https://www.data.gouv.fr/api/1/datasets/?q={q}&page_size=5` | |
| Indicateurs pays (PIB, population…) | `https://api.worldbank.org/v2/country/{FR;DE}/indicator/{NY.GDP.MKTP.CD ou SP.POP.TOTL}?format=json&date=2024` | Parfois lent (jusqu'à 20 s) |

## Géographie, France, météo

| Besoin | URL | Remarques |
|---|---|---|
| Coordonnées d'une ville | `https://geocoding-api.open-meteo.com/v1/search?name={ville}&count=1&language=fr` | |
| Météo actuelle et prévisions | `https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m&daily=temperature_2m_max,precipitation_sum` | |
| Météo passée | `https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=AAAA-MM-JJ&end_date=AAAA-MM-JJ&daily=temperature_2m_max` | |
| Lieu ou adresse dans le monde | `https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1` | User-Agent obligatoire, 1 requête/s maximum |
| Adresse en France (BAN) | `https://api-adresse.data.gouv.fr/search/?q={adresse}&limit=1` | |
| Commune française | `https://geo.api.gouv.fr/communes?nom={nom}&fields=nom,population,codesPostaux&limit=3` | |
| Entreprise française (SIREN) | `https://recherche-entreprises.api.gouv.fr/search?q={nom}&per_page=3` | |
| Jours fériés | `https://date.nager.at/api/v3/PublicHolidays/{année}/{FR}` | |

## Économie et finance

| Besoin | URL | Remarques |
|---|---|---|
| Taux de change (BCE) | `https://api.frankfurter.app/latest?from=EUR&to=USD,JPY` ; à une date : `https://api.frankfurter.app/2020-01-02?from=EUR&to=USD` | |
| Prix des cryptos | `https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=eur` | |

## Culture et consommation

| Besoin | URL | Remarques |
|---|---|---|
| Livres | `https://openlibrary.org/search.json?q={q}&limit=5` | |
| Musique (artistes, albums) | `https://musicbrainz.org/ws/2/artist/?query={q}&fmt=json&limit=3` | User-Agent obligatoire, 1 requête/s |
| Produit alimentaire (code-barres) | `https://world.openfoodfacts.org/api/v2/product/{EAN}.json?fields=product_name,nutriscore_grade` | |

## Écartées le 2026-10-25

- REST Countries : les versions gratuites (v3.1) sont abandonnées. Pour un pays, utiliser Wikidata ou la Banque mondiale.
- NASA APOD avec `DEMO_KEY` : erreur 500. Il faut une clé gratuite sur api.nasa.gov.
