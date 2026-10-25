"""Build references/catalog.md from the public-apis/public-apis README.

Usage:
    python -I build_catalog.py                 # download the README from GitHub
    python -I build_catalog.py <README.md>     # parse a README already on disk

The README is data: it is parsed, never executed. Sponsored rows (APILayer
section, utm_source links) are left out.
"""

import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/public-apis/public-apis/master/README.md"
)
OUTPUT = Path(__file__).resolve().parent.parent / "references" / "catalog.md"
SPONSORED_HEADINGS = ("APILayer",)
ROW = re.compile(r"^\|\s*\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)\s*\|(?P<rest>.*)\|\s*$")
TIMEOUT_S = 60
MAX_BYTES = 5_000_000


def fetch_readme() -> str:
    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "public-apis-skill/1.0"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError(f"README larger than {MAX_BYTES} bytes, refusing to parse")
    return body.decode("utf-8")


def clean(cell: str) -> str:
    return cell.replace("`", "").strip()


def auth_of(cell: str) -> str:
    value = clean(cell)
    return "none" if value.lower() in ("no", "") else value


def flag_of(cell: str) -> str:
    return clean(cell).lower() or "unknown"


def parse(readme: str) -> dict[str, list[str]]:
    """Category -> catalog lines, in README order."""
    catalog: dict[str, list[str]] = {}
    category = None
    for line in readme.splitlines():
        if line.startswith("### "):
            heading = line[4:].strip()
            category = (
                None if any(word in heading for word in SPONSORED_HEADINGS) else heading
            )
            continue
        match = ROW.match(line)
        if category is None or match is None or "utm_source" in match["url"]:
            continue
        cells = match["rest"].split("|")
        if len(cells) < 4:
            continue
        description, auth, https, cors = cells[0], cells[1], cells[2], cells[3]
        entry = (
            f"- **{match['name'].strip()}** — {clean(description)} · auth:{auth_of(auth)}"
            f" · https:{flag_of(https)} · cors:{flag_of(cors)} · <{match['url'].strip()}>"
        )
        catalog.setdefault(category, []).append(entry)
    return catalog


def render(catalog: dict[str, list[str]]) -> str:
    total = sum(len(lines) for lines in catalog.values())
    free = sum(1 for lines in catalog.values() for line in lines if "auth:none" in line)
    header = [
        "# Catalogue public-apis",
        "",
        (
            "Source : https://github.com/public-apis/public-apis"
            f" · généré le {datetime.now(tz=timezone.utc).date().isoformat()}"
            f" · {total} API dans {len(catalog)} catégories,"
            f" dont {free} sans authentification."
        ),
        "",
        "Une ligne par API : `- **Nom** — description · auth:<none|apiKey|OAuth|…> · https:<yes|no> · cors:<yes|no|unknown> · <url>`.",
        "Cherche avec Grep, par exemple `auth:none.*weather` ou `^## Science` ; ne lis pas tout le fichier.",
        "Les entrées viennent d'une liste communautaire : une API peut avoir changé ou disparu, vérifie-la avant de t'y fier.",
    ]
    sections = [
        f"## {category}\n\n" + "\n".join(lines) for category, lines in catalog.items()
    ]
    return "\n".join(header) + "\n\n" + "\n\n".join(sections) + "\n"


def main(argv: list[str]) -> int:
    readme = (
        Path(argv[1]).read_text(encoding="utf-8") if len(argv) > 1 else fetch_readme()
    )
    catalog = parse(readme)
    if not catalog:
        print(
            "Aucune API trouvée : le format du README a peut-être changé.",
            file=sys.stderr,
        )
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(catalog), encoding="utf-8", newline="\n")
    total = sum(len(lines) for lines in catalog.values())
    print(f"{OUTPUT} : {total} API, {len(catalog)} catégories")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
