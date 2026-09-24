"""Downloads CC0 artwork (puzzle motifs) and CC0 table textures into tools/synth/sources/.

    python -m tools.synth.fetch_sources [--motifs 60] [--textures 25] [--contact you@example.com]

NOT run automatically: downloading needs the user's approval. Until it has run, the generator
uses procedural motifs and backgrounds only (motifs.py, backgrounds.py pick these files up
automatically once they exist).

Why these sources:
- Art Institute of Chicago (api.artic.edu): artworks flagged is_public_domain are released under
  CC0, and the IIIF server delivers them at 3000 px. Real paintings and prints bring what
  procedural motifs cannot: brush texture, real composition, and real colour statistics.
  A mix of landscapes, cityscapes, still lifes and busy scenes is searched for, because those
  are what jigsaw puzzles are made of.
- Poly Haven (api.polyhaven.com): CC0 PBR textures; the 2k diffuse map of wood, fabric and a few
  plain surfaces makes a realistic table.

Polite by design: a descriptive User-Agent (AIC asks for one), at least REQUEST_INTERVAL_S
between requests, retries with backoff, and resumability — files already on disk are skipped and
sources/manifest.json (provenance and licence per file) is rewritten after every download.
Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SOURCES = Path(__file__).resolve().parent / "sources"
MOTIF_DIR = SOURCES / "motifs"
TEXTURE_DIR = SOURCES / "textures"
MANIFEST = SOURCES / "manifest.json"

AIC_SEARCH = "https://api.artic.edu/api/v1/artworks/search"
AIC_IIIF = "https://www.artic.edu/iiif/2/{image_id}/full/{width},/0/default.jpg"
POLYHAVEN_ASSETS = "https://api.polyhaven.com/assets"
POLYHAVEN_FILES = "https://api.polyhaven.com/files/{slug}"

REQUEST_INTERVAL_S = 1.0     # polite rate limit (AIC documents 60 requests/min for anonymous use)
RETRIES = 4
TIMEOUT_S = 60
MIN_IMAGE_WIDTH = 1500       # smaller originals would be upsampled into blurry motifs
ASPECT_RANGE = (0.55, 1.9)   # puzzle-shaped pictures only (no scrolls or panoramas)

# (search term, how many to take). Guessed mix of what puzzles are printed with.
AIC_QUERIES = [
    ("landscape", 14), ("cityscape", 8), ("street", 5), ("still life", 9), ("harbor boats", 5),
    ("market", 4), ("festival crowd", 4), ("garden flowers", 5), ("village", 4), ("ukiyo-e print", 5),
]
AIC_TYPES = ("Painting", "Print")
# Poly Haven category -> DATASET.md background label, and how many to take.
POLYHAVEN_CATEGORIES = [("wood", "wood", 12), ("fabric", "cloth", 9), ("plaster", "plain", 2), ("concrete", "plain", 2)]


class Fetcher:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._last = 0.0

    def _wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < REQUEST_INTERVAL_S:
            time.sleep(REQUEST_INTERVAL_S - dt)
        self._last = time.monotonic()

    def get(self, url: str) -> bytes:
        err: Exception | None = None
        for attempt in range(RETRIES):
            self._wait()
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "AIC-User-Agent": self.user_agent})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    raise
                err = e
            except (urllib.error.URLError, TimeoutError) as e:
                err = e
            time.sleep(2.0 ** attempt)
        raise RuntimeError(f"giving up on {url}: {err}")

    def json(self, url: str) -> dict:
        return json.loads(self.get(url).decode("utf-8"))


def _load_manifest() -> dict:
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"motifs": [], "textures": []}


def _save_manifest(m: dict) -> None:
    SOURCES.mkdir(parents=True, exist_ok=True)
    m["motifs"].sort(key=lambda x: x["file"])
    m["textures"].sort(key=lambda x: x["file"])
    MANIFEST.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def fetch_motifs(f: Fetcher, target: int, manifest: dict) -> None:
    MOTIF_DIR.mkdir(parents=True, exist_ok=True)
    have = {m["id"] for m in manifest["motifs"]}
    total = len(have)
    for term, want in AIC_QUERIES:
        if total >= target:
            break
        got = 0
        page = 1
        while got < want and total < target and page <= 5:
            params = {
                "q": term, "limit": 50, "page": page,
                "fields": "id,title,image_id,artist_display,date_display,artwork_type_title,is_public_domain,thumbnail",
                "query[term][is_public_domain]": "true",
            }
            res = f.json(AIC_SEARCH + "?" + urllib.parse.urlencode(params))
            items = res.get("data", [])
            if not items:
                break
            for it in items:
                if got >= want or total >= target:
                    break
                aid = f"aic:{it['id']}"
                thumb = it.get("thumbnail") or {}
                w, h = thumb.get("width") or 0, thumb.get("height") or 0
                if (aid in have or not it.get("is_public_domain") or not it.get("image_id")
                        or it.get("artwork_type_title") not in AIC_TYPES or w < MIN_IMAGE_WIDTH or h == 0
                        or not (ASPECT_RANGE[0] <= w / h <= ASPECT_RANGE[1])):
                    continue
                name = f"aic_{it['id']}.jpg"
                path = MOTIF_DIR / name
                url = None
                if not path.is_file():
                    for width in (3000, 1686, 843):  # the IIIF server caps some works below 3000
                        url = AIC_IIIF.format(image_id=it["image_id"], width=min(width, w))
                        try:
                            path.write_bytes(f.get(url))
                            break
                        except urllib.error.HTTPError:
                            url = None
                    if url is None:
                        continue
                manifest["motifs"].append({
                    "file": name, "id": aid, "title": it.get("title"), "artist": it.get("artist_display"),
                    "date": it.get("date_display"), "type": it.get("artwork_type_title"), "query": term,
                    "source": f"https://www.artic.edu/artworks/{it['id']}", "image": url or "(already on disk)",
                    "licence": "CC0 1.0 (Art Institute of Chicago public-domain artwork)",
                })
                have.add(aid)
                got += 1
                total += 1
                _save_manifest(manifest)
                print(f"  motif {total:3d}/{target}: {name}  {it.get('title', '')[:60]}")
            page += 1


def fetch_textures(f: Fetcher, target: int, manifest: dict) -> None:
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    have = {t["id"] for t in manifest["textures"]}
    total = len(have)
    for category, label, want in POLYHAVEN_CATEGORIES:
        if total >= target:
            break
        assets = f.json(POLYHAVEN_ASSETS + "?" + urllib.parse.urlencode({"t": "textures", "c": category}))
        got = 0
        for slug in sorted(assets):
            if got >= want or total >= target:
                break
            tid = f"polyhaven:{slug}"
            if tid in have:
                continue
            files = f.json(POLYHAVEN_FILES.format(slug=slug))
            diffuse = files.get("Diffuse") or files.get("diffuse") or {}
            entry = (diffuse.get("2k") or {}).get("jpg")
            if not entry:
                continue
            name = f"ph_{slug}.jpg"
            path = TEXTURE_DIR / name
            if not path.is_file():
                path.write_bytes(f.get(entry["url"]))
            manifest["textures"].append({
                "file": name, "id": tid, "background": label, "category": category,
                "title": assets[slug].get("name"), "source": f"https://polyhaven.com/a/{slug}", "image": entry["url"],
                "licence": "CC0 1.0 (Poly Haven)",
            })
            have.add(tid)
            got += 1
            total += 1
            _save_manifest(manifest)
            print(f"  texture {total:3d}/{target}: {name} ({label})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.synth.fetch_sources", description=__doc__.split("\n\n")[0])
    ap.add_argument("--motifs", type=int, default=60)
    ap.add_argument("--textures", type=int, default=25)
    ap.add_argument("--contact", default="", help="optional contact (e.g. an e-mail) added to the User-Agent")
    args = ap.parse_args(argv)
    ua = "PieceFinder-synth/0.1 (non-commercial jigsaw-recognition research; CC0 sources only"
    ua += f"; {args.contact})" if args.contact else ")"
    f = Fetcher(ua)
    manifest = _load_manifest()
    if args.motifs:
        print("Art Institute of Chicago (CC0 artworks):")
        fetch_motifs(f, args.motifs, manifest)
    if args.textures:
        print("Poly Haven (CC0 textures):")
        fetch_textures(f, args.textures, manifest)
    _save_manifest(manifest)
    print(f"done: {len(manifest['motifs'])} motifs, {len(manifest['textures'])} textures in {SOURCES}")


if __name__ == "__main__":
    main()
