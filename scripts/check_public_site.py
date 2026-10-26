"""Check the public entry points without network access or extra dependencies."""
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.refs = []
        self.videos = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.errors.append(f"duplicate id: {attrs['id']}")
            self.ids.add(attrs["id"])
        for key in ("href", "src", "poster"):
            if key in attrs:
                self.refs.append(attrs[key] or "")
        if tag == "video":
            self.videos.append(attrs)


def check_ref(source, ref):
    if not ref or ref == "+" or re.search(r"\bREPO\b", ref):
        return f"{source.name}: invalid reference {ref!r}"
    url = urlsplit(ref)
    if url.scheme or url.netloc:
        return None
    target = (source.parent / unquote(url.path)).resolve() if url.path else source
    if not target.is_relative_to(ROOT) or not target.exists():
        return f"{source.relative_to(ROOT)}: missing local target {ref}"
    if target.suffix == ".html" and url.fragment:
        doc = Page()
        doc.feed(target.read_text(encoding="utf-8"))
        if unquote(url.fragment) not in doc.ids:
            return f"{source.name}: missing fragment {ref}"
    return None


def main():
    source = ROOT / "index.html"
    page = Page()
    page.feed(source.read_text(encoding="utf-8"))
    errors = list(page.errors)
    errors += [error for ref in page.refs if (error := check_ref(source, ref))]
    for video in page.videos:
        if video.get("aria-labelledby") not in page.ids:
            errors.append(f"unlabelled video: {video.get('src')}")
        if "controls" not in video:
            errors.append(f"video has no controls: {video.get('src')}")
    for name in ("README.md", "docs/DEVELOPMENT.md", "docs/TECHNICAL_REPORT.md",
                 "docs/JUDGING.md", "docs/AUDIT.md", "docs/aws/DEPLOY.md",
                 "docs/Floorsight-Project-Overview.md"):
        source = ROOT / name
        if not source.is_file():
            errors.append(f"missing document: {name}")
            continue
        for ref in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", source.read_text(encoding="utf-8")):
            if error := check_ref(source, ref):
                errors.append(error)
    captions = ROOT / "videos/floorsight-demo.vtt"
    if not captions.is_file() or not captions.read_text(encoding="utf-8").startswith("WEBVTT\n"):
        errors.append("missing or invalid WebVTT captions")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Public site: {len(page.videos)} labelled videos; local assets, links and entry documents OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
