#!/usr/bin/env python3
"""Refresh Kodkod's reviewed OpenCode catalog. Inspect the diff before shipping."""
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, text):
        if self.cell is not None:
            self.cell += text

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.row is not None:
            self.row.append((self.cell or "").strip())
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def fetch(url):
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def main():
    metadata = json.loads(fetch("https://models.dev/api.json"))
    models = []
    for service, key in [("go", "opencode-go"), ("zen", "opencode")]:
        table = Tables()
        table.feed(fetch(f"https://opencode.ai/docs/{service}/"))
        for row in table.rows:
            if len(row) != 4 or not row[2].startswith("https://opencode.ai/"):
                continue
            protocol = {
                "responses": "responses",
                "completions": "chat_completions",
                "messages": "messages",
            }.get(row[2].split("/")[-1])
            if protocol is None:
                continue
            model = metadata[key]["models"].get(row[1], {})
            models.append({
                "service": service,
                "id": row[1],
                "name": row[0],
                "protocol": protocol,
                "vision": "image" in model.get("modalities", {}).get("input", []),
            })
    if not all(any(m["service"] == service for m in models)
               for service in ("go", "zen")):
        raise ValueError("Provider endpoint tables were missing; catalog was not changed")
    path = Path(__file__).resolve().parents[1] / "kodkod-providers/src/open-code-models.json"
    path.write_text(json.dumps(models, indent=2) + "\n")


if __name__ == "__main__":
    main()
