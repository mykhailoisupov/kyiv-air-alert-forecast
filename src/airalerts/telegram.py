"""Backfill a public Telegram channel through its web preview (no account needed).

Usage: python -m airalerts.telegram kpszsu --since 2026-04-27T00:00:00Z
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PREVIEW_URL = "https://t.me/s/{channel}"
HEADERS = {"User-Agent": "Mozilla/5.0 (research backfill; one request per second)"}


def parse_page(html: str) -> list[dict]:
    """Extract posts from one preview page, oldest first."""
    posts = []
    for message in BeautifulSoup(html, "html.parser").select("div.tgme_widget_message[data-post]"):
        stamp = message.select_one("a.tgme_widget_message_date time")
        if stamp is None:
            continue
        text = message.select_one("div.tgme_widget_message_text")
        meta = message.select_one("span.tgme_widget_message_meta")
        posts.append(
            {
                "id": int(message["data-post"].rsplit("/", 1)[1]),
                "published_utc": stamp["datetime"],
                # An edited post shows its current text, which may postdate publication.
                "edited": bool(meta and "edited" in meta.get_text()),
                "text": text.get_text(" ", strip=True) if text else "",
            }
        )
    return sorted(posts, key=lambda p: p["id"])


def load_posts(path: Path) -> pd.DataFrame:
    posts = pd.read_json(path, lines=True, dtype={"text": str})
    posts["published_utc"] = pd.to_datetime(posts["published_utc"], utc=True).dt.as_unit("s")
    return posts.drop_duplicates("id").sort_values("published_utc", kind="stable").reset_index(drop=True)


def scrape(channel: str, out_path: Path, since: pd.Timestamp, pause: float = 1.0) -> int:
    """Walk backwards from the newest post until `since`. Safe to interrupt and rerun."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    before = None
    if out_path.exists() and out_path.stat().st_size:
        done = load_posts(out_path)
        if done["published_utc"].min() < since:
            return len(done)
        before = int(done["id"].min())
    written = 0
    with out_path.open("a", encoding="utf-8") as out:
        while True:
            params = {"before": before} if before else None
            response = requests.get(PREVIEW_URL.format(channel=channel), params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            posts = parse_page(response.text)
            if not posts or (before and posts[0]["id"] >= before):
                break
            for post in posts:
                out.write(json.dumps(post, ensure_ascii=False) + "\n")
            out.flush()
            written += len(posts)
            before = posts[0]["id"]
            if pd.Timestamp(posts[0]["published_utc"]) < since:
                break
            time.sleep(pause)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channel")
    parser.add_argument("--since", required=True, help="UTC time to backfill to, e.g. 2026-04-27T00:00:00Z")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/telegram"))
    args = parser.parse_args()
    out_path = args.out_dir / f"{args.channel}.jsonl"
    written = scrape(args.channel, out_path, pd.Timestamp(args.since).tz_convert("UTC"))
    print(f"{written} posts written to {out_path}")


if __name__ == "__main__":
    main()
