"""
NBA Historical Player Stats Scraper
=====================================
Scrapes per-game, advanced, totals, per-36, and per-100-possessions stats
from Basketball Reference for every player across a range of seasons.

Requirements:
    pip install requests beautifulsoup4 pandas

Usage:
    python nba_scraper.py                        # 1980-2026 (default)
    python nba_scraper.py --start 2000 --end 2010
    python nba_scraper.py --output my_stats.csv
    python nba_scraper.py --delay 4              # seconds between requests

Output:
    nba_player_stats_1980_2026.csv  (or custom name)

Notes:
    - Basketball Reference rate-limits aggressively. Keep --delay >= 3.
    - Each season makes 5 requests (one per stat table), so for 46 seasons
      that's ~230 requests. At 3s delay expect ~12 minutes total.
    - The script resumes from where it left off if you re-run it
      (skips seasons already present in the output file).
"""

import argparse
import time
import re
import sys
from io import StringIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://www.basketball-reference.com"

# Each tuple: (table_id_on_page, short_label_for_column_prefix_if_needed)
# We pull five stat sheets and merge them on player + season.
STAT_TABLES = [
    ("per_game",        "per_game_stats"),      # points/rebounds/assists per game
    ("totals",          "totals_stats"),         # raw totals
    ("per_minute",      "per_minute_stats"),     # per 36 minutes
    ("per_poss",        "per_poss_stats"),       # per 100 possessions
    ("advanced",        "advanced_stats"),       # PER, WS, BPM, VORP, TS% …
]

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.basketball-reference.com/",
    "Connection": "keep-alive",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(SESSION_HEADERS)
    return s


def fetch_page(session: requests.Session, url: str, retries: int = 3, backoff: float = 10.0) -> BeautifulSoup | None:
    """Fetch a URL with retry/backoff. Returns BeautifulSoup or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 429:
                wait = backoff * attempt
                print(f"    [429] Rate limited. Waiting {wait}s …")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"    [HTTP {resp.status_code}] {url}")
                return None
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as exc:
            print(f"    [Error] {exc}. Attempt {attempt}/{retries}")
            time.sleep(backoff)
    return None


def extract_table(soup: BeautifulSoup, table_id: str) -> pd.DataFrame | None:
    """
    Find a <table id=table_id> in the soup (including inside HTML comments,
    which Basketball Reference uses for some secondary tables).
    Returns a cleaned DataFrame or None.
    """
    table = soup.find("table", {"id": table_id})

    # BBRef hides some tables inside HTML comments — unwrap them
    if table is None:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if table_id in comment:
                comment_soup = BeautifulSoup(comment, "html.parser")
                table = comment_soup.find("table", {"id": table_id})
                if table:
                    break

    if table is None:
        return None

    try:
        df = pd.read_html(StringIO(str(table)))[0]
    except Exception as exc:
        print(f"    [parse error] {table_id}: {exc}")
        return None

    # Drop repeated header rows that BBRef inserts every 20 rows
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"].copy()
        df.drop(columns=["Rk"], inplace=True, errors="ignore")

    # Drop rows with no player name
    if "Player" in df.columns:
        df = df[df["Player"].notna() & (df["Player"] != "")].copy()

    # Strip footnote asterisks (HOF indicator) from player names
    if "Player" in df.columns:
        df["Player"] = df["Player"].str.replace(r"\*$", "", regex=True)

    return df.reset_index(drop=True)


def scrape_season(session: requests.Session, season_end: int, delay: float) -> pd.DataFrame | None:
    """
    Scrape all stat tables for one NBA season (identified by the year the
    season ended, e.g. 2024 for the 2023-24 season).
    Returns a merged DataFrame with all available columns.
    """
    merged: pd.DataFrame | None = None

    for label, table_id in STAT_TABLES:
        url = f"{BASE_URL}/leagues/NBA_{season_end}_{label}.html"
        print(f"  Fetching {label:<12} → {url}")
        soup = fetch_page(session, url)

        if soup is None:
            print(f"    Skipping {label} (fetch failed)")
            time.sleep(delay)
            continue

        df = extract_table(soup, table_id)

        if df is None or df.empty:
            print(f"    Skipping {label} (table not found or empty)")
            time.sleep(delay)
            continue

        # Add season column
        df["Season"] = f"{season_end - 1}-{str(season_end)[-2:]}"

        # Prefix duplicate stat columns (keep Player/Season/Tm/Age/Pos unaffected)
        key_cols = {"Player", "Season", "Tm", "Age", "Pos", "G", "GS"}
        if merged is not None:
            # Rename columns that already exist in merged (except keys)
            rename = {}
            for col in df.columns:
                if col not in key_cols and col in merged.columns:
                    rename[col] = f"{label}_{col}"
            df.rename(columns=rename, inplace=True)

        if merged is None:
            merged = df
        else:
            # Merge on player + season + team (handles traded players with TOT rows)
            on_cols = [c for c in ["Player", "Season", "Tm", "Age", "Pos"] if c in merged.columns and c in df.columns]
            merged = pd.merge(merged, df, on=on_cols, how="outer")

        time.sleep(delay)

    return merged


def load_existing(path: Path) -> set:
    """Return set of seasons already saved so we can skip them on resume."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["Season"], nrows=10000)
        return set(df["Season"].dropna().unique())
    except Exception:
        return set()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape NBA player stats from Basketball Reference")
    parser.add_argument("--start",  type=int, default=1980, help="First season end-year (default 1980)")
    parser.add_argument("--end",    type=int, default=2026, help="Last season end-year  (default 2026)")
    parser.add_argument("--delay",  type=float, default=3.5, help="Seconds between requests (default 3.5)")
    parser.add_argument("--output", type=str, default="", help="Output CSV filename")
    args = parser.parse_args()

    start_year = args.start
    end_year   = args.end
    delay      = args.delay
    out_name   = args.output or f"nba_player_stats_{start_year}_{end_year}.csv"
    out_path   = Path(out_name)

    seasons = list(range(start_year, end_year + 1))
    total   = len(seasons)

    print(f"\n{'='*60}")
    print(f"  NBA Player Stats Scraper")
    print(f"  Seasons : {start_year} → {end_year}  ({total} seasons)")
    print(f"  Tables  : {', '.join(l for l, _ in STAT_TABLES)}")
    print(f"  Delay   : {delay}s between requests")
    print(f"  Output  : {out_path.resolve()}")
    print(f"{'='*60}\n")

    already_done = load_existing(out_path)
    if already_done:
        print(f"Resuming — {len(already_done)} seasons already in {out_path}\n")

    session    = make_session()
    all_frames = []
    skipped    = 0

    for i, year in enumerate(seasons, 1):
        season_label = f"{year-1}-{str(year)[-2:]}"

        if season_label in already_done:
            print(f"[{i:>3}/{total}] {season_label}  SKIP (already saved)")
            skipped += 1
            continue

        print(f"[{i:>3}/{total}] {season_label}")
        df = scrape_season(session, year, delay)

        if df is not None and not df.empty:
            all_frames.append(df)
            # Save incrementally so a crash doesn't lose everything
            combined = pd.concat(all_frames, ignore_index=True)
            mode   = "a" if out_path.exists() and skipped == 0 and i > 1 else "w"
            header = not (out_path.exists() and mode == "a")
            # Simplest: always rewrite from in-memory frames collected this run
            combined.to_csv(out_path, index=False)
            print(f"    ✓ {len(df)} player-rows saved  (running total: {len(combined)})")
        else:
            print(f"    ✗ No data returned for {season_label}")

        # Polite pause between seasons
        if i < total:
            time.sleep(delay)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_frames:
        final = pd.concat(all_frames, ignore_index=True)
        final.to_csv(out_path, index=False)
        print(f"  Done!  {len(final):,} rows × {len(final.columns)} columns")
        print(f"  Saved to: {out_path.resolve()}")
        print(f"\n  Columns ({len(final.columns)}):")
        for chunk_start in range(0, len(final.columns), 8):
            chunk = list(final.columns)[chunk_start:chunk_start+8]
            print("    " + "  ".join(chunk))
    else:
        print("  No data was collected. Check your network / BBRef access.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()