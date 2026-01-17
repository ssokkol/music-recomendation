from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class TrackRow:
    track_id: int
    title: str
    artist: str
    tempo: float
    energy: float
    danceability: float
    valence: float
    favorite: int


def load_tracks(csv_path: str | Path) -> List[TrackRow]:
    rows: List[TrackRow] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                TrackRow(
                    track_id=int(row["track_id"]),
                    title=row["title"],
                    artist=row["artist"],
                    tempo=float(row["tempo"]),
                    energy=float(row["energy"]),
                    danceability=float(row["danceability"]),
                    valence=float(row["valence"]),
                    favorite=int(row["favorite"]),
                )
            )
    return rows


def save_recommendations(rows: Iterable[Tuple[int, str, str, float]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "title", "artist", "score"])
        for track_id, title, artist, score in rows:
            writer.writerow([track_id, title, artist, f"{score:.4f}"])


def fetch_spotify_tracks() -> List[TrackRow]:
    # Placeholder for Spotify Web API integration.
    # Replace with API calls and map Spotify features to TrackRow fields.
    return []
