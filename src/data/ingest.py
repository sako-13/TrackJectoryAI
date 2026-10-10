"""Data ingestion.

World Athletics does not publish an official public API. :class:`WorldAthleticsClient`
targets the unofficial JSON endpoints the worldathletics.org website itself uses; it is
best-effort and may stop working if the site changes. When the API is unreachable (or
disabled via ``data.source``), the pipeline falls back to
:func:`generate_synthetic_dataset`, which simulates realistic career trajectories with
the same schema so the full pipeline is reproducible offline.

Output schema (long format, one row per race):
    athlete_id, athlete_name, sex, birth_year, event, date, season, age,
    meet_name, meet_level (1=major international .. 5=local), position,
    performance (seconds), wind (100m only, m/s)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils import ensure_dir

# Valid performance bounds per event, in seconds.
EVENT_BOUNDS = {
    "100m": (9.40, 16.00),
    "800m": (99.00, 170.00),
}

# Realistic population parameters per event/sex.
# tier -> (mean peak season best [s], sd)
EVENT_PROFILES = {
    "100m": {
        "M": {"elite": (9.99, 0.13), "intl": (10.45, 0.22), "nat": (11.05, 0.30), "dev": (11.85, 0.45)},
        "F": {"elite": (10.85, 0.14), "intl": (11.25, 0.22), "nat": (11.85, 0.30), "dev": (12.60, 0.45)},
    },
    "800m": {
        "M": {"elite": (104.5, 0.9), "intl": (109.0, 1.4), "nat": (114.5, 2.0), "dev": (122.0, 3.0)},
        "F": {"elite": (113.5, 1.0), "intl": (118.5, 1.6), "nat": (124.0, 2.4), "dev": (132.0, 3.4)},
    },
}

TIERS = ["elite", "intl", "nat", "dev"]
TIER_PROBS = [0.04, 0.16, 0.35, 0.45]
# Probability of competing at meet level 1..5, per tier.
TIER_LEVEL_PROBS = {
    "elite": [0.35, 0.30, 0.20, 0.10, 0.05],
    "intl": [0.08, 0.22, 0.30, 0.25, 0.15],
    "nat": [0.01, 0.06, 0.25, 0.38, 0.30],
    "dev": [0.00, 0.02, 0.08, 0.35, 0.55],
}
MEET_NAMES = {
    1: ["Olympic Games", "World Championships", "Diamond League", "World Indoor Championships"],
    2: ["Continental Tour Gold", "National Championships", "Diamond League Meeting", "Continental Tour Silver"],
    3: ["National Grand Prix", "Regional Championships", "Continental Tour Bronze", "University Championships"],
    4: ["State Championships", "Club Meet", "Relay Carnival", "Invitational"],
    5: ["Local Track Meet", "School Championships", "Community Meet", "Time Trial"],
}

FIRST_NAMES = [
    "Usain", "Noah", "Fred", "Ferdinand", "Kishane", "Trayvon", "Christian", "Justin",
    "Marvin", "Ameer", "Omar", "Femi", "Akani", "Zharnel", "Andre", "Kenneth",
    "Dina", "Shericka", "Elaine", "Sha'Carri", "Marie-Josee", "Daryll", "Jenna",
    "Mujinga", "Dafne", "Tori", "English", "Aleia", "Twanisha", "Melissa",
    "Kara", "Athing", "Keely", "Caster", "Faith", "Hellen", "Vivian", "Tsegay",
    "Aminatou", "Genzebe", "Habitam", "Nelly", "Laura", "Keely", "Sifan", "Freweyni",
]
LAST_NAMES = [
    "Bolt", "Lyles", "Kerley", "Oblique", "Thompson", "Hall", "Coleman", "Gatlin",
    "Bromell", "Vicaut", "Blake", "Ogunode", "Simmonds", "Hughes", "De Grasse",
    "Bednarek", "Tucker", "Kambundji", "Fraser-Pryce", "Jackson", "Richardson",
    "Asher-Smith", "Schippers", "Bowie", "Terry", "Jefferson", "Muhammad",
    "Korchagina", "Wambui", "Jelimo", "Semenya", "Kipyegon", "Moraa", "Hodgkinson",
    "Reekie", "Muir", "Tual", "Sedjati", "Arop", "Wanyonyi", "Kibet", "Makwala",
    "van Niekerk", "James", "Norman", "Benjamin", "Deadmon", "Warholm",
]


class WAApiError(RuntimeError):
    """Raised when the unofficial World Athletics API cannot be used."""


class WorldAthleticsClient:
    """Best-effort client for the unofficial worldathletics.org JSON API."""

    BASE = "https://worldathletics.org/api"

    def __init__(self, timeout: float = 10.0, session: requests.Session | None = None):
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "athlete-career-forecast/0.1 (research project)",
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, params: dict | None = None):
        resp = self.session.get(f"{self.BASE}{path}", params=params, timeout=self.timeout)
        if resp.status_code != 200:
            raise WAApiError(f"WA API {path} -> HTTP {resp.status_code}")
        content_type = resp.headers.get("content-type", "")
        if "json" not in content_type:
            raise WAApiError(f"WA API {path} returned {content_type}, expected JSON")
        return resp.json()

    def athlete_profile(self, athlete_id: int) -> dict:
        return self._get(f"/athletes/{athlete_id}")

    def athlete_results(self, athlete_id: int, year_from: int | None = None) -> list:
        params = {"yearFrom": year_from} if year_from else None
        return self._get(f"/athletes/{athlete_id}/results", params)

    def search_athletes(self, query: str, limit: int = 20) -> list:
        payload = self._get("/athletes/search", {"query": query, "limit": limit})
        if isinstance(payload, dict):
            return payload.get("athletes", payload.get("results", []))
        return payload

    @classmethod
    def is_available(cls, timeout: float = 6.0) -> bool:
        """Probe the API with a known athlete profile request."""
        try:
            cls(timeout=timeout).athlete_profile(14163121)  # Usain Bolt
            return True
        except Exception:
            return False


def normalize_wa_results(payload, athlete_id: int, event: str = "100m") -> pd.DataFrame:
    """Normalize a WA API athlete-results payload into the race schema (best-effort)."""
    if isinstance(payload, dict):
        items = payload.get("results", payload.get("competitionResults", []))
    else:
        items = payload
    rows = []
    for item in items or []:
        perf = item.get("performance") or item.get("result") or item.get("time")
        date = item.get("date") or item.get("competitionDate")
        if perf is None or date is None:
            continue
        venue = item.get("venue") or item.get("competition") or {}
        if isinstance(venue, dict):
            meet_name = venue.get("name", "")
        else:
            meet_name = str(venue)
        rows.append(
            {
                "athlete_id": athlete_id,
                "athlete_name": item.get("athleteName", ""),
                "sex": item.get("sex", ""),
                "birth_year": None,
                "event": event,
                "date": str(date)[:10],
                "season": int(str(date)[:4]),
                "age": item.get("age"),
                "meet_name": meet_name,
                "meet_level": item.get("meetLevel", 3),
                "position": item.get("position"),
                "performance": perf,
                "wind": item.get("wind"),
            }
        )
    return pd.DataFrame(rows)


def generate_synthetic_dataset(
    event: str = "100m",
    n_athletes: int = 2000,
    seed: int = 42,
    current_year: int = 2025,
) -> pd.DataFrame:
    """Simulate realistic season-by-season race results for ``n_athletes`` athletes.

    Each athlete draws a latent talent tier, a peak age (mixture of early/mid/late),
    a first-season level, total career improvement, and a post-peak decline rate.
    Season bests are the minimum of noisy race times; meet levels correlate with tier.
    """
    if event not in EVENT_PROFILES:
        raise ValueError(f"Unsupported event '{event}'. Choose one of {list(EVENT_PROFILES)}")
    rng = np.random.default_rng(seed)
    profiles = EVENT_PROFILES[event]
    floor, ceiling = EVENT_BOUNDS[event]
    rows = []
    for athlete_id in range(1, n_athletes + 1):
        sex = "M" if rng.random() < 0.55 else "F"
        profile = profiles[sex]
        tier = rng.choice(TIERS, p=TIER_PROBS)
        tier_idx = TIERS.index(tier)
        mu, sd = profile[tier]
        peak_time = float(np.clip(rng.normal(mu, sd), floor + 0.05, ceiling - 0.5))

        # Peak age mixture -> early peakers / mid / late bloomers
        comp = rng.random()
        if comp < 0.25:
            peak_age = rng.normal(21.0, 1.0)
        elif comp < 0.75:
            peak_age = rng.normal(24.0, 1.4)
        else:
            peak_age = rng.normal(27.0, 1.4)
        peak_age = float(np.clip(peak_age, 19.5, 32.0))

        first_age = float(np.clip(rng.normal(16.8, 1.7), 13.0, 21.0))
        total_gain = float(np.clip(rng.normal(0.85, 0.35), 0.15, 2.6))
        decline = float(np.clip(rng.normal(0.030, 0.022), 0.004, 0.11))
        last_age = int(np.clip(rng.normal(26.5, 4.0), max(20, first_age + 3), 39))
        birth_year = current_year - int(first_age) - int(rng.integers(0, 2))
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        first_time = peak_time + total_gain

        for age in range(int(np.ceil(first_age)), last_age + 1):
            if rng.random() < 0.04:  # missed season (injury, study, ...)
                continue
            age_f = float(age + rng.uniform(0.05, 0.95))
            if age <= peak_age:
                frac = (age - first_age) / max(peak_age - first_age, 0.5)
                ability = first_time - total_gain * min(frac, 1.0)
            else:
                ability = peak_time + decline * (age - peak_age)

            mean_races = [9.0, 7.0, 5.0, 3.5][tier_idx]
            n_races = int(np.clip(rng.poisson(mean_races), 1, 22))
            race_times = ability + rng.exponential(0.055 + 0.02 * rng.random(), size=n_races)
            if rng.random() < 0.1:  # occasional bad race
                race_times[-1] += rng.uniform(0.1, 0.4)
            level = int(rng.choice(5, p=TIER_LEVEL_PROBS[tier]) + 1)
            month = int(rng.integers(3, 11))
            day = int(rng.integers(1, 29))
            wind = float(np.clip(rng.normal(0.6, 1.6), -2.5, 2.9)) if event == "100m" else float("nan")
            for r in range(n_races):
                race_day = min(day + 3 * r, 28)
                rows.append(
                    {
                        "athlete_id": athlete_id,
                        "athlete_name": name,
                        "sex": sex,
                        "birth_year": birth_year,
                        "event": event,
                        "date": f"{birth_year + age:04d}-{month:02d}-{race_day:02d}",
                        "season": birth_year + age,
                        "age": age_f,
                        "meet_name": str(rng.choice(MEET_NAMES[level])),
                        "meet_level": level,
                        "position": int(1 + rng.poisson(2.5)),
                        "performance": float(np.clip(race_times[r], floor, ceiling)),
                        "wind": wind,
                    }
                )
    return pd.DataFrame(rows)


def _fetch_from_api(event: str, n_athletes: int) -> pd.DataFrame:
    """Pull real results from the unofficial WA API (used only when it responds)."""
    client = WorldAthleticsClient()
    frames = []
    seen: set[int] = set()
    for name in ["Usain Bolt", "Noah Lyles", "Fred Kerley", "Elaine Thompson-Herah",
                 "Shericka Jackson", "Dina Asher-Smith", "Femke Bol", "Keely Hodgkinson"]:
        try:
            results = client.search_athletes(name, limit=10)
        except WAApiError:
            continue
        for entry in results[: n_athletes // 4]:
            aid = entry.get("athleteId") or entry.get("id")
            if not aid or aid in seen:
                continue
            seen.add(aid)
            try:
                payload = client.athlete_results(int(aid))
                frames.append(normalize_wa_results(payload, int(aid), event))
            except WAApiError:
                continue
            if len(seen) >= n_athletes:
                break
        if len(seen) >= n_athletes:
            break
    if not frames:
        raise WAApiError("No athlete results could be retrieved from the WA API")
    return pd.concat(frames, ignore_index=True)


def fetch_event_dataset(
    event: str = "100m",
    n_athletes: int = 2000,
    seed: int = 42,
    cache_dir: str | Path = "data/raw",
    source: str = "auto",
    current_year: int = 2025,
) -> tuple[pd.DataFrame, str]:
    """Return the race-results DataFrame and the source it came from.

    Priority: on-disk cache -> World Athletics API (if allowed and reachable)
    -> synthetic generator. The result is always written to the cache directory.
    """
    cache_dir = ensure_dir(cache_dir)
    cache_path = cache_dir / f"results_{event}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path), "cache"

    if source in ("auto", "api"):
        try:
            if WorldAthleticsClient.is_available():
                df = _fetch_from_api(event, n_athletes)
                df.to_csv(cache_path, index=False)
                return df, "world_athletics_api"
        except Exception:
            pass  # fall through to synthetic

    df = generate_synthetic_dataset(event, n_athletes, seed, current_year)
    df.to_csv(cache_path, index=False)
    return df, "synthetic"


def save_sample_dataset(path: str | Path, event: str = "100m", n_athletes: int = 40, seed: int = 7) -> Path:
    """Write a small deterministic sample used by tests and the demo."""
    df = generate_synthetic_dataset(event, n_athletes, seed)
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path
