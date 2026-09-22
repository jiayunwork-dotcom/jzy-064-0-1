"""Case archive: named spectrum parameter sets persisted inside the container.

Cases are stored as a JSON document on the container filesystem (path from
the CASES_PATH environment variable). A fetch-limited demo case is seeded
on first start so anyone who launches the service can immediately verify
it against a known-consistent example. All file access is serialised with
a lock so parallel case submissions cannot corrupt or cross-contaminate
each other's entries.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

DEMO_CASE_NAME = "demo_fetch_limited"

# Built-in fetch-limited example: gamma > 1, peak at omega_p, and the
# inverted alpha makes Hs == 4*sqrt(m0) hold by construction.
DEMO_CASE = {
    "name": DEMO_CASE_NAME,
    "description": (
        "Fetch-limited JONSWAP example (gamma=3.3, Tp~12.57 s, Hs=4.0 m); "
        "alpha is inverted from the wave-height constraint."
    ),
    "params": {
        "omega_p": 0.5,
        "gamma": 3.3,
        "hs_target": 4.0,
        "wind_speed": 15.0,
        "fetch": 100_000.0,
    },
}


class CaseNotFoundError(KeyError):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"case {name!r} not found")


class CaseStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        with self._lock:
            self._cases = self._load()
            if DEMO_CASE_NAME not in self._cases:
                self._cases[DEMO_CASE_NAME] = DEMO_CASE
                self._save()

    def _load(self) -> dict:
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._cases, fh, indent=2, ensure_ascii=False)
        tmp.replace(self._path)

    def save_case(self, name: str, params: dict, description: str = "") -> dict:
        if not name or not name.strip():
            raise ValueError("case name must be a non-empty string")
        case = {"name": name, "description": description, "params": params}
        with self._lock:
            self._cases[name] = case
            self._save()
        return case

    def get_case(self, name: str) -> dict:
        with self._lock:
            try:
                return dict(self._cases[name])
            except KeyError:
                raise CaseNotFoundError(name) from None

    def list_cases(self) -> list[str]:
        with self._lock:
            return sorted(self._cases)

    def delete_case(self, name: str) -> None:
        with self._lock:
            if name not in self._cases:
                raise CaseNotFoundError(name)
            del self._cases[name]
            self._save()
