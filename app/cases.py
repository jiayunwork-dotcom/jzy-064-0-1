"""工况档管理：按工况名字建档、持久化到容器内 JSON，日后凭名字取回复算。

- 只持久化“入参”（谱参数定义），结果在取回时重新计算，保证复算用的是
  当前服务里同一份常数与同一套算法。
- 线程锁仅保护字典与文件读写；谱计算不持锁，不同工况并行各算各的。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from .errors import ConflictError, NotFoundError
from .fetch import GAMMA_FETCH_DEFAULT

DATA_DIR = Path(os.environ.get("JONSWAP_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
STORE_FILE = DATA_DIR / "cases.json"

# 服务自带的一组有限风区算例（Hasselmann 1973 风区关系）。
# 谁拉起服务都能立刻拿它核对：γ>1、峰落在 ωp 处、Hs 与 4√m0 对得上。
BUILTIN_CASES: dict[str, dict[str, Any]] = {
    "fetch-coastal-15ms-25km": {
        "description": "沿岸风区：U=15 m/s，F=25 km",
        "params": {"wind_speed": 15.0, "fetch": 25_000.0, "gamma": GAMMA_FETCH_DEFAULT},
        "builtin": True,
    },
    "fetch-sea-15ms-50km": {
        "description": "有限风区海域：U=15 m/s，F=50 km",
        "params": {"wind_speed": 15.0, "fetch": 50_000.0, "gamma": GAMMA_FETCH_DEFAULT},
        "builtin": True,
    },
    "fetch-storm-25ms-200km": {
        "description": "大风风区：U=25 m/s，F=200 km",
        "params": {"wind_speed": 25.0, "fetch": 200_000.0, "gamma": GAMMA_FETCH_DEFAULT},
        "builtin": True,
    },
}


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """剔除 None，只留定义工况所需字段。"""
    allowed = {"omega_p", "gamma", "alpha", "hs", "wind_speed", "fetch"}
    out = {k: v for k, v in params.items() if k in allowed and v is not None}
    return out


class CaseStore:
    def __init__(self, path: Path = STORE_FILE) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._cases: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        with self._lock:
            if self._path.exists():
                try:
                    with self._path.open("r", encoding="utf-8") as fh:
                        self._cases = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    self._cases = {}
            else:
                self._cases = {name: dict(spec) for name, spec in BUILTIN_CASES.items()}
                self._persist_locked()

    def _persist_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # 原子替换，避免并发写到一半损坏工况档
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._cases, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------- queries
    def list_cases(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": name,
                    "description": spec.get("description"),
                    "builtin": bool(spec.get("builtin", False)),
                    "params": _normalize_params(spec.get("params", {})),
                }
                for name, spec in sorted(self._cases.items())
            ]

    def get(self, name: str) -> dict[str, Any]:
        with self._lock:
            if name not in self._cases:
                raise NotFoundError(f"工况 '{name}' 不存在", field="name")
            spec = self._cases[name]
            return {
                "name": name,
                "description": spec.get("description"),
                "builtin": bool(spec.get("builtin", False)),
                "params": _normalize_params(spec.get("params", {})),
            }

    def params_of(self, name: str) -> dict[str, Any]:
        return self.get(name)["params"]

    # ------------------------------------------------------------ mutation
    def save(
        self,
        name: str,
        params: dict[str, Any],
        *,
        description: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        clean = _normalize_params(params)
        if not clean:
            from .errors import ValidationError

            raise ValidationError("工况参数为空，无法建档", field="params")
        with self._lock:
            if name in self._cases and not overwrite:
                raise ConflictError(f"工况 '{name}' 已存在，如需覆盖请显式声明", field="name")
            self._cases[name] = {
                "params": clean,
                "description": description,
                "builtin": False,
            }
            self._persist_locked()
        return self.get(name)

    def delete(self, name: str) -> None:
        with self._lock:
            if name not in self._cases:
                raise NotFoundError(f"工况 '{name}' 不存在", field="name")
            del self._cases[name]
            self._persist_locked()

    def reset_builtins(self) -> None:
        """重建自带算例（测试 / 运维用）。"""
        with self._lock:
            for name, spec in BUILTIN_CASES.items():
                self._cases.setdefault(name, dict(spec))
            self._persist_locked()


# 进程内单例；锁保证并发提交工况档操作互不干扰
store = CaseStore()
