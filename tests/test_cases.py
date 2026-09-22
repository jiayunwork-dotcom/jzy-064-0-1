"""工况档管理测试：建档、持久化、凭名字取回复算、自带算例。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app import cases as cases_mod
from app.cases import BUILTIN_CASES, CaseStore
from app.service import calculate_moments, resolve_input


@pytest.fixture
def store(tmp_path):
    s = CaseStore(tmp_path / "cases.json")
    return s


def test_builtin_cases_present_and_sound(store):
    names = {c["name"] for c in store.list_cases()}
    assert set(BUILTIN_CASES) <= names
    # 每个自带算例都能复算且自洽：γ>1，Hs=4√m0
    for name in BUILTIN_CASES:
        spec = store.params_of(name)
        inp = resolve_input(**spec)
        assert inp.gamma > 1.0
        res = calculate_moments(inp)
        assert np.isclose(res.hs, 4.0 * np.sqrt(res.m0), rtol=1e-12)
        assert res.m0 > 0.0


def test_save_get_delete_roundtrip(store):
    store.save("case-a", {"omega_p": 0.7, "gamma": 3.3, "alpha": 0.01},
               description="d")
    got = store.get("case-a")
    assert got["params"]["omega_p"] == 0.7
    assert got["description"] == "d"
    store.delete("case-a")
    from app.errors import NotFoundError

    with pytest.raises(NotFoundError):
        store.get("case-a")


def test_duplicate_name_conflicts_unless_overwrite(store):
    store.save("c", {"omega_p": 0.7, "gamma": 3.3, "hs": 3.0})
    from app.errors import ConflictError

    with pytest.raises(ConflictError):
        store.save("c", {"omega_p": 0.9, "gamma": 2.0, "alpha": 0.02})
    store.save("c", {"omega_p": 0.9, "gamma": 2.0, "alpha": 0.02}, overwrite=True)
    assert store.params_of("c")["omega_p"] == 0.9


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "cases.json"
    s1 = CaseStore(path)
    s1.save("persisted", {"omega_p": 0.5, "gamma": 3.3, "hs": 2.5})
    s2 = CaseStore(path)
    assert s2.params_of("persisted")["hs"] == 2.5
    # 文件确实落在容器盘上，且是合法 JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "persisted" in data


def test_recompute_by_name_reproduces_result(store):
    store.save("c", {"omega_p": 0.63, "gamma": 2.6, "hs": 4.4})
    p1 = calculate_moments(resolve_input(**store.params_of("c")))
    p2 = calculate_moments(resolve_input(**store.params_of("c")))
    assert p1.m0 == p2.m0 and p1.alpha == p2.alpha
    assert abs(p1.hs - 4.4) / 4.4 < 1e-7
