"""测试夹具：把工况档存储隔离到临时目录，并提供 HTTP TestClient。"""

from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """每个测试一个独立工况档目录，避免污染容器内默认数据文件。"""
    from app import api as api_mod
    from app import cases as cases_mod
    from app import main as main_mod
    from app.cases import CaseStore

    fresh = CaseStore(tmp_path / "cases.json")
    monkeypatch.setattr(cases_mod, "store", fresh)
    monkeypatch.setattr(api_mod, "store", fresh)
    monkeypatch.setattr(main_mod, "store", fresh)

    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as c:
        yield c
