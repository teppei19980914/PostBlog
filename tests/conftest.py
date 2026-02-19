"""pytest の共通設定とフィクスチャ."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir() -> Path:
    """テスト用の一時ディレクトリを提供するフィクスチャ。

    Returns:
        一時ディレクトリのパス。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
