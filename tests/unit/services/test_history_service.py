"""投稿履歴サービスのテスト。"""

import pytest

from postblog.infrastructure.storage.database import Database
from postblog.infrastructure.storage.history_repository import (
    HistoryRecord,
    HistoryRepository,
)
from postblog.services.history_service import HistoryService


@pytest.fixture()
def history_service() -> HistoryService:
    """テスト用の投稿履歴サービスフィクスチャ。"""
    db = Database(":memory:")
    db.initialize()
    repo = HistoryRepository(db)
    return HistoryService(repo)


class TestHistoryService:
    """HistoryServiceのテスト。"""

    def test_save_and_get(self, history_service: HistoryService) -> None:
        """保存と取得ができることを確認する。"""
        record = HistoryRecord(title="テスト", service_name="qiita")
        saved = history_service.save(record)

        found = history_service.get(saved.id)  # type: ignore[arg-type]

        assert found is not None
        assert found.title == "テスト"

    def test_get_nonexistent(self, history_service: HistoryService) -> None:
        """存在しないレコードでNoneが返されることを確認する。"""
        assert history_service.get(9999) is None

    def test_get_all(self, history_service: HistoryService) -> None:
        """全件取得ができることを確認する。"""
        history_service.save(HistoryRecord(title="記事1", service_name="qiita"))
        history_service.save(HistoryRecord(title="記事2", service_name="zenn"))

        records = history_service.get_all()
        assert len(records) == 2

    def test_delete(self, history_service: HistoryService) -> None:
        """削除ができることを確認する。"""
        saved = history_service.save(
            HistoryRecord(title="削除対象", service_name="qiita")
        )

        result = history_service.delete(saved.id)  # type: ignore[arg-type]

        assert result is True
        assert history_service.get(saved.id) is None  # type: ignore[arg-type]

    def test_delete_nonexistent(self, history_service: HistoryService) -> None:
        """存在しないレコードの削除でFalseが返されることを確認する。"""
        assert history_service.delete(9999) is False
