"""投稿履歴リポジトリのテスト。"""

import pytest

from postblog.infrastructure.storage.database import Database
from postblog.infrastructure.storage.history_repository import (
    HistoryRecord,
    HistoryRepository,
)


@pytest.fixture()
def history_repo() -> HistoryRepository:
    """テスト用の投稿履歴リポジトリフィクスチャ。"""
    db = Database(":memory:")
    db.initialize()
    return HistoryRepository(db)


class TestHistoryRepository:
    """HistoryRepositoryのテスト。"""

    def test_save_record(self, history_repo: HistoryRepository) -> None:
        """投稿履歴が保存されることを確認する。"""
        record = HistoryRecord(
            title="テスト記事",
            body_preview="本文のプレビュー...",
            blog_type_id="tech",
            service_name="qiita",
            article_url="https://qiita.com/test/items/123",
            status="published",
        )
        saved = history_repo.save(record)

        assert saved.id is not None
        assert saved.id > 0

    def test_save_and_find_by_id(self, history_repo: HistoryRepository) -> None:
        """保存した投稿履歴がIDで検索できることを確認する。"""
        record = HistoryRecord(
            title="テスト記事",
            body_preview="プレビュー",
            blog_type_id="tech",
            service_name="qiita",
            article_url="https://qiita.com/test",
        )
        saved = history_repo.save(record)

        found = history_repo.find_by_id(saved.id)  # type: ignore[arg-type]

        assert found is not None
        assert found.title == "テスト記事"
        assert found.service_name == "qiita"
        assert found.article_url == "https://qiita.com/test"

    def test_find_by_id_nonexistent(self, history_repo: HistoryRepository) -> None:
        """存在しないIDで検索するとNoneが返されることを確認する。"""
        result = history_repo.find_by_id(9999)
        assert result is None

    def test_find_all(self, history_repo: HistoryRepository) -> None:
        """全ての投稿履歴が取得できることを確認する。"""
        history_repo.save(HistoryRecord(title="記事1", service_name="qiita"))
        history_repo.save(HistoryRecord(title="記事2", service_name="zenn"))
        history_repo.save(HistoryRecord(title="記事3", service_name="wordpress"))

        records = history_repo.find_all()

        assert len(records) == 3

    def test_find_all_empty(self, history_repo: HistoryRepository) -> None:
        """履歴がない場合に空リストが返されることを確認する。"""
        records = history_repo.find_all()
        assert records == []

    def test_delete(self, history_repo: HistoryRepository) -> None:
        """投稿履歴が削除されることを確認する。"""
        record = HistoryRecord(title="削除対象", service_name="qiita")
        saved = history_repo.save(record)

        result = history_repo.delete(saved.id)  # type: ignore[arg-type]

        assert result is True
        assert history_repo.find_by_id(saved.id) is None  # type: ignore[arg-type]

    def test_delete_nonexistent(self, history_repo: HistoryRepository) -> None:
        """存在しない履歴の削除でFalseが返されることを確認する。"""
        result = history_repo.delete(9999)
        assert result is False

    def test_save_failed_record(self, history_repo: HistoryRepository) -> None:
        """失敗した投稿の履歴が保存されることを確認する。"""
        record = HistoryRecord(
            title="失敗した記事",
            service_name="wordpress",
            status="failed",
        )
        saved = history_repo.save(record)
        found = history_repo.find_by_id(saved.id)  # type: ignore[arg-type]

        assert found is not None
        assert found.status == "failed"
        assert found.article_url is None
