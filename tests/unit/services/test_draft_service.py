"""下書きサービスのテスト。"""

import pytest

from postblog.infrastructure.storage.database import Database
from postblog.infrastructure.storage.draft_repository import DraftRepository
from postblog.models.draft import Draft
from postblog.services.draft_service import DraftService


@pytest.fixture()
def draft_service() -> DraftService:
    """テスト用の下書きサービスフィクスチャ。"""
    db = Database(":memory:")
    db.initialize()
    repo = DraftRepository(db)
    return DraftService(repo)


class TestDraftService:
    """DraftServiceのテスト。"""

    def test_save_and_get(self, draft_service: DraftService) -> None:
        """保存と取得ができることを確認する。"""
        draft = Draft(title="テスト記事", body="本文")
        saved = draft_service.save(draft)

        found = draft_service.get(saved.id)  # type: ignore[arg-type]

        assert found is not None
        assert found.title == "テスト記事"

    def test_get_nonexistent(self, draft_service: DraftService) -> None:
        """存在しない下書きでNoneが返されることを確認する。"""
        assert draft_service.get(9999) is None

    def test_get_all(self, draft_service: DraftService) -> None:
        """全件取得ができることを確認する。"""
        draft_service.save(Draft(title="記事1", body="本文1"))
        draft_service.save(Draft(title="記事2", body="本文2"))

        drafts = draft_service.get_all()
        assert len(drafts) == 2

    def test_delete(self, draft_service: DraftService) -> None:
        """削除ができることを確認する。"""
        saved = draft_service.save(Draft(title="削除対象", body="本文"))

        result = draft_service.delete(saved.id)  # type: ignore[arg-type]

        assert result is True
        assert draft_service.get(saved.id) is None  # type: ignore[arg-type]

    def test_delete_nonexistent(self, draft_service: DraftService) -> None:
        """存在しない下書きの削除でFalseが返されることを確認する。"""
        assert draft_service.delete(9999) is False
