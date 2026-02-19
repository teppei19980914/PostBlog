"""下書きリポジトリのテスト。"""

import pytest

from postblog.infrastructure.storage.database import Database
from postblog.infrastructure.storage.draft_repository import DraftRepository
from postblog.models.draft import Draft


@pytest.fixture()
def draft_repo() -> DraftRepository:
    """テスト用の下書きリポジトリフィクスチャ。"""
    db = Database(":memory:")
    db.initialize()
    return DraftRepository(db)


class TestDraftRepository:
    """DraftRepositoryのテスト。"""

    def test_save_new_draft(self, draft_repo: DraftRepository) -> None:
        """新規下書きが保存されることを確認する。"""
        draft = Draft(
            title="テスト記事", body="# テスト\n\n本文です。", blog_type_id="tech"
        )
        saved = draft_repo.save(draft)

        assert saved.id is not None
        assert saved.id > 0

    def test_save_and_find_by_id(self, draft_repo: DraftRepository) -> None:
        """保存した下書きがIDで検索できることを確認する。"""
        draft = Draft(
            title="テスト記事",
            body="本文です。",
            tags=["Python", "テスト"],
            blog_type_id="tech",
        )
        saved = draft_repo.save(draft)

        found = draft_repo.find_by_id(saved.id)  # type: ignore[arg-type]

        assert found is not None
        assert found.title == "テスト記事"
        assert found.body == "本文です。"
        assert found.tags == ["Python", "テスト"]
        assert found.blog_type_id == "tech"

    def test_find_by_id_nonexistent(self, draft_repo: DraftRepository) -> None:
        """存在しないIDで検索するとNoneが返されることを確認する。"""
        result = draft_repo.find_by_id(9999)
        assert result is None

    def test_find_all(self, draft_repo: DraftRepository) -> None:
        """全ての下書きが取得できることを確認する。"""
        draft_repo.save(Draft(title="記事1", body="本文1"))
        draft_repo.save(Draft(title="記事2", body="本文2"))
        draft_repo.save(Draft(title="記事3", body="本文3"))

        drafts = draft_repo.find_all()

        assert len(drafts) == 3

    def test_find_all_empty(self, draft_repo: DraftRepository) -> None:
        """下書きがない場合に空リストが返されることを確認する。"""
        drafts = draft_repo.find_all()
        assert drafts == []

    def test_update_existing_draft(self, draft_repo: DraftRepository) -> None:
        """既存の下書きが更新されることを確認する。"""
        draft = Draft(title="元のタイトル", body="元の本文")
        saved = draft_repo.save(draft)

        saved.title = "更新後のタイトル"
        saved.body = "更新後の本文"
        draft_repo.save(saved)

        found = draft_repo.find_by_id(saved.id)  # type: ignore[arg-type]

        assert found is not None
        assert found.title == "更新後のタイトル"
        assert found.body == "更新後の本文"

    def test_delete(self, draft_repo: DraftRepository) -> None:
        """下書きが削除されることを確認する。"""
        draft = Draft(title="削除対象", body="本文")
        saved = draft_repo.save(draft)

        result = draft_repo.delete(saved.id)  # type: ignore[arg-type]

        assert result is True
        assert draft_repo.find_by_id(saved.id) is None  # type: ignore[arg-type]

    def test_delete_nonexistent(self, draft_repo: DraftRepository) -> None:
        """存在しない下書きの削除でFalseが返されることを確認する。"""
        result = draft_repo.delete(9999)
        assert result is False

    def test_save_with_hearing_data(self, draft_repo: DraftRepository) -> None:
        """ヒアリングデータ付きの下書きが保存されることを確認する。"""
        draft = Draft(
            title="テスト",
            body="本文",
            hearing_data='{"blog_type_id": "tech", "completed": true}',
        )
        saved = draft_repo.save(draft)
        found = draft_repo.find_by_id(saved.id)  # type: ignore[arg-type]

        assert found is not None
        assert found.hearing_data == '{"blog_type_id": "tech", "completed": true}'
