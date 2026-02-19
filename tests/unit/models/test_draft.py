"""Draftモデルのテスト。"""

from datetime import datetime

from postblog.models.draft import Draft


class TestDraft:
    """Draftのテスト。"""

    def test_create_draft_minimal(self) -> None:
        """デフォルト値でDraftが作成されることを確認する。"""
        draft = Draft()

        assert draft.id is None
        assert draft.title == ""
        assert draft.body == ""
        assert draft.tags == []
        assert draft.blog_type_id == ""
        assert draft.hearing_data == ""
        assert isinstance(draft.created_at, datetime)
        assert isinstance(draft.updated_at, datetime)

    def test_create_draft_full(self) -> None:
        """全フィールド指定でDraftが作成されることを確認する。"""
        now = datetime(2024, 6, 1, 12, 0, 0)
        draft = Draft(
            id=1,
            title="下書きタイトル",
            body="# 下書き\n\n本文です。",
            tags=["Python", "入門"],
            blog_type_id="tech",
            hearing_data='{"blog_type_id": "tech"}',
            created_at=now,
            updated_at=now,
        )

        assert draft.id == 1
        assert draft.title == "下書きタイトル"
        assert draft.body == "# 下書き\n\n本文です。"
        assert draft.tags == ["Python", "入門"]
        assert draft.blog_type_id == "tech"
        assert draft.hearing_data == '{"blog_type_id": "tech"}'
        assert draft.created_at == now

    def test_draft_is_mutable(self) -> None:
        """Draftがミュータブルであることを確認する。"""
        draft = Draft(title="元のタイトル", body="元の本文")

        draft.title = "変更後のタイトル"
        draft.body = "変更後の本文"
        draft.id = 42

        assert draft.title == "変更後のタイトル"
        assert draft.body == "変更後の本文"
        assert draft.id == 42

    def test_draft_tags_are_independent(self) -> None:
        """デフォルトのタグリストが独立していることを確認する。"""
        draft1 = Draft()
        draft2 = Draft()

        draft1.tags.append("tag1")

        assert len(draft2.tags) == 0

    def test_draft_timestamps_default_to_now(self) -> None:
        """タイムスタンプがデフォルトで現在時刻になることを確認する。"""
        before = datetime.now()
        draft = Draft()
        after = datetime.now()

        assert before <= draft.created_at <= after
        assert before <= draft.updated_at <= after

    def test_draft_new_has_none_id(self) -> None:
        """新規下書きのIDがNoneであることを確認する。"""
        draft = Draft(title="新規", body="本文")
        assert draft.id is None

    def test_draft_with_id(self) -> None:
        """ID付きの下書きが正しく作成されることを確認する。"""
        draft = Draft(id=100, title="保存済み", body="本文")
        assert draft.id == 100
