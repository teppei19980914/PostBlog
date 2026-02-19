"""ホームコントローラのテスト。"""

from datetime import datetime
from unittest.mock import MagicMock

from postblog.controllers.home_controller import HomeController
from postblog.infrastructure.storage.history_repository import HistoryRecord
from postblog.models.draft import Draft


class TestGetRecentDrafts:
    """get_recent_drafts メソッドのテスト。"""

    def test_returns_drafts_sorted_by_updated_at(self) -> None:
        """下書きが更新日時の降順で返されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        draft_service.get_all.return_value = [
            Draft(
                id=1, title="古い記事", body="本文1", updated_at=datetime(2024, 1, 1)
            ),
            Draft(
                id=2, title="新しい記事", body="本文2", updated_at=datetime(2024, 6, 1)
            ),
        ]
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_drafts()

        assert len(result) == 2
        assert result[0]["title"] == "新しい記事"
        assert result[1]["title"] == "古い記事"

    def test_returns_limited_drafts(self) -> None:
        """件数制限が適用されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        draft_service.get_all.return_value = [
            Draft(id=i, title=f"記事{i}", body=f"本文{i}") for i in range(5)
        ]
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_drafts(limit=3)

        assert len(result) == 3

    def test_returns_empty_list_when_no_drafts(self) -> None:
        """下書きがない場合に空リストが返されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        draft_service.get_all.return_value = []
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_drafts()

        assert result == []

    def test_returns_empty_list_on_exception(self) -> None:
        """例外発生時に空リストが返されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        draft_service.get_all.side_effect = RuntimeError("DB error")
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_drafts()

        assert result == []

    def test_draft_with_no_title_shows_untitled(self) -> None:
        """タイトルなしの下書きが「無題」と表示されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        draft_service.get_all.return_value = [Draft(id=1, title="", body="本文")]
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_drafts()

        assert result[0]["title"] == "無題"

    def test_long_body_is_truncated_with_preview(self) -> None:
        """長い本文がプレビューで切り詰められることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        long_body = "あ" * 200
        draft_service.get_all.return_value = [Draft(id=1, title="記事", body=long_body)]
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_drafts()

        assert result[0]["preview"].endswith("...")
        assert len(result[0]["preview"]) == 103  # 100文字 + "..."

    def test_short_body_is_not_truncated(self) -> None:
        """短い本文がそのまま返されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        draft_service.get_all.return_value = [
            Draft(id=1, title="記事", body="短い本文")
        ]
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_drafts()

        assert result[0]["preview"] == "短い本文"


class TestGetRecentHistory:
    """get_recent_history メソッドのテスト。"""

    def test_returns_history_records(self) -> None:
        """投稿履歴が返されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        history_service.get_all.return_value = [
            HistoryRecord(
                id=1,
                title="記事1",
                service_name="Qiita",
                status="published",
                article_url="https://qiita.com/1",
                published_at=datetime(2024, 6, 1),
            ),
        ]
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_history()

        assert len(result) == 1
        assert result[0]["title"] == "記事1"
        assert result[0]["service_name"] == "Qiita"
        assert result[0]["article_url"] == "https://qiita.com/1"

    def test_returns_limited_history(self) -> None:
        """件数制限が適用されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        history_service.get_all.return_value = [
            HistoryRecord(id=i, title=f"記事{i}", service_name="Qiita")
            for i in range(5)
        ]
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_history(limit=2)

        assert len(result) == 2

    def test_returns_empty_list_on_exception(self) -> None:
        """例外発生時に空リストが返されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        history_service.get_all.side_effect = RuntimeError("DB error")
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_history()

        assert result == []

    def test_history_with_no_title_shows_untitled(self) -> None:
        """タイトルなしの履歴が「無題」と表示されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        history_service.get_all.return_value = [
            HistoryRecord(id=1, title="", service_name="Qiita")
        ]
        controller = HomeController(draft_service, history_service)

        result = controller.get_recent_history()

        assert result[0]["title"] == "無題"


class TestDeleteDraft:
    """delete_draft メソッドのテスト。"""

    def test_delete_success(self) -> None:
        """下書き削除が成功することを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        draft_service.delete.return_value = True
        controller = HomeController(draft_service, history_service)

        assert controller.delete_draft(1) is True
        draft_service.delete.assert_called_once_with(1)

    def test_delete_not_found(self) -> None:
        """存在しない下書きの削除がFalseを返すことを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        draft_service.delete.return_value = False
        controller = HomeController(draft_service, history_service)

        assert controller.delete_draft(999) is False

    def test_delete_on_exception(self) -> None:
        """例外発生時にFalseが返されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        draft_service.delete.side_effect = RuntimeError("DB error")
        controller = HomeController(draft_service, history_service)

        assert controller.delete_draft(1) is False


class TestDeleteHistory:
    """delete_history メソッドのテスト。"""

    def test_delete_success(self) -> None:
        """投稿履歴削除が成功することを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        history_service.delete.return_value = True
        controller = HomeController(draft_service, history_service)

        assert controller.delete_history(1) is True
        history_service.delete.assert_called_once_with(1)

    def test_delete_on_exception(self) -> None:
        """例外発生時にFalseが返されることを確認する。"""
        draft_service = MagicMock()
        history_service = MagicMock()
        history_service.delete.side_effect = RuntimeError("DB error")
        controller = HomeController(draft_service, history_service)

        assert controller.delete_history(1) is False
