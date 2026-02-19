"""ヒアリングコントローラのテスト。"""

from unittest.mock import MagicMock

import pytest

from postblog.controllers.hearing_controller import (
    MAX_MESSAGE_LENGTH,
    HearingController,
)
from postblog.exceptions import ValidationError
from postblog.models.hearing import HearingMessage, HearingResult


class TestStartHearing:
    """start_hearing メソッドのテスト。"""

    def test_start_hearing_with_valid_blog_type(self) -> None:
        """有効なブログ種別でヒアリングが開始されることを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="tech")
        controller = HearingController(hearing_service, async_runner)

        result = controller.start_hearing("tech")

        assert result["blog_type_id"] == "tech"
        assert result["blog_type_name"] == "技術ブログ"
        assert len(result["hearing_items"]) > 0
        assert controller.hearing_result is not None
        assert controller.blog_type is not None

    def test_start_hearing_with_invalid_blog_type(self) -> None:
        """無効なブログ種別でValidationErrorが発生することを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        controller = HearingController(hearing_service, async_runner)

        with pytest.raises(ValidationError, match="無効なブログ種別"):
            controller.start_hearing("invalid_type")

    def test_start_hearing_returns_hearing_items(self) -> None:
        """ヒアリング項目が返されることを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="diary")
        controller = HearingController(hearing_service, async_runner)

        result = controller.start_hearing("diary")

        items = result["hearing_items"]
        assert all("key" in item for item in items)
        assert all("question" in item for item in items)
        assert all("required" in item for item in items)

    def test_start_hearing_all_blog_types(self) -> None:
        """全ブログ種別でヒアリングが開始できることを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        controller = HearingController(hearing_service, async_runner)

        for type_id in ["tech", "diary", "review", "news", "howto"]:
            hearing_service.start_hearing.return_value = HearingResult(
                blog_type_id=type_id
            )
            result = controller.start_hearing(type_id)
            assert result["blog_type_id"] == type_id


class TestSendMessage:
    """send_message メソッドのテスト。"""

    def test_send_message_calls_async_runner(self) -> None:
        """メッセージ送信がAsyncRunnerを経由することを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="tech")
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("tech")

        controller.send_message("テストメッセージ")

        async_runner.run.assert_called_once()

    def test_send_message_without_hearing_raises_error(self) -> None:
        """ヒアリング未開始時にValidationErrorが発生することを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        controller = HearingController(hearing_service, async_runner)

        with pytest.raises(ValidationError, match="ヒアリングが開始されていません"):
            controller.send_message("テスト")

    def test_send_empty_message_raises_error(self) -> None:
        """空メッセージでValidationErrorが発生することを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="tech")
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("tech")

        with pytest.raises(ValidationError, match="メッセージを入力してください"):
            controller.send_message("")

    def test_send_whitespace_only_message_raises_error(self) -> None:
        """空白のみのメッセージでValidationErrorが発生することを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="tech")
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("tech")

        with pytest.raises(ValidationError, match="メッセージを入力してください"):
            controller.send_message("   ")

    def test_send_too_long_message_raises_error(self) -> None:
        """長すぎるメッセージでValidationErrorが発生することを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="tech")
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("tech")

        with pytest.raises(ValidationError, match=f"{MAX_MESSAGE_LENGTH}文字以内"):
            controller.send_message("あ" * (MAX_MESSAGE_LENGTH + 1))

    def test_send_message_with_callbacks(self) -> None:
        """コールバック付きでメッセージ送信できることを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="tech")
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("tech")
        on_success = MagicMock()
        on_error = MagicMock()

        controller.send_message("テスト", on_success=on_success, on_error=on_error)

        async_runner.run.assert_called_once()
        _, kwargs = async_runner.run.call_args
        assert kwargs["on_success"] is not None
        assert kwargs["on_error"] is on_error


class TestFinishHearing:
    """finish_hearing メソッドのテスト。"""

    def test_finish_hearing_calls_async_runner(self) -> None:
        """ヒアリング終了がAsyncRunnerを経由することを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="tech")
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("tech")

        controller.finish_hearing()

        async_runner.run.assert_called_once()

    def test_finish_hearing_without_hearing_raises_error(self) -> None:
        """ヒアリング未開始時にValidationErrorが発生することを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        controller = HearingController(hearing_service, async_runner)

        with pytest.raises(ValidationError, match="ヒアリングが開始されていません"):
            controller.finish_hearing()

    def test_finish_hearing_updates_result_on_success(self) -> None:
        """成功時にヒアリング結果が更新されることを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="tech")
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("tech")
        user_callback = MagicMock()

        controller.finish_hearing(on_success=user_callback)

        # AsyncRunnerのon_successコールバックを取得して呼び出す
        _, kwargs = async_runner.run.call_args
        internal_on_success = kwargs["on_success"]

        updated_result = HearingResult(
            blog_type_id="tech", summary="サマリー", completed=True
        )
        internal_on_success(updated_result)

        assert controller.hearing_result is not None
        assert controller.hearing_result.completed is True
        user_callback.assert_called_once_with(updated_result)


class TestGetProgress:
    """get_progress メソッドのテスト。"""

    def test_progress_with_no_hearing(self) -> None:
        """ヒアリング未開始時の進捗が0であることを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        controller = HearingController(hearing_service, async_runner)

        result = controller.get_progress()

        assert result == {"completed": 0, "total": 0, "percentage": 0}

    def test_progress_with_messages(self) -> None:
        """メッセージ送信後の進捗が更新されることを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_result = HearingResult(
            blog_type_id="tech",
            messages=[
                HearingMessage(role="user", content="メッセージ1"),
                HearingMessage(role="assistant", content="応答1"),
                HearingMessage(role="user", content="メッセージ2"),
                HearingMessage(role="assistant", content="応答2"),
            ],
        )
        hearing_service.start_hearing.return_value = hearing_result
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("tech")
        # hearing_resultを直接設定（メッセージ付き）
        controller._hearing_result = hearing_result

        result = controller.get_progress()

        assert result["completed"] == 2
        assert result["total"] == 5  # techは5項目
        assert result["percentage"] == 40

    def test_progress_does_not_exceed_total(self) -> None:
        """進捗がtotalを超えないことを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_result = HearingResult(
            blog_type_id="diary",
            messages=[
                HearingMessage(role="user", content=f"メッセージ{i}") for i in range(10)
            ],
        )
        hearing_service.start_hearing.return_value = hearing_result
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("diary")
        controller._hearing_result = hearing_result

        result = controller.get_progress()

        assert result["completed"] <= result["total"]


class TestReset:
    """reset メソッドのテスト。"""

    def test_reset_clears_state(self) -> None:
        """リセットで状態がクリアされることを確認する。"""
        hearing_service = MagicMock()
        async_runner = MagicMock()
        hearing_service.start_hearing.return_value = HearingResult(blog_type_id="tech")
        controller = HearingController(hearing_service, async_runner)
        controller.start_hearing("tech")

        controller.reset()

        assert controller.hearing_result is None
        assert controller.blog_type is None


class TestValidateMessage:
    """_validate_message 静的メソッドのテスト。"""

    def test_valid_message(self) -> None:
        """有効なメッセージが通ることを確認する。"""
        result = HearingController._validate_message("有効なメッセージ")
        assert result == "有効なメッセージ"

    def test_trims_whitespace(self) -> None:
        """前後の空白がトリムされることを確認する。"""
        result = HearingController._validate_message("  テスト  ")
        assert result == "テスト"

    def test_max_length_message(self) -> None:
        """最大長のメッセージが通ることを確認する。"""
        message = "あ" * MAX_MESSAGE_LENGTH
        result = HearingController._validate_message(message)
        assert len(result) == MAX_MESSAGE_LENGTH
