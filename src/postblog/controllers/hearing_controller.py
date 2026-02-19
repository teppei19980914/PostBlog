"""ヒアリング画面コントローラ。

ヒアリングの開始・メッセージ送信・終了をAsyncRunner経由で管理する。
"""

import logging
from typing import Any

from postblog.exceptions import ValidationError
from postblog.infrastructure.async_runner import AsyncRunner
from postblog.models.blog_type import BlogType
from postblog.models.hearing import HearingResult
from postblog.services.hearing_service import HearingService
from postblog.templates.hearing_templates import get_blog_type


logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 5000


class HearingController:
    """ヒアリングフローを管理するコントローラ。

    Args:
        hearing_service: ヒアリングサービス。
        async_runner: 非同期ランナー。
    """

    def __init__(
        self,
        hearing_service: HearingService,
        async_runner: AsyncRunner,
    ) -> None:
        self._hearing_service = hearing_service
        self._async_runner = async_runner
        self._hearing_result: HearingResult | None = None
        self._blog_type: BlogType | None = None

    @property
    def hearing_result(self) -> HearingResult | None:
        """現在のヒアリング結果。"""
        return self._hearing_result

    @property
    def blog_type(self) -> BlogType | None:
        """現在のブログ種別。"""
        return self._blog_type

    def start_hearing(self, blog_type_id: str) -> dict[str, Any]:
        """ヒアリングを開始する。

        Args:
            blog_type_id: ブログ種別ID。

        Returns:
            初期化結果の辞書。

        Raises:
            ValidationError: 無効なブログ種別IDの場合。
        """
        blog_type = get_blog_type(blog_type_id)
        if blog_type is None:
            raise ValidationError(f"無効なブログ種別: {blog_type_id}")

        self._blog_type = blog_type
        self._hearing_result = self._hearing_service.start_hearing(blog_type)

        logger.info("ヒアリングを開始しました: blog_type=%s", blog_type_id)
        return {
            "blog_type_id": blog_type.id,
            "blog_type_name": blog_type.name,
            "hearing_items": [
                {
                    "key": item.key,
                    "question": item.question,
                    "required": item.required,
                }
                for item in blog_type.hearing_items
            ],
        }

    def send_message(
        self,
        user_message: str,
        on_success: Any = None,
        on_error: Any = None,
    ) -> None:
        """ユーザーメッセージを送信する（非同期）。

        Args:
            user_message: ユーザーのメッセージ。
            on_success: 成功時コールバック。
            on_error: 失敗時コールバック。

        Raises:
            ValidationError: ヒアリングが未開始またはメッセージが不正な場合。
        """
        if self._hearing_result is None or self._blog_type is None:
            raise ValidationError("ヒアリングが開始されていません。")

        validated = self._validate_message(user_message)

        hearing_result = self._hearing_result
        blog_type = self._blog_type

        async def _send() -> str:
            return await self._hearing_service.send_message(
                hearing_result, validated, blog_type
            )

        self._async_runner.run(_send(), on_success=on_success, on_error=on_error)

    def finish_hearing(
        self,
        on_success: Any = None,
        on_error: Any = None,
    ) -> None:
        """ヒアリングを終了してサマリーを生成する（非同期）。

        Args:
            on_success: 成功時コールバック。
            on_error: 失敗時コールバック。

        Raises:
            ValidationError: ヒアリングが未開始の場合。
        """
        if self._hearing_result is None:
            raise ValidationError("ヒアリングが開始されていません。")

        hearing_result = self._hearing_result

        async def _finish() -> HearingResult:
            return await self._hearing_service.generate_summary(hearing_result)

        def _on_success(result: HearingResult) -> None:
            self._hearing_result = result
            if on_success is not None:
                on_success(result)

        self._async_runner.run(_finish(), on_success=_on_success, on_error=on_error)

    def get_progress(self) -> dict[str, int]:
        """ヒアリング進捗を取得する。

        Returns:
            進捗情報の辞書（completed, total, percentage）。
        """
        if self._blog_type is None or self._hearing_result is None:
            return {"completed": 0, "total": 0, "percentage": 0}

        total = len(self._blog_type.hearing_items)
        messages = self._hearing_result.messages
        # ユーザーメッセージ数を進捗として使用（最大値はtotal）
        user_messages = sum(1 for m in messages if m.role == "user")
        completed = min(user_messages, total)
        percentage = int((completed / total) * 100) if total > 0 else 0

        return {"completed": completed, "total": total, "percentage": percentage}

    def reset(self) -> None:
        """ヒアリング状態をリセットする。"""
        self._hearing_result = None
        self._blog_type = None
        logger.info("ヒアリング状態をリセットしました")

    @staticmethod
    def _validate_message(message: str) -> str:
        """メッセージを検証する。

        Args:
            message: 検証するメッセージ。

        Returns:
            トリムされたメッセージ。

        Raises:
            ValidationError: メッセージが空または長すぎる場合。
        """
        trimmed = message.strip()
        if not trimmed:
            raise ValidationError("メッセージを入力してください。")
        if len(trimmed) > MAX_MESSAGE_LENGTH:
            raise ValidationError(
                f"メッセージは{MAX_MESSAGE_LENGTH}文字以内で入力してください。"
            )
        return trimmed
