"""ヒアリング→サマリー生成ワークフローの統合テスト。"""

import json
from unittest.mock import AsyncMock

import pytest

from postblog.controllers.hearing_controller import HearingController
from postblog.exceptions import ValidationError
from postblog.infrastructure.async_runner import AsyncRunner
from postblog.services.hearing_service import HearingService
from postblog.templates.hearing_templates import TECH_BLOG


class TestHearingWorkflow:
    """ヒアリングフロー全体の統合テスト。"""

    @pytest.mark.asyncio()
    async def test_hearing_send_message_and_generate_summary(
        self, mock_llm: AsyncMock, hearing_service: HearingService
    ) -> None:
        """HearingServiceでメッセージ送受信→サマリー生成の一連のフローを確認する。"""
        result = hearing_service.start_hearing(TECH_BLOG)
        assert result.blog_type_id == "tech"
        assert result.messages == []

        mock_llm.chat = AsyncMock(
            return_value="Pythonについてですね。詳しく教えてください。"
        )
        response = await hearing_service.send_message(
            result, "Pythonの入門記事を書きたいです", TECH_BLOG
        )
        assert "Python" in response
        assert len(result.messages) == 2

        mock_llm.chat = AsyncMock(
            return_value="初心者向けですね。環境構築も含めますか？"
        )
        await hearing_service.send_message(
            result, "初心者向けの記事にしたいです", TECH_BLOG
        )
        assert len(result.messages) == 4

        summary_data = {
            "summary": "Python入門記事。初心者向けに環境構築からHello Worldまでを解説。",
            "answers": {"topic": "Python入門", "target_reader": "初心者"},
            "seo_keywords": "Python 入門 初心者",
            "seo_target_audience": "プログラミング初心者",
            "seo_search_intent": "Python環境構築方法を知りたい",
        }
        mock_llm.chat = AsyncMock(return_value=json.dumps(summary_data))

        result = await hearing_service.generate_summary(result)
        assert result.completed is True
        assert "Python入門" in result.summary
        assert result.seo_keywords == "Python 入門 初心者"
        assert result.answers["topic"] == "Python入門"

    @pytest.mark.asyncio()
    async def test_hearing_summary_with_non_json_response(
        self, mock_llm: AsyncMock, hearing_service: HearingService
    ) -> None:
        """LLMがJSON以外を返した場合、応答全文をサマリーとして使用することを確認する。"""
        result = hearing_service.start_hearing(TECH_BLOG)

        mock_llm.chat = AsyncMock(return_value="回答をまとめた文章です。")
        await hearing_service.send_message(result, "テスト", TECH_BLOG)

        mock_llm.chat = AsyncMock(return_value="これはJSON形式ではないサマリーです。")
        result = await hearing_service.generate_summary(result)

        assert result.completed is True
        assert result.summary == "これはJSON形式ではないサマリーです。"

    def test_hearing_controller_start_and_progress(
        self,
        hearing_service: HearingService,
        async_runner: AsyncRunner,
    ) -> None:
        """HearingControllerで開始→進捗確認のフローを確認する。"""
        controller = HearingController(hearing_service, async_runner)

        info = controller.start_hearing("tech")
        assert info["blog_type_id"] == "tech"
        assert info["blog_type_name"] == "技術ブログ"
        assert len(info["hearing_items"]) == 5

        progress = controller.get_progress()
        assert progress["completed"] == 0
        assert progress["total"] == 5
        assert progress["percentage"] == 0

    def test_hearing_controller_invalid_blog_type(
        self,
        hearing_service: HearingService,
        async_runner: AsyncRunner,
    ) -> None:
        """無効なブログ種別IDでValidationErrorが発生することを確認する。"""
        controller = HearingController(hearing_service, async_runner)

        with pytest.raises(ValidationError, match="無効なブログ種別"):
            controller.start_hearing("nonexistent")

    def test_hearing_controller_reset(
        self,
        hearing_service: HearingService,
        async_runner: AsyncRunner,
    ) -> None:
        """リセット後に状態がクリアされることを確認する。"""
        controller = HearingController(hearing_service, async_runner)

        controller.start_hearing("tech")
        assert controller.hearing_result is not None
        assert controller.blog_type is not None

        controller.reset()
        assert controller.hearing_result is None
        assert controller.blog_type is None
