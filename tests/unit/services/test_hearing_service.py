"""ヒアリングサービスのテスト。"""

import json
from unittest.mock import AsyncMock

import pytest

from postblog.infrastructure.llm.base import LLMClient
from postblog.models.hearing import HearingResult
from postblog.services.hearing_service import HearingService
from postblog.templates.hearing_templates import TECH_BLOG


def _create_mock_llm(response: str = "テスト応答") -> LLMClient:
    """モックLLMクライアントを生成する。"""
    mock = AsyncMock(spec=LLMClient)
    mock.chat = AsyncMock(return_value=response)
    return mock


class TestHearingService:
    """HearingServiceのテスト。"""

    def test_start_hearing(self) -> None:
        """ヒアリング開始でHearingResultが初期化されることを確認する。"""
        service = HearingService(_create_mock_llm())
        result = service.start_hearing(TECH_BLOG)

        assert result.blog_type_id == "tech"
        assert result.messages == []
        assert result.completed is False

    @pytest.mark.asyncio()
    async def test_send_message(self) -> None:
        """メッセージ送信でAI応答が返されることを確認する。"""
        mock_llm = _create_mock_llm("テーマについて教えてください。")
        service = HearingService(mock_llm)
        hearing_result = HearingResult(blog_type_id="tech")

        response = await service.send_message(
            hearing_result, "Pythonについて書きたい", TECH_BLOG
        )

        assert response == "テーマについて教えてください。"
        assert len(hearing_result.messages) == 2
        assert hearing_result.messages[0].role == "user"
        assert hearing_result.messages[1].role == "assistant"

    @pytest.mark.asyncio()
    async def test_send_message_adds_to_history(self) -> None:
        """メッセージ送信が履歴に追加されることを確認する。"""
        mock_llm = _create_mock_llm("応答1")
        service = HearingService(mock_llm)
        hearing_result = HearingResult(blog_type_id="tech")

        await service.send_message(hearing_result, "質問1", TECH_BLOG)

        mock_llm.chat = AsyncMock(return_value="応答2")
        await service.send_message(hearing_result, "質問2", TECH_BLOG)

        assert len(hearing_result.messages) == 4

    @pytest.mark.asyncio()
    async def test_generate_summary_valid_json(self) -> None:
        """有効なJSONサマリーが生成されることを確認する。"""
        summary_data = {
            "summary": "Python入門記事についてのヒアリング",
            "answers": {"topic": "Python", "level": "初級"},
            "seo_keywords": "Python 入門",
            "seo_target_audience": "プログラミング初心者",
            "seo_search_intent": "Pythonの基本を学びたい",
        }
        mock_llm = _create_mock_llm(json.dumps(summary_data))
        service = HearingService(mock_llm)
        hearing_result = HearingResult(blog_type_id="tech")

        result = await service.generate_summary(hearing_result)

        assert result.summary == "Python入門記事についてのヒアリング"
        assert result.seo_keywords == "Python 入門"
        assert result.completed is True

    @pytest.mark.asyncio()
    async def test_generate_summary_invalid_json(self) -> None:
        """不正なJSONの場合にレスポンス全文がサマリーになることを確認する。"""
        mock_llm = _create_mock_llm("これはJSONではありません")
        service = HearingService(mock_llm)
        hearing_result = HearingResult(blog_type_id="tech")

        result = await service.generate_summary(hearing_result)

        assert result.summary == "これはJSONではありません"
        assert result.completed is True
