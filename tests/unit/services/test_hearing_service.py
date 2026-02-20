"""ヒアリングサービスのテスト。"""

import json
from unittest.mock import AsyncMock

import pytest

from postblog.infrastructure.llm.base import LLMClient
from postblog.models.hearing import HearingResult
from postblog.services.hearing_service import HearingService, _extract_json
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
    async def test_generate_summary_json_in_code_block(self) -> None:
        """コードブロックで囲まれたJSONサマリーが正しくパースされることを確認する。"""
        summary_data = {
            "summary": "日記記事のヒアリング",
            "answers": {"topic": "旅行"},
            "seo_keywords": "旅行 日記",
            "seo_target_audience": "旅行好き",
            "seo_search_intent": "旅行記を読みたい",
        }
        response = f"```json\n{json.dumps(summary_data)}\n```"
        mock_llm = _create_mock_llm(response)
        service = HearingService(mock_llm)
        hearing_result = HearingResult(blog_type_id="diary")

        result = await service.generate_summary(hearing_result)

        assert result.summary == "日記記事のヒアリング"
        assert result.seo_keywords == "旅行 日記"
        assert result.completed is True

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


class TestExtractJson:
    """_extract_json関数のテスト。"""

    def test_extract_plain_json(self) -> None:
        """プレーンなJSON文字列をパースできることを確認する。"""
        text = '{"key": "value"}'
        assert _extract_json(text) == {"key": "value"}

    def test_extract_json_from_code_block(self) -> None:
        """マークダウンコードブロックからJSONを抽出できることを確認する。"""
        text = '```json\n{"key": "value"}\n```'
        assert _extract_json(text) == {"key": "value"}

    def test_extract_json_from_code_block_without_lang(self) -> None:
        """言語指定なしのコードブロックからJSONを抽出できることを確認する。"""
        text = '```\n{"key": "value"}\n```'
        assert _extract_json(text) == {"key": "value"}

    def test_extract_json_with_surrounding_text(self) -> None:
        """前後にテキストがある場合でもJSONを抽出できることを確認する。"""
        text = '以下がサマリーです。\n{"key": "value"}\n以上です。'
        assert _extract_json(text) == {"key": "value"}

    def test_extract_json_code_block_with_invalid_json_falls_through(self) -> None:
        """コードブロック内が不正JSONの場合、ブレース抽出にフォールバックすることを確認する。"""
        text = '```json\n不正なJSON\n```\n{"key": "fallback"}'
        assert _extract_json(text) == {"key": "fallback"}

    def test_extract_json_brace_with_invalid_json_raises(self) -> None:
        """ブレースはあるがJSONとして不正な場合にValueErrorが発生することを確認する。"""
        text = "結果は {不正: データ} です。"
        with pytest.raises(ValueError, match="JSONの抽出に失敗しました"):
            _extract_json(text)

    def test_extract_json_raises_on_no_json(self) -> None:
        """JSONが含まれない場合にValueErrorが発生することを確認する。"""
        with pytest.raises(ValueError, match="JSONの抽出に失敗しました"):
            _extract_json("これはJSONではありません")
