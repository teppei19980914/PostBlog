"""LLMクライアントのテスト。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from postblog.infrastructure.llm.openai_client import OpenAIClient


class TestOpenAIClient:
    """OpenAIClientのテスト。"""

    @pytest.mark.asyncio()
    async def test_chat(self) -> None:
        """chatメソッドがレスポンスを返すことを確認する。"""
        client = OpenAIClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, World!"

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.chat([{"role": "user", "content": "Hi"}])

        assert result == "Hello, World!"

    @pytest.mark.asyncio()
    async def test_chat_empty_response(self) -> None:
        """空のレスポンスが正しく処理されることを確認する。"""
        client = OpenAIClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.chat([{"role": "user", "content": "Hi"}])

        assert result == ""

    @pytest.mark.asyncio()
    async def test_chat_with_custom_model(self) -> None:
        """カスタムモデルが使用されることを確認する。"""
        client = OpenAIClient(api_key="test-key", model="gpt-4o-mini")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            await client.chat(
                [{"role": "user", "content": "Hi"}], model="gpt-3.5-turbo"
            )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio()
    async def test_chat_stream(self) -> None:
        """ストリーミングがチャンクを返すことを確認する。"""
        client = OpenAIClient(api_key="test-key")

        # AsyncIteratorをモック
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " World"
        mock_chunk3 = MagicMock()
        mock_chunk3.choices = []

        async def mock_aiter(*_args, **_kwargs):
            for chunk in [mock_chunk1, mock_chunk2, mock_chunk3]:
                yield chunk

        mock_stream = mock_aiter()

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_stream,
        ):
            chunks = []
            async for chunk in client.chat_stream([{"role": "user", "content": "Hi"}]):
                chunks.append(chunk)

        assert chunks == ["Hello", " World"]

    @pytest.mark.asyncio()
    async def test_test_connection_success(self) -> None:
        """接続テスト成功を確認する。"""
        client = OpenAIClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.test_connection()

        assert result is True

    @pytest.mark.asyncio()
    async def test_test_connection_failure(self) -> None:
        """接続テスト失敗を確認する。"""
        client = OpenAIClient(api_key="invalid-key")

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await client.test_connection()

        assert result is False
