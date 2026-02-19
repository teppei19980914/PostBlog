"""OpenAI LLMクライアント実装。"""

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from postblog.infrastructure.llm.base import LLMClient


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"


class OpenAIClient(LLMClient):
    """OpenAI APIを使用したLLMクライアント。

    Args:
        api_key: OpenAI APIキー。
        model: デフォルトのモデル名。
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """チャット補完を実行する。

        Args:
            messages: メッセージリスト。
            model: 使用するモデル名。
            temperature: 温度パラメータ。

        Returns:
            LLMの応答テキスト。
        """
        use_model = model or self._model
        logger.debug(
            "OpenAI chat request: model=%s, messages=%d", use_model, len(messages)
        )

        response = await self._client.chat.completions.create(
            model=use_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )

        content = response.choices[0].message.content or ""
        logger.debug("OpenAI chat response: %d chars", len(content))
        return content

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """ストリーミングでチャット補完を実行する。

        Args:
            messages: メッセージリスト。
            model: 使用するモデル名。
            temperature: 温度パラメータ。

        Yields:
            応答テキストのチャンク。
        """
        use_model = model or self._model
        logger.debug(
            "OpenAI stream request: model=%s, messages=%d", use_model, len(messages)
        )

        stream = await self._client.chat.completions.create(
            model=use_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:  # type: ignore[union-attr]
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def test_connection(self) -> bool:
        """接続テストを実行する。

        Returns:
            接続成功の場合True。
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=5,
            )
            return len(response.choices) > 0
        except Exception:
            logger.exception("OpenAI接続テストに失敗しました")
            return False
