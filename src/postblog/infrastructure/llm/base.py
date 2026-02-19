"""LLMクライアントの抽象基底クラス。"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMClient(ABC):
    """LLMクライアントの抽象基底クラス。

    各LLMプロバイダーの実装はこのクラスを継承する。
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """チャット補完を実行する。

        Args:
            messages: メッセージリスト（role, content）。
            model: 使用するモデル名（Noneの場合はデフォルト）。
            temperature: 生成時の温度パラメータ。

        Returns:
            LLMの応答テキスト。
        """

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """ストリーミングでチャット補完を実行する。

        Args:
            messages: メッセージリスト（role, content）。
            model: 使用するモデル名（Noneの場合はデフォルト）。
            temperature: 生成時の温度パラメータ。

        Yields:
            応答テキストのチャンク。
        """

    @abstractmethod
    async def test_connection(self) -> bool:
        """接続テストを実行する。

        Returns:
            接続成功の場合True。
        """
