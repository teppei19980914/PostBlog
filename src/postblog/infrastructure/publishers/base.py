"""ブログ投稿クライアントの抽象基底クラス。"""

from abc import ABC, abstractmethod

from postblog.models.publish_result import PublishRequest, PublishResult


class BlogPublisher(ABC):
    """ブログ投稿クライアントの抽象基底クラス。

    各投稿サービスの実装はこのクラスを継承する。
    """

    @property
    @abstractmethod
    def service_name(self) -> str:
        """サービス名を返す。"""

    @abstractmethod
    async def publish(self, request: PublishRequest) -> PublishResult:
        """記事を投稿する。

        Args:
            request: 投稿リクエスト。

        Returns:
            投稿結果。
        """

    @abstractmethod
    async def test_connection(self) -> bool:
        """接続テストを実行する。

        Returns:
            接続成功の場合True。
        """
