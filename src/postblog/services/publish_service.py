"""投稿サービス。

複数のブログサービスへの投稿を管理する。
"""

import logging

from postblog.infrastructure.publishers.base import BlogPublisher
from postblog.models.publish_result import PublishRequest, PublishResult


logger = logging.getLogger(__name__)


class PublishService:
    """ブログ投稿を管理するサービス。"""

    def __init__(self) -> None:
        self._publishers: dict[str, BlogPublisher] = {}

    def register_publisher(self, publisher: BlogPublisher) -> None:
        """投稿クライアントを登録する。

        Args:
            publisher: 投稿クライアント。
        """
        self._publishers[publisher.service_name] = publisher
        logger.info("投稿クライアントを登録しました: %s", publisher.service_name)

    def get_publishers(self) -> dict[str, BlogPublisher]:
        """登録済みの投稿クライアントを取得する。

        Returns:
            サービス名をキーとする投稿クライアントの辞書。
        """
        return dict(self._publishers)

    async def publish(
        self, request: PublishRequest, service_names: list[str]
    ) -> list[PublishResult]:
        """指定されたサービスに記事を投稿する。

        Args:
            request: 投稿リクエスト。
            service_names: 投稿先サービス名のリスト。

        Returns:
            各サービスの投稿結果リスト。
        """
        results: list[PublishResult] = []

        for name in service_names:
            publisher = self._publishers.get(name)
            if publisher is None:
                results.append(
                    PublishResult(
                        success=False,
                        service_name=name,
                        error_message=f"未登録のサービス: {name}",
                    )
                )
                continue

            try:
                result = await publisher.publish(request)
                results.append(result)
                logger.info("投稿結果: service=%s, success=%s", name, result.success)
            except Exception as e:
                logger.error(
                    "投稿中にエラーが発生しました: service=%s, error=%s", name, e
                )
                results.append(
                    PublishResult(
                        success=False,
                        service_name=name,
                        error_message=str(e),
                    )
                )

        return results

    async def test_connection(self, service_name: str) -> bool:
        """指定されたサービスの接続テストを実行する。

        Args:
            service_name: サービス名。

        Returns:
            接続成功の場合True。
        """
        publisher = self._publishers.get(service_name)
        if publisher is None:
            return False
        return await publisher.test_connection()
