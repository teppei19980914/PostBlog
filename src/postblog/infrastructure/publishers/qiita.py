"""Qiita投稿クライアント。

Qiita API v2を使用した記事投稿。
"""

import logging

import httpx

from postblog.infrastructure.publishers.base import BlogPublisher
from postblog.models.publish_result import PublishRequest, PublishResult


logger = logging.getLogger(__name__)

QIITA_API_BASE = "https://qiita.com/api/v2"


class QiitaPublisher(BlogPublisher):
    """Qiitaへの記事投稿クライアント。

    Args:
        api_token: Qiita個人用アクセストークン。
    """

    def __init__(self, api_token: str) -> None:
        self._api_token = api_token

    @property
    def service_name(self) -> str:
        """サービス名を返す。"""
        return "qiita"

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Qiitaに記事を投稿する。

        Args:
            request: 投稿リクエスト。

        Returns:
            投稿結果。
        """
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "title": request.title,
            "body": request.body,
            "tags": [{"name": tag, "versions": []} for tag in request.tags[:5]],
            "private": request.status == "draft",
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{QIITA_API_BASE}/items",
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                article_url = data.get("url", "")
                logger.info("Qiitaに投稿しました: %s", article_url)
                return PublishResult(
                    success=True,
                    service_name=self.service_name,
                    article_url=article_url,
                )
        except Exception as e:
            logger.error("Qiita投稿に失敗しました: %s", e)
            return PublishResult(
                success=False,
                service_name=self.service_name,
                error_message=str(e),
            )

    async def test_connection(self) -> bool:
        """接続テストを実行する。"""
        headers = {"Authorization": f"Bearer {self._api_token}"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{QIITA_API_BASE}/authenticated_user",
                    headers=headers,
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception:
            logger.exception("Qiita接続テストに失敗しました")
            return False
