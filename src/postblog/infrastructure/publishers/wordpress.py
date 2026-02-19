"""WordPress投稿クライアント。

WordPress REST APIを使用した記事投稿。
"""

import logging

import httpx
import mistune

from postblog.infrastructure.publishers.base import BlogPublisher
from postblog.models.publish_result import PublishRequest, PublishResult


logger = logging.getLogger(__name__)


class WordPressPublisher(BlogPublisher):
    """WordPressへの記事投稿クライアント。

    Args:
        site_url: WordPressサイトのURL。
        username: ユーザー名。
        password: アプリケーションパスワード。
    """

    def __init__(self, site_url: str, username: str, password: str) -> None:
        self._site_url = site_url.rstrip("/")
        self._username = username
        self._password = password

    @property
    def service_name(self) -> str:
        """サービス名を返す。"""
        return "wordpress"

    async def publish(self, request: PublishRequest) -> PublishResult:
        """WordPressに記事を投稿する。

        Args:
            request: 投稿リクエスト。

        Returns:
            投稿結果。
        """
        # MarkdownをHTMLに変換
        html_body = mistune.html(request.body)

        payload = {
            "title": request.title,
            "content": html_body,
            "status": "publish" if request.status == "publish" else "draft",
            "tags": [],
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._site_url}/wp-json/wp/v2/posts",
                    json=payload,
                    auth=(self._username, self._password),
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                article_url = data.get("link", "")
                logger.info("WordPressに投稿しました: %s", article_url)
                return PublishResult(
                    success=True,
                    service_name=self.service_name,
                    article_url=article_url,
                )
        except Exception as e:
            logger.error("WordPress投稿に失敗しました: %s", e)
            return PublishResult(
                success=False,
                service_name=self.service_name,
                error_message=str(e),
            )

    async def test_connection(self) -> bool:
        """接続テストを実行する。"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self._site_url}/wp-json/wp/v2/users/me",
                    auth=(self._username, self._password),
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception:
            logger.exception("WordPress接続テストに失敗しました")
            return False
