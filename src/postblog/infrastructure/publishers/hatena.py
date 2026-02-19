"""はてなブログ投稿クライアント。

AtomPub APIを使用した記事投稿。
"""

import hashlib
import logging
import secrets
from base64 import b64encode
from datetime import UTC, datetime

import httpx

from postblog.infrastructure.publishers.base import BlogPublisher
from postblog.models.publish_result import PublishRequest, PublishResult


logger = logging.getLogger(__name__)


class HatenaPublisher(BlogPublisher):
    """はてなブログへの記事投稿クライアント。

    Args:
        hatena_id: はてなID。
        blog_id: ブログID。
        api_key: APIキー。
    """

    def __init__(self, hatena_id: str, blog_id: str, api_key: str) -> None:
        self._hatena_id = hatena_id
        self._blog_id = blog_id
        self._api_key = api_key

    @property
    def service_name(self) -> str:
        """サービス名を返す。"""
        return "hatena"

    def _build_wsse_header(self) -> str:
        """WSSE認証ヘッダーを生成する。

        Returns:
            WSSE認証ヘッダー文字列。
        """
        nonce = secrets.token_bytes(20)
        created = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        digest = hashlib.sha1(
            nonce + created.encode() + self._api_key.encode()
        ).digest()

        nonce_b64 = b64encode(nonce).decode()
        digest_b64 = b64encode(digest).decode()

        return (
            f'UsernameToken Username="{self._hatena_id}", '
            f'PasswordDigest="{digest_b64}", '
            f'Nonce="{nonce_b64}", '
            f'Created="{created}"'
        )

    async def publish(self, request: PublishRequest) -> PublishResult:
        """はてなブログに記事を投稿する。

        Args:
            request: 投稿リクエスト。

        Returns:
            投稿結果。
        """
        categories = "".join(f'<category term="{tag}" />' for tag in request.tags)
        draft = "yes" if request.status == "draft" else "no"

        xml_body = f"""<?xml version="1.0" encoding="utf-8"?>
<entry xmlns="http://www.w3.org/2005/Atom"
       xmlns:app="http://www.w3.org/2007/app">
  <title>{request.title}</title>
  <content type="text/plain">{request.body}</content>
  {categories}
  <app:control>
    <app:draft>{draft}</app:draft>
  </app:control>
</entry>"""

        url = f"https://blog.hatena.ne.jp/{self._hatena_id}/{self._blog_id}/atom/entry"
        headers = {
            "X-WSSE": self._build_wsse_header(),
            "Content-Type": "application/atom+xml",
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, content=xml_body, headers=headers, timeout=30.0
                )
                response.raise_for_status()
                logger.info("はてなブログに投稿しました")
                return PublishResult(
                    success=True,
                    service_name=self.service_name,
                    article_url=url,
                )
        except Exception as e:
            logger.error("はてなブログ投稿に失敗しました: %s", e)
            return PublishResult(
                success=False,
                service_name=self.service_name,
                error_message=str(e),
            )

    async def test_connection(self) -> bool:
        """接続テストを実行する。"""
        url = f"https://blog.hatena.ne.jp/{self._hatena_id}/{self._blog_id}/atom/entry"
        headers = {
            "X-WSSE": self._build_wsse_header(),
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10.0)
                return response.status_code == 200
        except Exception:
            logger.exception("はてなブログ接続テストに失敗しました")
            return False
