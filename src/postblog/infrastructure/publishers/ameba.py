"""Amebaブログ投稿クライアント。

メール投稿機能を使用した記事投稿。
"""

import logging
import smtplib
from email.mime.text import MIMEText

from postblog.infrastructure.publishers.base import BlogPublisher
from postblog.models.publish_result import PublishRequest, PublishResult


logger = logging.getLogger(__name__)


class AmebaPublisher(BlogPublisher):
    """Amebaブログへの記事投稿クライアント（メール投稿）。

    Args:
        from_email: 送信元メールアドレス。
        posting_email: Ameba投稿用メールアドレス。
        smtp_server: SMTPサーバーアドレス。
        smtp_port: SMTPポート番号。
        smtp_password: SMTPパスワード。
    """

    def __init__(
        self,
        from_email: str,
        posting_email: str,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_password: str = "",
    ) -> None:
        self._from_email = from_email
        self._posting_email = posting_email
        self._smtp_server = smtp_server
        self._smtp_port = smtp_port
        self._smtp_password = smtp_password

    @property
    def service_name(self) -> str:
        """サービス名を返す。"""
        return "ameba"

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Amebaブログにメールで投稿する。

        Args:
            request: 投稿リクエスト。

        Returns:
            投稿結果。
        """
        try:
            msg = MIMEText(request.body, "plain", "utf-8")
            msg["Subject"] = request.title
            msg["From"] = self._from_email
            msg["To"] = self._posting_email

            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                server.starttls()
                server.login(self._from_email, self._smtp_password)
                server.send_message(msg)

            logger.info("Amebaブログにメール投稿しました")
            return PublishResult(
                success=True,
                service_name=self.service_name,
            )
        except Exception as e:
            logger.error("Amebaブログ投稿に失敗しました: %s", e)
            return PublishResult(
                success=False,
                service_name=self.service_name,
                error_message=str(e),
            )

    async def test_connection(self) -> bool:
        """接続テストを実行する。"""
        try:
            with smtplib.SMTP(self._smtp_server, self._smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self._from_email, self._smtp_password)
            return True
        except Exception:
            logger.exception("Amebaブログ接続テストに失敗しました")
            return False
