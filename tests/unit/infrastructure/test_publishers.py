"""ブログ投稿クライアントのテスト。"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from postblog.infrastructure.publishers.ameba import AmebaPublisher
from postblog.infrastructure.publishers.hatena import HatenaPublisher
from postblog.infrastructure.publishers.markdown_export import MarkdownExportPublisher
from postblog.infrastructure.publishers.qiita import QiitaPublisher
from postblog.infrastructure.publishers.wordpress import WordPressPublisher
from postblog.infrastructure.publishers.zenn import ZennPublisher
from postblog.models.publish_result import PublishRequest


def _mock_httpx_client(
    post_return: MagicMock | None = None,
    post_side_effect: Exception | None = None,
    get_return: MagicMock | None = None,
    get_side_effect: Exception | None = None,
) -> AsyncMock:
    """httpx.AsyncClientのモックを生成する。"""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    if post_side_effect:
        mock_client.post = AsyncMock(side_effect=post_side_effect)
    else:
        mock_client.post = AsyncMock(return_value=post_return)
    if get_side_effect:
        mock_client.get = AsyncMock(side_effect=get_side_effect)
    else:
        mock_client.get = AsyncMock(return_value=get_return)
    return mock_client


class TestQiitaPublisher:
    """QiitaPublisherのテスト。"""

    def test_service_name(self) -> None:
        """サービス名が正しいことを確認する。"""
        publisher = QiitaPublisher(api_token="test-token")
        assert publisher.service_name == "qiita"

    @pytest.mark.asyncio()
    async def test_publish_success(self) -> None:
        """投稿成功を確認する。"""
        publisher = QiitaPublisher(api_token="test-token")
        request = PublishRequest(title="テスト記事", body="# テスト", tags=["Python"])

        mock_response = MagicMock()
        mock_response.json.return_value = {"url": "https://qiita.com/test/items/123"}
        mock_response.raise_for_status = MagicMock()

        with patch("postblog.infrastructure.publishers.qiita.httpx.AsyncClient") as cls:
            cls.return_value = _mock_httpx_client(post_return=mock_response)
            result = await publisher.publish(request)

        assert result.success is True
        assert result.article_url == "https://qiita.com/test/items/123"

    @pytest.mark.asyncio()
    async def test_publish_failure(self) -> None:
        """投稿失敗を確認する。"""
        publisher = QiitaPublisher(api_token="invalid-token")
        request = PublishRequest(title="テスト", body="本文")

        with patch("postblog.infrastructure.publishers.qiita.httpx.AsyncClient") as cls:
            cls.return_value = _mock_httpx_client(post_side_effect=Exception("401"))
            result = await publisher.publish(request)

        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.asyncio()
    async def test_test_connection_success(self) -> None:
        """接続テスト成功を確認する。"""
        publisher = QiitaPublisher(api_token="test-token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("postblog.infrastructure.publishers.qiita.httpx.AsyncClient") as cls:
            cls.return_value = _mock_httpx_client(get_return=mock_resp)
            result = await publisher.test_connection()

        assert result is True

    @pytest.mark.asyncio()
    async def test_test_connection_failure(self) -> None:
        """接続テスト失敗を確認する。"""
        publisher = QiitaPublisher(api_token="invalid-token")

        with patch("postblog.infrastructure.publishers.qiita.httpx.AsyncClient") as cls:
            cls.return_value = _mock_httpx_client(get_side_effect=Exception("error"))
            result = await publisher.test_connection()

        assert result is False


class TestWordPressPublisher:
    """WordPressPublisherのテスト。"""

    def test_service_name(self) -> None:
        """サービス名が正しいことを確認する。"""
        publisher = WordPressPublisher("https://example.com", "user", "pass")
        assert publisher.service_name == "wordpress"

    @pytest.mark.asyncio()
    async def test_publish_success(self) -> None:
        """投稿成功を確認する。"""
        publisher = WordPressPublisher("https://example.com", "user", "pass")
        request = PublishRequest(title="テスト", body="# テスト", tags=["Python"])

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"link": "https://example.com/?p=1"}
        mock_resp.raise_for_status = MagicMock()

        with patch(
            "postblog.infrastructure.publishers.wordpress.httpx.AsyncClient"
        ) as cls:
            cls.return_value = _mock_httpx_client(post_return=mock_resp)
            result = await publisher.publish(request)

        assert result.success is True
        assert result.article_url == "https://example.com/?p=1"

    @pytest.mark.asyncio()
    async def test_publish_failure(self) -> None:
        """投稿失敗を確認する。"""
        publisher = WordPressPublisher("https://example.com", "user", "pass")
        request = PublishRequest(title="テスト", body="本文")

        with patch(
            "postblog.infrastructure.publishers.wordpress.httpx.AsyncClient"
        ) as cls:
            cls.return_value = _mock_httpx_client(post_side_effect=Exception("403"))
            result = await publisher.publish(request)

        assert result.success is False

    @pytest.mark.asyncio()
    async def test_test_connection_success(self) -> None:
        """接続テスト成功を確認する。"""
        publisher = WordPressPublisher("https://example.com", "user", "pass")
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch(
            "postblog.infrastructure.publishers.wordpress.httpx.AsyncClient"
        ) as cls:
            cls.return_value = _mock_httpx_client(get_return=mock_resp)
            result = await publisher.test_connection()

        assert result is True

    @pytest.mark.asyncio()
    async def test_test_connection_failure(self) -> None:
        """接続テスト失敗を確認する。"""
        publisher = WordPressPublisher("https://example.com", "user", "pass")

        with patch(
            "postblog.infrastructure.publishers.wordpress.httpx.AsyncClient"
        ) as cls:
            cls.return_value = _mock_httpx_client(get_side_effect=Exception("err"))
            result = await publisher.test_connection()

        assert result is False


class TestHatenaPublisher:
    """HatenaPublisherのテスト。"""

    def test_service_name(self) -> None:
        """サービス名が正しいことを確認する。"""
        publisher = HatenaPublisher("user", "blog.example.com", "api_key")
        assert publisher.service_name == "hatena"

    def test_build_wsse_header(self) -> None:
        """WSSEヘッダーが生成されることを確認する。"""
        publisher = HatenaPublisher("user", "blog.example.com", "api_key")
        header = publisher._build_wsse_header()

        assert "UsernameToken" in header
        assert 'Username="user"' in header
        assert "PasswordDigest=" in header
        assert "Nonce=" in header
        assert "Created=" in header

    @pytest.mark.asyncio()
    async def test_publish_success(self) -> None:
        """投稿成功を確認する。"""
        publisher = HatenaPublisher("user", "blog.example.com", "api_key")
        request = PublishRequest(title="テスト", body="本文", tags=["Python"])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch(
            "postblog.infrastructure.publishers.hatena.httpx.AsyncClient"
        ) as cls:
            cls.return_value = _mock_httpx_client(post_return=mock_resp)
            result = await publisher.publish(request)

        assert result.success is True

    @pytest.mark.asyncio()
    async def test_publish_failure(self) -> None:
        """投稿失敗を確認する。"""
        publisher = HatenaPublisher("user", "blog.example.com", "api_key")
        request = PublishRequest(title="テスト", body="本文")

        with patch(
            "postblog.infrastructure.publishers.hatena.httpx.AsyncClient"
        ) as cls:
            cls.return_value = _mock_httpx_client(post_side_effect=Exception("401"))
            result = await publisher.publish(request)

        assert result.success is False

    @pytest.mark.asyncio()
    async def test_test_connection_success(self) -> None:
        """接続テスト成功を確認する。"""
        publisher = HatenaPublisher("user", "blog.example.com", "api_key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch(
            "postblog.infrastructure.publishers.hatena.httpx.AsyncClient"
        ) as cls:
            cls.return_value = _mock_httpx_client(get_return=mock_resp)
            result = await publisher.test_connection()

        assert result is True

    @pytest.mark.asyncio()
    async def test_test_connection_failure(self) -> None:
        """接続テスト失敗を確認する。"""
        publisher = HatenaPublisher("user", "blog.example.com", "api_key")

        with patch(
            "postblog.infrastructure.publishers.hatena.httpx.AsyncClient"
        ) as cls:
            cls.return_value = _mock_httpx_client(get_side_effect=Exception("err"))
            result = await publisher.test_connection()

        assert result is False


class TestAmebaPublisher:
    """AmebaPublisherのテスト。"""

    def test_service_name(self) -> None:
        """サービス名が正しいことを確認する。"""
        publisher = AmebaPublisher("from@example.com", "to@example.com")
        assert publisher.service_name == "ameba"

    @pytest.mark.asyncio()
    async def test_publish_success(self) -> None:
        """投稿成功を確認する。"""
        publisher = AmebaPublisher(
            "from@example.com", "to@example.com", smtp_password="pass"
        )
        request = PublishRequest(title="テスト", body="本文")

        with patch(
            "postblog.infrastructure.publishers.ameba.smtplib.SMTP"
        ) as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp.__exit__ = MagicMock(return_value=None)
            mock_smtp_cls.return_value = mock_smtp

            result = await publisher.publish(request)

        assert result.success is True

    @pytest.mark.asyncio()
    async def test_publish_failure(self) -> None:
        """投稿失敗を確認する。"""
        publisher = AmebaPublisher("from@example.com", "to@example.com")
        request = PublishRequest(title="テスト", body="本文")

        with patch(
            "postblog.infrastructure.publishers.ameba.smtplib.SMTP",
            side_effect=Exception("SMTP error"),
        ):
            result = await publisher.publish(request)

        assert result.success is False

    @pytest.mark.asyncio()
    async def test_test_connection_success(self) -> None:
        """接続テスト成功を確認する。"""
        publisher = AmebaPublisher(
            "from@example.com", "to@example.com", smtp_password="pass"
        )

        with patch(
            "postblog.infrastructure.publishers.ameba.smtplib.SMTP"
        ) as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp.__exit__ = MagicMock(return_value=None)
            mock_smtp_cls.return_value = mock_smtp

            result = await publisher.test_connection()

        assert result is True

    @pytest.mark.asyncio()
    async def test_test_connection_failure(self) -> None:
        """接続テスト失敗を確認する。"""
        publisher = AmebaPublisher("from@example.com", "to@example.com")

        with patch(
            "postblog.infrastructure.publishers.ameba.smtplib.SMTP",
            side_effect=Exception("error"),
        ):
            result = await publisher.test_connection()

        assert result is False


class TestZennPublisher:
    """ZennPublisherのテスト。"""

    def test_service_name(self) -> None:
        """サービス名が正しいことを確認する。"""
        publisher = ZennPublisher("/tmp/zenn", "token")
        assert publisher.service_name == "zenn"

    @pytest.mark.asyncio()
    async def test_publish_success(self, tmp_dir: Path) -> None:
        """投稿成功を確認する。"""
        # Git repoをモック
        publisher = ZennPublisher(str(tmp_dir), "token")
        request = PublishRequest(title="Test Article", body="# Hello", tags=["Python"])

        with patch("postblog.infrastructure.publishers.zenn.git.Repo") as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.index.add = MagicMock()
            mock_repo.index.commit = MagicMock()
            mock_repo.remotes.origin.push = MagicMock()
            mock_repo_cls.return_value = mock_repo

            result = await publisher.publish(request)

        assert result.success is True
        # ファイルが作成されたことを確認
        articles_dir = tmp_dir / "articles"
        assert articles_dir.exists()

    @pytest.mark.asyncio()
    async def test_publish_failure(self, tmp_dir: Path) -> None:
        """投稿失敗を確認する。"""
        publisher = ZennPublisher(str(tmp_dir), "token")
        request = PublishRequest(title="Test", body="Body")

        with patch(
            "postblog.infrastructure.publishers.zenn.git.Repo",
            side_effect=Exception("Git error"),
        ):
            result = await publisher.publish(request)

        assert result.success is False

    @pytest.mark.asyncio()
    async def test_test_connection_success(self, tmp_dir: Path) -> None:
        """接続テスト成功を確認する。"""
        publisher = ZennPublisher(str(tmp_dir), "token")

        with patch("postblog.infrastructure.publishers.zenn.git.Repo") as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.remotes.origin.exists.return_value = True
            mock_repo_cls.return_value = mock_repo

            result = await publisher.test_connection()

        assert result is True

    @pytest.mark.asyncio()
    async def test_test_connection_failure(self, tmp_dir: Path) -> None:
        """接続テスト失敗を確認する。"""
        publisher = ZennPublisher(str(tmp_dir), "token")

        with patch(
            "postblog.infrastructure.publishers.zenn.git.Repo",
            side_effect=Exception("error"),
        ):
            result = await publisher.test_connection()

        assert result is False


class TestMarkdownExportPublisher:
    """MarkdownExportPublisherのテスト。"""

    def test_service_name(self) -> None:
        """サービス名が正しいことを確認する。"""
        publisher = MarkdownExportPublisher(service_label="note")
        assert publisher.service_name == "note"

    def test_default_service_name(self) -> None:
        """デフォルトサービス名が正しいことを確認する。"""
        publisher = MarkdownExportPublisher()
        assert publisher.service_name == "markdown_export"

    @pytest.mark.asyncio()
    async def test_publish_creates_file(self, tmp_dir: Path) -> None:
        """エクスポートでファイルが作成されることを確認する。"""
        publisher = MarkdownExportPublisher(export_dir=tmp_dir)
        request = PublishRequest(
            title="Test Article", body="# Hello\n\nWorld", tags=["Python", "Test"]
        )

        result = await publisher.publish(request)

        assert result.success is True
        assert result.article_url is not None
        exported_file = Path(result.article_url)
        assert exported_file.exists()
        content = exported_file.read_text(encoding="utf-8")
        assert "Test Article" in content
        assert "# Hello" in content

    @pytest.mark.asyncio()
    async def test_publish_failure(self) -> None:
        """エクスポート失敗を確認する。"""
        publisher = MarkdownExportPublisher(
            export_dir=Path("/nonexistent/readonly/dir")
        )
        request = PublishRequest(title="Test", body="Body")

        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            result = await publisher.publish(request)

        assert result.success is False

    @pytest.mark.asyncio()
    async def test_test_connection_success(self, tmp_dir: Path) -> None:
        """接続テスト成功を確認する。"""
        publisher = MarkdownExportPublisher(export_dir=tmp_dir)
        result = await publisher.test_connection()
        assert result is True

    @pytest.mark.asyncio()
    async def test_test_connection_failure(self) -> None:
        """接続テスト失敗を確認する。"""
        publisher = MarkdownExportPublisher(
            export_dir=Path("/nonexistent/readonly/dir")
        )

        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            result = await publisher.test_connection()

        assert result is False
