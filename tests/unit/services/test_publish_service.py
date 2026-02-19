"""投稿サービスのテスト。"""

from unittest.mock import AsyncMock

import pytest

from postblog.infrastructure.publishers.base import BlogPublisher
from postblog.models.publish_result import PublishRequest, PublishResult
from postblog.services.publish_service import PublishService


def _create_mock_publisher(
    name: str, success: bool = True, url: str = "https://example.com/article"
) -> BlogPublisher:
    """モック投稿クライアントを生成する。"""
    mock = AsyncMock(spec=BlogPublisher)
    mock.service_name = name
    if success:
        mock.publish = AsyncMock(
            return_value=PublishResult(success=True, service_name=name, article_url=url)
        )
    else:
        mock.publish = AsyncMock(
            return_value=PublishResult(
                success=False, service_name=name, error_message="error"
            )
        )
    mock.test_connection = AsyncMock(return_value=success)
    return mock


class TestPublishService:
    """PublishServiceのテスト。"""

    def test_register_publisher(self) -> None:
        """投稿クライアントが登録されることを確認する。"""
        service = PublishService()
        publisher = _create_mock_publisher("qiita")
        service.register_publisher(publisher)

        assert "qiita" in service.get_publishers()

    def test_get_publishers(self) -> None:
        """登録済みクライアントが取得されることを確認する。"""
        service = PublishService()
        service.register_publisher(_create_mock_publisher("qiita"))
        service.register_publisher(_create_mock_publisher("zenn"))

        publishers = service.get_publishers()
        assert len(publishers) == 2

    @pytest.mark.asyncio()
    async def test_publish_success(self) -> None:
        """投稿成功を確認する。"""
        service = PublishService()
        service.register_publisher(_create_mock_publisher("qiita"))

        request = PublishRequest(title="テスト", body="本文")
        results = await service.publish(request, ["qiita"])

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio()
    async def test_publish_multiple_services(self) -> None:
        """複数サービスへの投稿を確認する。"""
        service = PublishService()
        service.register_publisher(_create_mock_publisher("qiita"))
        service.register_publisher(_create_mock_publisher("zenn"))

        request = PublishRequest(title="テスト", body="本文")
        results = await service.publish(request, ["qiita", "zenn"])

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio()
    async def test_publish_unregistered_service(self) -> None:
        """未登録サービスへの投稿でエラーが返されることを確認する。"""
        service = PublishService()

        request = PublishRequest(title="テスト", body="本文")
        results = await service.publish(request, ["unknown"])

        assert len(results) == 1
        assert results[0].success is False
        assert "未登録" in (results[0].error_message or "")

    @pytest.mark.asyncio()
    async def test_publish_with_exception(self) -> None:
        """投稿中の例外が処理されることを確認する。"""
        service = PublishService()
        mock_pub = _create_mock_publisher("qiita")
        mock_pub.publish = AsyncMock(side_effect=Exception("Network error"))
        service.register_publisher(mock_pub)

        request = PublishRequest(title="テスト", body="本文")
        results = await service.publish(request, ["qiita"])

        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio()
    async def test_test_connection_registered(self) -> None:
        """登録済みサービスの接続テストを確認する。"""
        service = PublishService()
        service.register_publisher(_create_mock_publisher("qiita"))

        result = await service.test_connection("qiita")
        assert result is True

    @pytest.mark.asyncio()
    async def test_test_connection_unregistered(self) -> None:
        """未登録サービスの接続テストでFalseが返されることを確認する。"""
        service = PublishService()

        result = await service.test_connection("unknown")
        assert result is False
