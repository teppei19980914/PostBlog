"""投稿ワークフローの統合テスト。"""

from unittest.mock import AsyncMock

import pytest

from postblog.controllers.publish_controller import PublishController
from postblog.infrastructure.async_runner import AsyncRunner
from postblog.infrastructure.storage.history_repository import HistoryRecord
from postblog.models.article import Article
from postblog.models.publish_result import PublishRequest, PublishResult
from postblog.services.history_service import HistoryService
from postblog.services.publish_service import PublishService


def _create_test_article() -> Article:
    """テスト用の記事を作成する。"""
    return Article(
        title="テスト記事",
        body="# テスト記事\n\n本文です。",
        tags=["Python", "テスト"],
        blog_type_id="tech",
        meta_description="テスト用の記事です",
    )


class TestPublishWorkflow:
    """投稿ワークフローの統合テスト。"""

    @pytest.mark.asyncio()
    async def test_publish_to_single_service(
        self,
        publish_service: PublishService,
        mock_publisher_qiita: AsyncMock,
    ) -> None:
        """単一サービスへの投稿を確認する。"""
        publish_service.register_publisher(mock_publisher_qiita)

        request = PublishRequest(
            title="テスト記事",
            body="# テスト記事\n\n本文です。",
            tags=["Python"],
            status="publish",
        )

        results = await publish_service.publish(request, ["qiita"])
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].service_name == "qiita"
        assert "qiita.com" in results[0].article_url

    @pytest.mark.asyncio()
    async def test_publish_to_multiple_services(
        self,
        publish_service: PublishService,
        mock_publisher_qiita: AsyncMock,
        mock_publisher_zenn: AsyncMock,
    ) -> None:
        """複数サービスへの同時投稿を確認する。"""
        publish_service.register_publisher(mock_publisher_qiita)
        publish_service.register_publisher(mock_publisher_zenn)

        request = PublishRequest(
            title="テスト記事",
            body="本文です。",
            tags=["Python"],
        )

        results = await publish_service.publish(request, ["qiita", "zenn"])
        assert len(results) == 2
        assert all(r.success for r in results)

        service_names = {r.service_name for r in results}
        assert service_names == {"qiita", "zenn"}

    @pytest.mark.asyncio()
    async def test_publish_to_unregistered_service(
        self,
        publish_service: PublishService,
        mock_publisher_qiita: AsyncMock,
    ) -> None:
        """未登録サービスへの投稿がエラーになることを確認する。"""
        publish_service.register_publisher(mock_publisher_qiita)

        request = PublishRequest(title="テスト", body="本文")
        results = await publish_service.publish(request, ["qiita", "unknown"])

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert "未登録" in results[1].error_message

    @pytest.mark.asyncio()
    async def test_publish_with_service_error(
        self,
        publish_service: PublishService,
    ) -> None:
        """投稿中のエラーがPublishResultとして返されることを確認する。"""
        error_publisher = AsyncMock()
        error_publisher.service_name = "error_service"
        error_publisher.publish = AsyncMock(
            side_effect=ConnectionError("接続タイムアウト")
        )

        publish_service.register_publisher(error_publisher)

        request = PublishRequest(title="テスト", body="本文")
        results = await publish_service.publish(request, ["error_service"])

        assert len(results) == 1
        assert results[0].success is False
        assert "接続タイムアウト" in results[0].error_message

    @pytest.mark.asyncio()
    async def test_connection_test(
        self,
        publish_service: PublishService,
        mock_publisher_qiita: AsyncMock,
    ) -> None:
        """接続テストが正常に動作することを確認する。"""
        publish_service.register_publisher(mock_publisher_qiita)

        assert await publish_service.test_connection("qiita") is True
        assert await publish_service.test_connection("unknown") is False


class TestPublishControllerIntegration:
    """PublishControllerの統合テスト。"""

    def test_get_available_services(
        self,
        publish_service: PublishService,
        history_service: HistoryService,
        async_runner: AsyncRunner,
        mock_publisher_qiita: AsyncMock,
        mock_publisher_zenn: AsyncMock,
    ) -> None:
        """利用可能なサービス一覧の取得を確認する。"""
        publish_service.register_publisher(mock_publisher_qiita)
        publish_service.register_publisher(mock_publisher_zenn)

        controller = PublishController(publish_service, history_service, async_runner)
        services = controller.get_available_services()

        assert len(services) == 2
        service_names = [s["name"] for s in services]
        assert "qiita" in service_names
        assert "zenn" in service_names

    def test_validate_publish_request(
        self,
        publish_service: PublishService,
        history_service: HistoryService,
        async_runner: AsyncRunner,
        mock_publisher_qiita: AsyncMock,
    ) -> None:
        """投稿バリデーションを確認する。"""
        publish_service.register_publisher(mock_publisher_qiita)

        controller = PublishController(publish_service, history_service, async_runner)
        article = _create_test_article()

        errors = controller.validate_publish_request(article, ["qiita"])
        assert errors == []

        errors = controller.validate_publish_request(article, [])
        assert len(errors) > 0

    def test_summarize_results(
        self,
        publish_service: PublishService,
        history_service: HistoryService,
        async_runner: AsyncRunner,
    ) -> None:
        """投稿結果のサマリー生成を確認する。"""
        controller = PublishController(publish_service, history_service, async_runner)

        results = [
            PublishResult(
                success=True,
                service_name="qiita",
                article_url="https://qiita.com/test",
            ),
            PublishResult(
                success=False,
                service_name="zenn",
                error_message="認証エラー",
            ),
        ]

        summary = controller.summarize_results(results)
        assert summary["total"] == 2
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 1
        assert len(summary["successes"]) == 1
        assert len(summary["failures"]) == 1


class TestHistoryIntegration:
    """投稿履歴の統合テスト。"""

    def test_save_and_retrieve_history(self, history_service: HistoryService) -> None:
        """投稿履歴の保存と取得を確認する。"""
        record = HistoryRecord(
            title="テスト記事",
            service_name="qiita",
            article_url="https://qiita.com/test",
            status="success",
        )

        saved = history_service.save(record)
        assert saved.id is not None

        all_records = history_service.get_all()
        assert len(all_records) == 1
        assert all_records[0].title == "テスト記事"

        loaded = history_service.get(saved.id)
        assert loaded is not None
        assert loaded.service_name == "qiita"

    def test_delete_history(self, history_service: HistoryService) -> None:
        """投稿履歴の削除を確認する。"""
        record = history_service.save(
            HistoryRecord(
                title="削除テスト",
                service_name="zenn",
                status="success",
            )
        )

        assert history_service.delete(record.id) is True
        assert history_service.get(record.id) is None
        assert len(history_service.get_all()) == 0
