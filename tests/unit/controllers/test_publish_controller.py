"""投稿コントローラのテスト。"""

from unittest.mock import MagicMock

import pytest

from postblog.controllers.publish_controller import PublishController
from postblog.exceptions import ValidationError
from postblog.models.article import Article
from postblog.models.publish_result import PublishResult


class TestGetAvailableServices:
    """get_available_services メソッドのテスト。"""

    def test_returns_registered_services(self) -> None:
        """登録済みサービスが返されることを確認する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.service_name = "Qiita"
        publish_service.get_publishers.return_value = {"Qiita": mock_publisher}
        controller = PublishController(publish_service, history_service, async_runner)

        result = controller.get_available_services()

        assert len(result) == 1
        assert result[0]["name"] == "Qiita"

    def test_returns_empty_when_no_services(self) -> None:
        """サービス未登録時に空リストが返されることを確認する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        publish_service.get_publishers.return_value = {}
        controller = PublishController(publish_service, history_service, async_runner)

        result = controller.get_available_services()

        assert result == []


class TestValidatePublishRequest:
    """validate_publish_request メソッドのテスト。"""

    def _create_controller(self) -> PublishController:
        """テスト用コントローラを作成する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.service_name = "Qiita"
        publish_service.get_publishers.return_value = {"Qiita": mock_publisher}
        return PublishController(publish_service, history_service, async_runner)

    def test_valid_request(self) -> None:
        """有効なリクエストでエラーが空であることを確認する。"""
        controller = self._create_controller()
        article = Article(title="テスト記事", body="本文")

        errors = controller.validate_publish_request(article, ["Qiita"])

        assert errors == []

    def test_empty_title(self) -> None:
        """タイトルなしでエラーが返されることを確認する。"""
        controller = self._create_controller()
        article = Article(title="", body="本文")

        errors = controller.validate_publish_request(article, ["Qiita"])

        assert any("タイトル" in e for e in errors)

    def test_empty_body(self) -> None:
        """本文なしでエラーが返されることを確認する。"""
        controller = self._create_controller()
        article = Article(title="テスト", body="")

        errors = controller.validate_publish_request(article, ["Qiita"])

        assert any("本文" in e for e in errors)

    def test_no_service_selected(self) -> None:
        """サービス未選択でエラーが返されることを確認する。"""
        controller = self._create_controller()
        article = Article(title="テスト", body="本文")

        errors = controller.validate_publish_request(article, [])

        assert any("投稿先" in e for e in errors)

    def test_unregistered_service(self) -> None:
        """未登録サービスでエラーが返されることを確認する。"""
        controller = self._create_controller()
        article = Article(title="テスト", body="本文")

        errors = controller.validate_publish_request(article, ["UnknownService"])

        assert any("未登録" in e for e in errors)

    def test_multiple_errors(self) -> None:
        """複数エラーが同時に返されることを確認する。"""
        controller = self._create_controller()
        article = Article(title="", body="")

        errors = controller.validate_publish_request(article, [])

        assert len(errors) >= 3


class TestPublish:
    """publish メソッドのテスト。"""

    def test_publish_calls_async_runner(self) -> None:
        """投稿がAsyncRunnerを経由することを確認する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.service_name = "Qiita"
        publish_service.get_publishers.return_value = {"Qiita": mock_publisher}
        controller = PublishController(publish_service, history_service, async_runner)
        article = Article(title="テスト記事", body="本文")

        controller.publish(article, ["Qiita"])

        async_runner.run.assert_called_once()

    def test_publish_with_validation_error_raises(self) -> None:
        """バリデーションエラー時にValidationErrorが発生することを確認する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        publish_service.get_publishers.return_value = {}
        controller = PublishController(publish_service, history_service, async_runner)
        article = Article(title="", body="")

        with pytest.raises(ValidationError):
            controller.publish(article, [])

    def test_publish_saves_history_on_success(self) -> None:
        """投稿成功時に履歴が保存されることを確認する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.service_name = "Qiita"
        publish_service.get_publishers.return_value = {"Qiita": mock_publisher}
        controller = PublishController(publish_service, history_service, async_runner)
        article = Article(title="テスト記事", body="本文", blog_type_id="tech")

        controller.publish(article, ["Qiita"])

        # AsyncRunnerのon_successを呼び出し
        _, kwargs = async_runner.run.call_args
        results = [
            PublishResult(
                success=True,
                service_name="Qiita",
                article_url="https://qiita.com/1",
            )
        ]
        kwargs["on_success"](results)

        history_service.save.assert_called_once()

    def test_publish_with_custom_status(self) -> None:
        """カスタムステータスで投稿できることを確認する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.service_name = "Qiita"
        publish_service.get_publishers.return_value = {"Qiita": mock_publisher}
        controller = PublishController(publish_service, history_service, async_runner)
        article = Article(title="テスト記事", body="本文")

        controller.publish(article, ["Qiita"], status="draft")

        async_runner.run.assert_called_once()


class TestRetryPublish:
    """retry_publish メソッドのテスト。"""

    def test_retry_calls_publish(self) -> None:
        """リトライがpublishを呼び出すことを確認する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.service_name = "Qiita"
        publish_service.get_publishers.return_value = {"Qiita": mock_publisher}
        controller = PublishController(publish_service, history_service, async_runner)
        article = Article(title="テスト記事", body="本文")

        controller.retry_publish(article, ["Qiita"])

        async_runner.run.assert_called_once()


class TestSummarizeResults:
    """summarize_results 静的メソッドのテスト。"""

    def test_summarize_all_success(self) -> None:
        """全成功時の集計を確認する。"""
        results = [
            PublishResult(
                success=True, service_name="Qiita", article_url="https://qiita.com/1"
            ),
            PublishResult(
                success=True, service_name="Zenn", article_url="https://zenn.dev/1"
            ),
        ]

        summary = PublishController.summarize_results(results)

        assert summary["total"] == 2
        assert summary["success_count"] == 2
        assert summary["failure_count"] == 0
        assert len(summary["successes"]) == 2
        assert len(summary["failures"]) == 0

    def test_summarize_all_failure(self) -> None:
        """全失敗時の集計を確認する。"""
        results = [
            PublishResult(
                success=False, service_name="Qiita", error_message="認証エラー"
            ),
        ]

        summary = PublishController.summarize_results(results)

        assert summary["total"] == 1
        assert summary["success_count"] == 0
        assert summary["failure_count"] == 1
        assert summary["failures"][0]["error_message"] == "認証エラー"

    def test_summarize_mixed_results(self) -> None:
        """成功と失敗が混在する場合の集計を確認する。"""
        results = [
            PublishResult(
                success=True, service_name="Qiita", article_url="https://qiita.com/1"
            ),
            PublishResult(success=False, service_name="Zenn", error_message="エラー"),
        ]

        summary = PublishController.summarize_results(results)

        assert summary["total"] == 2
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 1

    def test_summarize_empty_results(self) -> None:
        """空結果の集計を確認する。"""
        summary = PublishController.summarize_results([])

        assert summary["total"] == 0
        assert summary["success_count"] == 0
        assert summary["failure_count"] == 0


class TestSaveHistory:
    """_save_history メソッドのテスト。"""

    def test_saves_history_for_each_result(self) -> None:
        """各結果に対して履歴が保存されることを確認する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        controller = PublishController(publish_service, history_service, async_runner)
        article = Article(title="テスト記事", body="本文" * 100, blog_type_id="tech")
        results = [
            PublishResult(
                success=True, service_name="Qiita", article_url="https://qiita.com/1"
            ),
            PublishResult(success=False, service_name="Zenn", error_message="エラー"),
        ]

        controller._save_history(article, results)

        assert history_service.save.call_count == 2

    def test_save_history_continues_on_error(self) -> None:
        """1件の保存失敗でも残りが保存されることを確認する。"""
        publish_service = MagicMock()
        history_service = MagicMock()
        async_runner = MagicMock()
        # 最初のsaveでエラー、2回目は成功
        history_service.save.side_effect = [RuntimeError("DB error"), MagicMock()]
        controller = PublishController(publish_service, history_service, async_runner)
        article = Article(title="テスト記事", body="本文", blog_type_id="tech")
        results = [
            PublishResult(success=True, service_name="Qiita"),
            PublishResult(success=True, service_name="Zenn"),
        ]

        controller._save_history(article, results)

        assert history_service.save.call_count == 2
