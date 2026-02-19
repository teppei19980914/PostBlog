"""投稿コントローラ。

投稿先サービスの取得・バリデーション・投稿実行を管理する。
"""

import logging
from datetime import datetime
from typing import Any

from postblog.exceptions import ValidationError
from postblog.infrastructure.async_runner import AsyncRunner
from postblog.infrastructure.storage.history_repository import HistoryRecord
from postblog.models.article import Article
from postblog.models.publish_result import PublishRequest, PublishResult
from postblog.services.history_service import HistoryService
from postblog.services.publish_service import PublishService


logger = logging.getLogger(__name__)


class PublishController:
    """ブログ投稿を管理するコントローラ。

    Args:
        publish_service: 投稿サービス。
        history_service: 投稿履歴サービス。
        async_runner: 非同期ランナー。
    """

    def __init__(
        self,
        publish_service: PublishService,
        history_service: HistoryService,
        async_runner: AsyncRunner,
    ) -> None:
        self._publish_service = publish_service
        self._history_service = history_service
        self._async_runner = async_runner

    def get_available_services(self) -> list[dict[str, str]]:
        """利用可能な投稿サービス一覧を取得する。

        Returns:
            サービス情報の辞書リスト。
        """
        publishers = self._publish_service.get_publishers()
        return [
            {"name": name, "service_name": publisher.service_name}
            for name, publisher in publishers.items()
        ]

    def validate_publish_request(
        self, article: Article, service_names: list[str]
    ) -> list[str]:
        """投稿リクエストを検証する。

        Args:
            article: 投稿する記事。
            service_names: 投稿先サービス名のリスト。

        Returns:
            エラーメッセージのリスト（空なら有効）。
        """
        errors: list[str] = []

        if not article.title.strip():
            errors.append("タイトルが入力されていません。")

        if not article.body.strip():
            errors.append("本文が入力されていません。")

        if not service_names:
            errors.append("投稿先サービスが選択されていません。")

        publishers = self._publish_service.get_publishers()
        for name in service_names:
            if name not in publishers:
                errors.append(f"未登録のサービス: {name}")

        return errors

    def publish(
        self,
        article: Article,
        service_names: list[str],
        status: str = "publish",
        on_success: Any = None,
        on_error: Any = None,
    ) -> None:
        """記事を投稿する（非同期）。

        Args:
            article: 投稿する記事。
            service_names: 投稿先サービス名のリスト。
            status: 投稿ステータス（"publish" または "draft"）。
            on_success: 成功時コールバック。
            on_error: 失敗時コールバック。

        Raises:
            ValidationError: バリデーションエラーの場合。
        """
        errors = self.validate_publish_request(article, service_names)
        if errors:
            raise ValidationError("、".join(errors))

        request = PublishRequest(
            title=article.title,
            body=article.body,
            tags=article.tags,
            status=status,
            blog_type_id=article.blog_type_id,
        )

        async def _publish() -> list[PublishResult]:
            return await self._publish_service.publish(request, service_names)

        def _on_success(results: list[PublishResult]) -> None:
            self._save_history(article, results)
            if on_success is not None:
                on_success(results)

        self._async_runner.run(_publish(), on_success=_on_success, on_error=on_error)

    def retry_publish(
        self,
        article: Article,
        failed_service_names: list[str],
        status: str = "publish",
        on_success: Any = None,
        on_error: Any = None,
    ) -> None:
        """失敗したサービスへの再投稿を実行する（非同期）。

        Args:
            article: 投稿する記事。
            failed_service_names: 再投稿するサービス名のリスト。
            status: 投稿ステータス。
            on_success: 成功時コールバック。
            on_error: 失敗時コールバック。
        """
        self.publish(article, failed_service_names, status, on_success, on_error)

    def _save_history(self, article: Article, results: list[PublishResult]) -> None:
        """投稿結果を履歴に保存する。

        Args:
            article: 投稿した記事。
            results: 投稿結果リスト。
        """
        for result in results:
            try:
                record = HistoryRecord(
                    title=article.title,
                    body_preview=article.body[:200],
                    blog_type_id=article.blog_type_id,
                    service_name=result.service_name,
                    article_url=result.article_url,
                    status="published" if result.success else "failed",
                    published_at=datetime.now(),
                )
                self._history_service.save(record)
            except Exception:
                logger.exception(
                    "投稿履歴の保存に失敗しました: service=%s",
                    result.service_name,
                )

    @staticmethod
    def summarize_results(results: list[PublishResult]) -> dict[str, Any]:
        """投稿結果を集計する。

        Args:
            results: 投稿結果リスト。

        Returns:
            集計結果の辞書。
        """
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        return {
            "total": len(results),
            "success_count": len(successes),
            "failure_count": len(failures),
            "successes": [
                {
                    "service_name": r.service_name,
                    "article_url": r.article_url,
                }
                for r in successes
            ],
            "failures": [
                {
                    "service_name": r.service_name,
                    "error_message": r.error_message,
                }
                for r in failures
            ],
        }
