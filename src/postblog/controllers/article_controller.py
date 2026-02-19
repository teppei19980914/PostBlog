"""記事編集コントローラ。

記事の生成・SEO分析・下書き保存・再生成を管理する。
"""

import logging
from datetime import datetime
from typing import Any

from postblog.exceptions import ValidationError
from postblog.infrastructure.async_runner import AsyncRunner
from postblog.models.article import Article
from postblog.models.draft import Draft
from postblog.models.hearing import HearingResult
from postblog.models.seo import SeoAdvice, SeoAnalysisResult
from postblog.services.article_service import ArticleService
from postblog.services.draft_service import DraftService
from postblog.services.seo_service import analyze_seo


logger = logging.getLogger(__name__)


class ArticleController:
    """記事の生成・編集・SEO分析を管理するコントローラ。

    Args:
        article_service: 記事生成サービス。
        draft_service: 下書き管理サービス。
        async_runner: 非同期ランナー。
    """

    def __init__(
        self,
        article_service: ArticleService,
        draft_service: DraftService,
        async_runner: AsyncRunner,
    ) -> None:
        self._article_service = article_service
        self._draft_service = draft_service
        self._async_runner = async_runner
        self._current_article: Article | None = None
        self._current_seo_advice: SeoAdvice | None = None
        self._hearing_result: HearingResult | None = None

    @property
    def current_article(self) -> Article | None:
        """現在編集中の記事。"""
        return self._current_article

    @property
    def current_seo_advice(self) -> SeoAdvice | None:
        """現在のSEO対策ポイント。"""
        return self._current_seo_advice

    def generate_article(
        self,
        hearing_result: HearingResult,
        on_success: Any = None,
        on_error: Any = None,
    ) -> None:
        """ヒアリング結果から記事を生成する（非同期）。

        Args:
            hearing_result: ヒアリング結果。
            on_success: 成功時コールバック。
            on_error: 失敗時コールバック。

        Raises:
            ValidationError: ヒアリング結果が不完全な場合。
        """
        if not hearing_result.completed:
            raise ValidationError("ヒアリングが完了していません。")
        if not hearing_result.summary:
            raise ValidationError("ヒアリングサマリーがありません。")

        self._hearing_result = hearing_result

        async def _generate() -> tuple[Article, SeoAdvice]:
            return await self._article_service.generate(hearing_result)

        def _on_success(result: tuple[Article, SeoAdvice]) -> None:
            article, seo_advice = result
            self._current_article = article
            self._current_seo_advice = seo_advice
            if on_success is not None:
                on_success(result)

        self._async_runner.run(_generate(), on_success=_on_success, on_error=on_error)

    def regenerate_article(
        self,
        on_success: Any = None,
        on_error: Any = None,
    ) -> None:
        """記事を再生成する（非同期）。

        Args:
            on_success: 成功時コールバック。
            on_error: 失敗時コールバック。

        Raises:
            ValidationError: ヒアリング結果がない場合。
        """
        if self._hearing_result is None:
            raise ValidationError("ヒアリング結果がありません。再生成できません。")

        hearing_result = self._hearing_result

        async def _generate() -> tuple[Article, SeoAdvice]:
            return await self._article_service.generate(hearing_result)

        def _on_success(result: tuple[Article, SeoAdvice]) -> None:
            article, seo_advice = result
            self._current_article = article
            self._current_seo_advice = seo_advice
            if on_success is not None:
                on_success(result)

        self._async_runner.run(_generate(), on_success=_on_success, on_error=on_error)

    def update_article(
        self,
        title: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
        meta_description: str | None = None,
    ) -> Article:
        """記事を更新する。

        Args:
            title: 更新するタイトル。
            body: 更新する本文。
            tags: 更新するタグリスト。
            meta_description: 更新するメタディスクリプション。

        Returns:
            更新後の記事。

        Raises:
            ValidationError: 記事がない場合。
        """
        if self._current_article is None:
            raise ValidationError("編集中の記事がありません。")

        if title is not None:
            self._validate_title(title)
            self._current_article.title = title

        if body is not None:
            self._current_article.body = body

        if tags is not None:
            self._validate_tags(tags)
            self._current_article.tags = tags

        if meta_description is not None:
            self._current_article.meta_description = meta_description

        self._current_article.updated_at = datetime.now()
        return self._current_article

    def analyze_seo(self) -> SeoAnalysisResult:
        """現在の記事のSEO分析を実行する。

        Returns:
            SEO分析結果。

        Raises:
            ValidationError: 記事がない場合。
        """
        if self._current_article is None:
            raise ValidationError("分析対象の記事がありません。")

        result = analyze_seo(
            title=self._current_article.title,
            body=self._current_article.body,
            keyword=self._current_article.seo_keywords,
            meta_description=self._current_article.meta_description,
        )
        logger.info("SEO分析を実行しました: score=%d", result.score)
        return result

    def save_draft(self) -> Draft:
        """現在の記事を下書きとして保存する。

        Returns:
            保存された下書き。

        Raises:
            ValidationError: 記事がない場合。
        """
        if self._current_article is None:
            raise ValidationError("保存する記事がありません。")

        draft = Draft(
            title=self._current_article.title,
            body=self._current_article.body,
            tags=self._current_article.tags,
            blog_type_id=self._current_article.blog_type_id,
        )

        saved = self._draft_service.save(draft)
        logger.info("下書きを保存しました: id=%s", saved.id)
        return saved

    def load_draft(self, draft_id: int) -> Article:
        """下書きを記事として読み込む。

        Args:
            draft_id: 下書きID。

        Returns:
            読み込まれた記事。

        Raises:
            ValidationError: 下書きが見つからない場合。
        """
        draft = self._draft_service.get(draft_id)
        if draft is None:
            raise ValidationError(f"下書きが見つかりません: id={draft_id}")

        self._current_article = Article(
            title=draft.title,
            body=draft.body,
            tags=draft.tags,
            blog_type_id=draft.blog_type_id,
            created_at=draft.created_at,
            updated_at=draft.updated_at,
        )
        self._current_seo_advice = None
        self._hearing_result = None

        logger.info("下書きを読み込みました: id=%s", draft_id)
        return self._current_article

    def reset(self) -> None:
        """コントローラの状態をリセットする。"""
        self._current_article = None
        self._current_seo_advice = None
        self._hearing_result = None

    @staticmethod
    def _validate_title(title: str) -> None:
        """タイトルを検証する。

        Args:
            title: 検証するタイトル。

        Raises:
            ValidationError: タイトルが不正な場合。
        """
        if not title.strip():
            raise ValidationError("タイトルを入力してください。")
        if len(title) > 100:
            raise ValidationError("タイトルは100文字以内で入力してください。")

    @staticmethod
    def _validate_tags(tags: list[str]) -> None:
        """タグを検証する。

        Args:
            tags: 検証するタグリスト。

        Raises:
            ValidationError: タグが不正な場合。
        """
        for tag in tags:
            if len(tag) > 30:
                raise ValidationError(f"タグは30文字以内で入力してください: {tag}")
