"""記事コントローラのテスト。"""

from unittest.mock import MagicMock

import pytest

from postblog.controllers.article_controller import ArticleController
from postblog.exceptions import ValidationError
from postblog.models.article import Article
from postblog.models.draft import Draft
from postblog.models.hearing import HearingResult
from postblog.models.seo import SeoAdvice, SeoAdviceItem


class TestGenerateArticle:
    """generate_article メソッドのテスト。"""

    def test_generate_article_calls_async_runner(self) -> None:
        """記事生成がAsyncRunnerを経由することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)

        hearing_result = HearingResult(
            blog_type_id="tech", summary="サマリー", completed=True
        )
        controller.generate_article(hearing_result)

        async_runner.run.assert_called_once()

    def test_generate_article_with_incomplete_hearing_raises_error(self) -> None:
        """未完了のヒアリング結果でValidationErrorが発生することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)

        hearing_result = HearingResult(blog_type_id="tech", completed=False)
        with pytest.raises(ValidationError, match="ヒアリングが完了していません"):
            controller.generate_article(hearing_result)

    def test_generate_article_with_no_summary_raises_error(self) -> None:
        """サマリーなしのヒアリング結果でValidationErrorが発生することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)

        hearing_result = HearingResult(blog_type_id="tech", completed=True, summary="")
        with pytest.raises(ValidationError, match="ヒアリングサマリーがありません"):
            controller.generate_article(hearing_result)

    def test_generate_article_updates_state_on_success(self) -> None:
        """成功時に記事とSEOアドバイスが保持されることを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)

        hearing_result = HearingResult(
            blog_type_id="tech", summary="サマリー", completed=True
        )
        user_callback = MagicMock()
        controller.generate_article(hearing_result, on_success=user_callback)

        # AsyncRunnerのon_successを呼び出し
        _, kwargs = async_runner.run.call_args
        article = Article(title="テスト記事", body="# テスト\n\n本文")
        seo_advice = SeoAdvice(
            items=[SeoAdviceItem(category="c", point="p", reason="r", edit_tip="e")],
            summary="SEO OK",
        )
        kwargs["on_success"]((article, seo_advice))

        assert controller.current_article is article
        assert controller.current_seo_advice is seo_advice
        user_callback.assert_called_once()


class TestRegenerateArticle:
    """regenerate_article メソッドのテスト。"""

    def test_regenerate_without_hearing_result_raises_error(self) -> None:
        """ヒアリング結果なしでValidationErrorが発生することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)

        with pytest.raises(ValidationError, match="ヒアリング結果がありません"):
            controller.regenerate_article()

    def test_regenerate_calls_async_runner(self) -> None:
        """再生成がAsyncRunnerを経由することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)

        hearing_result = HearingResult(
            blog_type_id="tech", summary="サマリー", completed=True
        )
        controller.generate_article(hearing_result)
        async_runner.run.reset_mock()

        controller.regenerate_article()

        async_runner.run.assert_called_once()


class TestUpdateArticle:
    """update_article メソッドのテスト。"""

    def _create_controller_with_article(self) -> ArticleController:
        """記事をセットしたコントローラを作成するヘルパー。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)
        controller._current_article = Article(
            title="元のタイトル", body="元の本文", tags=["python"]
        )
        return controller

    def test_update_title(self) -> None:
        """タイトルが更新されることを確認する。"""
        controller = self._create_controller_with_article()

        result = controller.update_article(title="新しいタイトル")

        assert result.title == "新しいタイトル"

    def test_update_body(self) -> None:
        """本文が更新されることを確認する。"""
        controller = self._create_controller_with_article()

        result = controller.update_article(body="新しい本文")

        assert result.body == "新しい本文"

    def test_update_tags(self) -> None:
        """タグが更新されることを確認する。"""
        controller = self._create_controller_with_article()

        result = controller.update_article(tags=["python", "web"])

        assert result.tags == ["python", "web"]

    def test_update_meta_description(self) -> None:
        """メタディスクリプションが更新されることを確認する。"""
        controller = self._create_controller_with_article()

        result = controller.update_article(meta_description="説明文")

        assert result.meta_description == "説明文"

    def test_update_without_article_raises_error(self) -> None:
        """記事なしでValidationErrorが発生することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)

        with pytest.raises(ValidationError, match="編集中の記事がありません"):
            controller.update_article(title="新しいタイトル")

    def test_update_with_empty_title_raises_error(self) -> None:
        """空タイトルでValidationErrorが発生することを確認する。"""
        controller = self._create_controller_with_article()

        with pytest.raises(ValidationError, match="タイトルを入力してください"):
            controller.update_article(title="")

    def test_update_with_long_title_raises_error(self) -> None:
        """長すぎるタイトルでValidationErrorが発生することを確認する。"""
        controller = self._create_controller_with_article()

        with pytest.raises(ValidationError, match="100文字以内"):
            controller.update_article(title="あ" * 101)

    def test_update_with_long_tag_raises_error(self) -> None:
        """長すぎるタグでValidationErrorが発生することを確認する。"""
        controller = self._create_controller_with_article()

        with pytest.raises(ValidationError, match="30文字以内"):
            controller.update_article(tags=["あ" * 31])

    def test_update_multiple_fields(self) -> None:
        """複数フィールドを同時に更新できることを確認する。"""
        controller = self._create_controller_with_article()

        result = controller.update_article(
            title="新タイトル", body="新本文", tags=["新タグ"]
        )

        assert result.title == "新タイトル"
        assert result.body == "新本文"
        assert result.tags == ["新タグ"]

    def test_update_preserves_unchanged_fields(self) -> None:
        """更新しないフィールドが保持されることを確認する。"""
        controller = self._create_controller_with_article()

        result = controller.update_article(title="新しいタイトル")

        assert result.body == "元の本文"
        assert result.tags == ["python"]


class TestAnalyzeSeo:
    """analyze_seo メソッドのテスト。"""

    def test_analyze_seo_returns_result(self) -> None:
        """SEO分析結果が返されることを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)
        controller._current_article = Article(
            title="Python入門ガイド",
            body="# Python入門ガイド\n\n## Pythonとは\n\n### 基本構文\n\nPythonは"
            + "テスト本文。" * 200,
            seo_keywords="python",
            meta_description="Pythonの入門ガイドです。初心者向けに基本構文からサンプルコードまで解説します。"
            * 2,
        )

        result = controller.analyze_seo()

        assert result.score >= 0
        assert len(result.items) == 13

    def test_analyze_seo_without_article_raises_error(self) -> None:
        """記事なしでValidationErrorが発生することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)

        with pytest.raises(ValidationError, match="分析対象の記事がありません"):
            controller.analyze_seo()


class TestSaveDraft:
    """save_draft メソッドのテスト。"""

    def test_save_draft_success(self) -> None:
        """下書き保存が成功することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        saved_draft = Draft(id=1, title="テスト記事", body="本文")
        draft_service.save.return_value = saved_draft
        controller = ArticleController(article_service, draft_service, async_runner)
        controller._current_article = Article(
            title="テスト記事", body="本文", tags=["python"], blog_type_id="tech"
        )

        result = controller.save_draft()

        assert result.id == 1
        draft_service.save.assert_called_once()

    def test_save_draft_without_article_raises_error(self) -> None:
        """記事なしでValidationErrorが発生することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)

        with pytest.raises(ValidationError, match="保存する記事がありません"):
            controller.save_draft()


class TestLoadDraft:
    """load_draft メソッドのテスト。"""

    def test_load_draft_success(self) -> None:
        """下書き読み込みが成功することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        draft_service.get.return_value = Draft(
            id=1, title="テスト記事", body="本文", tags=["python"], blog_type_id="tech"
        )
        controller = ArticleController(article_service, draft_service, async_runner)

        result = controller.load_draft(1)

        assert result.title == "テスト記事"
        assert result.body == "本文"
        assert controller.current_article is not None
        assert controller.current_seo_advice is None

    def test_load_draft_not_found_raises_error(self) -> None:
        """存在しない下書きでValidationErrorが発生することを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        draft_service.get.return_value = None
        controller = ArticleController(article_service, draft_service, async_runner)

        with pytest.raises(ValidationError, match="下書きが見つかりません"):
            controller.load_draft(999)


class TestReset:
    """reset メソッドのテスト。"""

    def test_reset_clears_state(self) -> None:
        """リセットで状態がクリアされることを確認する。"""
        article_service = MagicMock()
        draft_service = MagicMock()
        async_runner = MagicMock()
        controller = ArticleController(article_service, draft_service, async_runner)
        controller._current_article = Article(title="テスト", body="本文")
        controller._current_seo_advice = SeoAdvice()

        controller.reset()

        assert controller.current_article is None
        assert controller.current_seo_advice is None
