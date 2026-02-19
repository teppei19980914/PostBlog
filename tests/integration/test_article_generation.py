"""ヒアリング→記事生成→SEO分析→下書き保存ワークフローの統合テスト。"""

import json
from unittest.mock import AsyncMock

import pytest

from postblog.controllers.article_controller import ArticleController
from postblog.exceptions import ValidationError
from postblog.infrastructure.async_runner import AsyncRunner
from postblog.models.article import Article
from postblog.models.draft import Draft
from postblog.models.hearing import HearingMessage, HearingResult
from postblog.services.article_service import ArticleService
from postblog.services.draft_service import DraftService
from postblog.templates.prompts import SEO_ADVICE_END_MARKER, SEO_ADVICE_START_MARKER


def _build_full_article_response() -> str:
    """SEOアドバイス付きの完全な記事レスポンスを構築する。"""
    article = (
        "# Python入門ガイド\n\n"
        "## はじめに\n\n"
        "Pythonは初心者にも扱いやすいプログラミング言語です。\n\n"
        "## 環境構築\n\n"
        "Python公式サイトからインストーラーをダウンロードします。\n\n"
        "## Hello World\n\n"
        '```python\nprint("Hello, World!")\n```\n\n'
        "## まとめ\n\n"
        "Pythonは簡単に始められる言語です。"
    )

    advice = json.dumps(
        {
            "items": [
                {
                    "category": "タイトル",
                    "point": "キーワード配置",
                    "reason": "「Python入門」が先頭にあり検索上位を狙える",
                    "edit_tip": "そのままで問題なし",
                },
                {
                    "category": "本文",
                    "point": "見出し構造",
                    "reason": "H2が適切に分割されている",
                    "edit_tip": "キーワードを含む見出しを追加するとさらに良い",
                },
            ],
            "summary": "全体的にSEO対策が良好",
            "target_keyword": "Python入門",
            "generated_at": "2024-01-01T00:00:00",
        }
    )

    return f"{article}\n\n{SEO_ADVICE_START_MARKER}\n{advice}\n{SEO_ADVICE_END_MARKER}"


def _create_completed_hearing_result() -> HearingResult:
    """完了済みのヒアリング結果を作成する。"""
    result = HearingResult(blog_type_id="tech")
    result.messages = [
        HearingMessage(role="user", content="Pythonの入門記事を書きたいです"),
        HearingMessage(role="assistant", content="Pythonですね。対象読者は？"),
        HearingMessage(role="user", content="初心者向けです"),
        HearingMessage(
            role="assistant", content="了解しました。環境構築も含めますか？"
        ),
    ]
    result.summary = (
        "Python入門記事。初心者向けに環境構築からHello Worldまでを解説する。"
    )
    result.seo_keywords = "Python 入門 初心者"
    result.seo_target_audience = "プログラミング初心者"
    result.seo_search_intent = "Python環境構築方法を知りたい"
    result.completed = True
    return result


class TestArticleGeneration:
    """記事生成ワークフローの統合テスト。"""

    @pytest.mark.asyncio()
    async def test_generate_article_from_hearing_result(
        self, mock_llm: AsyncMock, article_service: ArticleService
    ) -> None:
        """ヒアリング結果から記事とSEOアドバイスが生成されることを確認する。"""
        mock_llm.chat = AsyncMock(return_value=_build_full_article_response())

        hearing_result = _create_completed_hearing_result()
        article, seo_advice = await article_service.generate(hearing_result)

        assert article.title == "Python入門ガイド"
        assert "環境構築" in article.body
        assert article.blog_type_id == "tech"
        assert article.seo_keywords == "Python 入門 初心者"

        assert len(seo_advice.items) == 2
        assert seo_advice.items[0].category == "タイトル"
        assert seo_advice.summary == "全体的にSEO対策が良好"


class TestArticleEditing:
    """記事編集の統合テスト。"""

    def test_update_article_and_analyze_seo(
        self,
        mock_llm: AsyncMock,
        article_service: ArticleService,
        draft_service: DraftService,
        async_runner: AsyncRunner,
    ) -> None:
        """記事更新→SEO分析のフローを確認する。"""
        controller = ArticleController(article_service, draft_service, async_runner)

        controller._current_article = Article(
            title="Python入門ガイド",
            body="# Python入門ガイド\n\nPythonは初心者にも扱いやすいです。",
            tags=["Python", "入門"],
            blog_type_id="tech",
            seo_keywords="Python 入門",
        )

        updated = controller.update_article(title="Python入門完全ガイド2024")
        assert updated.title == "Python入門完全ガイド2024"

        updated = controller.update_article(tags=["Python", "入門", "2024"])
        assert len(updated.tags) == 3

        result = controller.analyze_seo()
        assert result.score >= 0
        assert result.score <= 100
        assert len(result.items) > 0

    def test_update_article_validation(
        self,
        mock_llm: AsyncMock,
        article_service: ArticleService,
        draft_service: DraftService,
        async_runner: AsyncRunner,
    ) -> None:
        """記事更新のバリデーションを確認する。"""
        controller = ArticleController(article_service, draft_service, async_runner)

        with pytest.raises(ValidationError, match="編集中の記事がありません"):
            controller.update_article(title="テスト")


class TestDraftWorkflow:
    """下書き保存・読み込みの統合テスト。"""

    def test_save_and_load_draft(
        self,
        mock_llm: AsyncMock,
        article_service: ArticleService,
        draft_service: DraftService,
        async_runner: AsyncRunner,
    ) -> None:
        """下書き保存→読み込みの往復を確認する。"""
        controller = ArticleController(article_service, draft_service, async_runner)

        controller._current_article = Article(
            title="下書きテスト記事",
            body="# 下書きテスト記事\n\n本文です。",
            tags=["テスト", "下書き"],
            blog_type_id="tech",
        )

        draft = controller.save_draft()
        assert draft.id is not None
        assert draft.title == "下書きテスト記事"

        controller.reset()
        assert controller.current_article is None

        loaded = controller.load_draft(draft.id)
        assert loaded.title == "下書きテスト記事"
        assert loaded.body == "# 下書きテスト記事\n\n本文です。"
        assert loaded.tags == ["テスト", "下書き"]

    def test_load_nonexistent_draft(
        self,
        mock_llm: AsyncMock,
        article_service: ArticleService,
        draft_service: DraftService,
        async_runner: AsyncRunner,
    ) -> None:
        """存在しない下書きIDで読み込むとエラーになることを確認する。"""
        controller = ArticleController(article_service, draft_service, async_runner)

        with pytest.raises(ValidationError, match="下書きが見つかりません"):
            controller.load_draft(9999)

    def test_save_multiple_drafts_and_list(
        self,
        draft_service: DraftService,
    ) -> None:
        """複数の下書き保存→一覧取得を確認する。"""
        draft1 = draft_service.save(Draft(title="記事1", body="本文1", tags=["tag1"]))
        draft2 = draft_service.save(Draft(title="記事2", body="本文2", tags=["tag2"]))

        all_drafts = draft_service.get_all()
        assert len(all_drafts) == 2

        loaded = draft_service.get(draft1.id)
        assert loaded is not None
        assert loaded.title == "記事1"

        assert draft_service.delete(draft2.id) is True
        assert len(draft_service.get_all()) == 1
