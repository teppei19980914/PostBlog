"""エンドツーエンドの統合テスト。

ヒアリング→記事生成→編集→SEO分析→下書き保存→投稿の全フローを検証する。
"""

import json
from unittest.mock import AsyncMock

import pytest

from postblog.controllers.article_controller import ArticleController
from postblog.controllers.hearing_controller import HearingController
from postblog.controllers.home_controller import HomeController
from postblog.controllers.publish_controller import PublishController
from postblog.infrastructure.async_runner import AsyncRunner
from postblog.infrastructure.storage.history_repository import HistoryRecord
from postblog.models.article import Article
from postblog.models.draft import Draft
from postblog.models.publish_result import PublishRequest
from postblog.services.article_service import ArticleService
from postblog.services.draft_service import DraftService
from postblog.services.hearing_service import HearingService
from postblog.services.history_service import HistoryService
from postblog.services.publish_service import PublishService
from postblog.templates.prompts import SEO_ADVICE_END_MARKER, SEO_ADVICE_START_MARKER


def _build_article_llm_response() -> str:
    """テスト用の記事生成LLMレスポンスを構築する。"""
    article_body = (
        "# Python入門完全ガイド\n\n"
        "## はじめに\n\nPythonは世界で最も人気のある言語です。\n\n"
        "## 環境構築\n\nPython公式サイトからダウンロードします。\n\n"
        "## Hello World\n\n```python\nprint('Hello')\n```\n\n"
        "## まとめ\n\nPythonは簡単に始められます。"
    )
    seo_json = json.dumps(
        {
            "items": [
                {
                    "category": "タイトル",
                    "point": "キーワード配置",
                    "reason": "Python入門が先頭で良い",
                    "edit_tip": "そのまま",
                }
            ],
            "summary": "良好なSEO構成",
            "target_keyword": "Python入門",
            "generated_at": "2024-01-01",
        }
    )
    return (
        f"{article_body}\n\n"
        f"{SEO_ADVICE_START_MARKER}\n{seo_json}\n{SEO_ADVICE_END_MARKER}"
    )


def _build_summary_data() -> dict:
    """テスト用のサマリーデータを構築する。"""
    return {
        "summary": "Python初心者向けの入門記事。環境構築とHello Worldを解説する。",
        "answers": {
            "topic": "Python入門",
            "target_reader": "初心者",
            "problem": "環境構築方法が分からない",
            "solution": "ステップバイステップで解説",
        },
        "seo_keywords": "Python 入門 初心者 環境構築",
        "seo_target_audience": "プログラミング初心者",
        "seo_search_intent": "Python環境構築を1から知りたい",
    }


@pytest.mark.integration()
class TestEndToEndWorkflow:
    """ヒアリングから投稿完了までの全フローを検証する統合テスト。"""

    @pytest.mark.asyncio()
    async def test_full_workflow_hearing_to_article(
        self,
        mock_llm: AsyncMock,
        hearing_service: HearingService,
        article_service: ArticleService,
        draft_service: DraftService,
        async_runner: AsyncRunner,
    ) -> None:
        """ヒアリング→記事生成→編集→下書き保存のフローを検証する。"""
        # Step 1: ヒアリング
        hearing_controller = HearingController(hearing_service, async_runner)
        info = hearing_controller.start_hearing("tech")
        assert info["blog_type_id"] == "tech"

        hearing_result = hearing_controller.hearing_result
        blog_type = hearing_controller.blog_type

        mock_llm.chat = AsyncMock(
            return_value="Pythonの入門記事ですね。対象読者のレベルを教えてください。"
        )
        await hearing_service.send_message(
            hearing_result, "Pythonの入門記事を書きたいです", blog_type
        )

        mock_llm.chat = AsyncMock(
            return_value="初心者向けですね。何か具体的に解決したい課題はありますか？"
        )
        await hearing_service.send_message(
            hearing_result, "初心者向けにしたいです", blog_type
        )

        # サマリー生成
        mock_llm.chat = AsyncMock(return_value=json.dumps(_build_summary_data()))
        hearing_result = await hearing_service.generate_summary(hearing_result)
        assert hearing_result.completed is True

        # Step 2: 記事生成
        mock_llm.chat = AsyncMock(return_value=_build_article_llm_response())
        article_controller = ArticleController(
            article_service, draft_service, async_runner
        )
        article, seo_advice = await article_service.generate(hearing_result)
        article_controller._current_article = article
        article_controller._current_seo_advice = seo_advice
        article_controller._hearing_result = hearing_result

        assert article.title == "Python入門完全ガイド"
        assert len(seo_advice.items) == 1

        # Step 3: 記事編集
        article_controller.update_article(
            title="【2024年版】Python入門完全ガイド",
            tags=["Python", "入門", "初心者"],
            meta_description="Python初心者向けの環境構築からHello Worldまでを解説",
        )
        assert (
            article_controller.current_article.title
            == "【2024年版】Python入門完全ガイド"
        )

        # Step 4: SEO分析
        seo_result = article_controller.analyze_seo()
        assert seo_result.score >= 0

        # Step 5: 下書き保存
        draft = article_controller.save_draft()
        assert draft.id is not None
        assert draft.title == "【2024年版】Python入門完全ガイド"

    @pytest.mark.asyncio()
    async def test_full_workflow_publish(
        self,
        mock_llm: AsyncMock,
        article_service: ArticleService,
        draft_service: DraftService,
        history_service: HistoryService,
        publish_service: PublishService,
        async_runner: AsyncRunner,
        mock_publisher_qiita: AsyncMock,
    ) -> None:
        """記事作成→投稿→結果確認のフローを検証する。"""
        article_controller = ArticleController(
            article_service, draft_service, async_runner
        )
        article_controller._current_article = Article(
            title="テスト投稿記事",
            body="# テスト投稿記事\n\n本文です。",
            tags=["Python"],
            blog_type_id="tech",
        )

        # 投稿
        publish_service.register_publisher(mock_publisher_qiita)
        publish_controller = PublishController(
            publish_service, history_service, async_runner
        )

        current_article = article_controller.current_article
        errors = publish_controller.validate_publish_request(current_article, ["qiita"])
        assert errors == []

        request = PublishRequest(
            title=current_article.title,
            body=current_article.body,
            tags=current_article.tags,
            status="publish",
            blog_type_id=current_article.blog_type_id,
        )
        results = await publish_service.publish(request, ["qiita"])

        assert len(results) == 1
        assert results[0].success is True

        summary = publish_controller.summarize_results(results)
        assert summary["total"] == 1
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 0

    @pytest.mark.asyncio()
    async def test_workflow_with_draft_resume(
        self,
        mock_llm: AsyncMock,
        article_service: ArticleService,
        draft_service: DraftService,
        async_runner: AsyncRunner,
    ) -> None:
        """下書き保存→再開→編集→SEO分析のワークフローを検証する。"""
        controller = ArticleController(article_service, draft_service, async_runner)

        controller._current_article = Article(
            title="途中記事",
            body="# 途中記事\n\nここまで書いた。",
            tags=["WIP"],
            blog_type_id="tech",
            seo_keywords="テスト キーワード",
        )

        draft = controller.save_draft()
        draft_id = draft.id

        controller.reset()
        assert controller.current_article is None

        loaded_article = controller.load_draft(draft_id)
        assert loaded_article.title == "途中記事"
        assert loaded_article.tags == ["WIP"]

        controller.update_article(
            title="完成した記事",
            body="# 完成した記事\n\n全文を書き終えました。\n\n## まとめ\n\n完了。",
            tags=["Python", "完成"],
        )
        assert controller.current_article.title == "完成した記事"

        seo_result = controller.analyze_seo()
        assert seo_result.score >= 0

    def test_home_controller_with_drafts_and_history(
        self,
        draft_service: DraftService,
        history_service: HistoryService,
    ) -> None:
        """HomeControllerで下書き・履歴の一覧取得を確認する。"""
        home_controller = HomeController(draft_service, history_service)

        assert home_controller.get_recent_drafts() == []
        assert home_controller.get_recent_history() == []

        draft_service.save(Draft(title="下書き1", body="本文1"))
        draft_service.save(Draft(title="下書き2", body="本文2"))

        drafts = home_controller.get_recent_drafts(limit=10)
        assert len(drafts) == 2

        history_service.save(
            HistoryRecord(title="投稿1", service_name="qiita", status="success")
        )

        history = home_controller.get_recent_history(limit=10)
        assert len(history) == 1
        assert history[0]["title"] == "投稿1"
