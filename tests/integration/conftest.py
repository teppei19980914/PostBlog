"""統合テスト用の共通フィクスチャ。"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from postblog.infrastructure.async_runner import AsyncRunner
from postblog.infrastructure.llm.base import LLMClient
from postblog.infrastructure.publishers.base import BlogPublisher
from postblog.infrastructure.storage.database import Database
from postblog.infrastructure.storage.draft_repository import DraftRepository
from postblog.infrastructure.storage.history_repository import HistoryRepository
from postblog.models.publish_result import PublishResult
from postblog.services.article_service import ArticleService
from postblog.services.draft_service import DraftService
from postblog.services.hearing_service import HearingService
from postblog.services.history_service import HistoryService
from postblog.services.publish_service import PublishService
from postblog.templates.prompts import SEO_ADVICE_END_MARKER, SEO_ADVICE_START_MARKER


def _build_article_response(
    title: str = "テスト記事タイトル",
    body: str = "本文パラグラフです。",
    include_seo: bool = True,
) -> str:
    """テスト用のLLM記事生成レスポンスを構築する。"""
    article = f"# {title}\n\n{body}"
    if not include_seo:
        return article

    advice_json = json.dumps(
        {
            "items": [
                {
                    "category": "タイトル",
                    "point": "キーワード配置",
                    "reason": "検索上位表示のため",
                    "edit_tip": "先頭にキーワードを配置",
                },
                {
                    "category": "本文",
                    "point": "見出し構造",
                    "reason": "可読性とSEO向上",
                    "edit_tip": "H2/H3を適切に配置",
                },
            ],
            "summary": "SEO対策が適切に行われています",
            "target_keyword": "Python入門",
            "generated_at": "2024-01-01T00:00:00",
        }
    )
    return f"{article}\n\n{SEO_ADVICE_START_MARKER}\n{advice_json}\n{SEO_ADVICE_END_MARKER}"


def _build_summary_response(
    summary: str = "Pythonの入門記事。初心者向けに環境構築からHello Worldまでを解説する。",
) -> str:
    """テスト用のLLMサマリーレスポンスを構築する。"""
    return json.dumps(
        {
            "summary": summary,
            "answers": {"topic": "Python入門", "target_reader": "初心者"},
            "seo_keywords": "Python 入門 初心者",
            "seo_target_audience": "プログラミング初心者",
            "seo_search_intent": "Python環境構築方法を知りたい",
        }
    )


@pytest.fixture
def mock_llm() -> AsyncMock:
    """モックLLMクライアント。"""
    mock = AsyncMock(spec=LLMClient)
    mock.chat = AsyncMock(return_value="テスト応答です。")
    mock.test_connection = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_publisher_qiita() -> AsyncMock:
    """モックQiita投稿クライアント。"""
    mock = AsyncMock(spec=BlogPublisher)
    mock.service_name = "qiita"
    mock.publish = AsyncMock(
        return_value=PublishResult(
            success=True,
            service_name="qiita",
            article_url="https://qiita.com/user/items/test123",
        )
    )
    mock.test_connection = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_publisher_zenn() -> AsyncMock:
    """モックZenn投稿クライアント。"""
    mock = AsyncMock(spec=BlogPublisher)
    mock.service_name = "zenn"
    mock.publish = AsyncMock(
        return_value=PublishResult(
            success=True,
            service_name="zenn",
            article_url="https://zenn.dev/user/articles/test123",
        )
    )
    mock.test_connection = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def in_memory_db() -> Database:
    """インメモリSQLiteデータベース。"""
    db = Database(db_path=":memory:")
    db.initialize()
    return db


@pytest.fixture
def draft_repo(in_memory_db: Database) -> DraftRepository:
    """下書きリポジトリ。"""
    return DraftRepository(in_memory_db)


@pytest.fixture
def history_repo(in_memory_db: Database) -> HistoryRepository:
    """投稿履歴リポジトリ。"""
    return HistoryRepository(in_memory_db)


@pytest.fixture
def hearing_service(mock_llm: AsyncMock) -> HearingService:
    """ヒアリングサービス。"""
    return HearingService(mock_llm)


@pytest.fixture
def article_service(mock_llm: AsyncMock) -> ArticleService:
    """記事生成サービス。"""
    return ArticleService(mock_llm)


@pytest.fixture
def draft_service(draft_repo: DraftRepository) -> DraftService:
    """下書きサービス。"""
    return DraftService(draft_repo)


@pytest.fixture
def history_service(history_repo: HistoryRepository) -> HistoryService:
    """投稿履歴サービス。"""
    return HistoryService(history_repo)


@pytest.fixture
def publish_service() -> PublishService:
    """投稿サービス。"""
    return PublishService()


@pytest.fixture
def async_runner() -> AsyncRunner:
    """非同期ランナー（テスト用に起動・停止を管理）。"""
    runner = AsyncRunner()
    runner.start()
    yield runner
    runner.stop()


@pytest.fixture
def tmp_dir() -> Path:
    """テスト用の一時ディレクトリ。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
