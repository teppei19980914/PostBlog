"""Zenn投稿クライアント。

GitHub連携によるZenn記事投稿（Zenn CLIリポジトリへのpush）。
"""

import logging
from pathlib import Path

import git

from postblog.infrastructure.publishers.base import BlogPublisher
from postblog.models.publish_result import PublishRequest, PublishResult


logger = logging.getLogger(__name__)


class ZennPublisher(BlogPublisher):
    """Zennへの記事投稿クライアント。

    Args:
        repo_path: Zenn CLIリポジトリのローカルパス。
        github_token: GitHubアクセストークン。
    """

    def __init__(self, repo_path: str, github_token: str) -> None:
        self._repo_path = Path(repo_path)
        self._github_token = github_token

    @property
    def service_name(self) -> str:
        """サービス名を返す。"""
        return "zenn"

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Zennに記事を投稿する（GitHubリポジトリにpush）。

        Args:
            request: 投稿リクエスト。

        Returns:
            投稿結果。
        """
        try:
            articles_dir = self._repo_path / "articles"
            articles_dir.mkdir(parents=True, exist_ok=True)

            # スラッグ生成（簡易版）
            slug = request.title.lower().replace(" ", "-")[:50]
            article_path = articles_dir / f"{slug}.md"

            # フロントマター生成
            tags_str = "\n".join(f'  - "{tag}"' for tag in request.tags[:5])
            published = "true" if request.status == "publish" else "false"
            content = f"""---
title: "{request.title}"
emoji: "📝"
type: "tech"
topics:
{tags_str}
published: {published}
---

{request.body}
"""
            article_path.write_text(content, encoding="utf-8")

            # Git操作
            repo = git.Repo(self._repo_path)
            repo.index.add([str(article_path.relative_to(self._repo_path))])
            repo.index.commit(f"Add article: {request.title}")
            repo.remotes.origin.push()

            logger.info("Zennに投稿しました: %s", article_path.name)
            return PublishResult(
                success=True,
                service_name=self.service_name,
                article_url=f"https://zenn.dev/articles/{slug}",
            )
        except Exception as e:
            logger.error("Zenn投稿に失敗しました: %s", e)
            return PublishResult(
                success=False,
                service_name=self.service_name,
                error_message=str(e),
            )

    async def test_connection(self) -> bool:
        """接続テストを実行する。"""
        try:
            repo = git.Repo(self._repo_path)
            return repo.remotes.origin.exists()
        except Exception:
            logger.exception("Zenn接続テストに失敗しました")
            return False
