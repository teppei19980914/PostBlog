"""Markdownエクスポートクライアント。

記事をMarkdownファイルとしてローカルに保存する。
note / Wantedly など、API非公開サービス向け。
"""

import logging
from pathlib import Path

from postblog.infrastructure.publishers.base import BlogPublisher
from postblog.models.publish_result import PublishRequest, PublishResult


logger = logging.getLogger(__name__)

# デフォルトのエクスポートディレクトリ
DEFAULT_EXPORT_DIR = Path.home() / ".postblog" / "exports"


class MarkdownExportPublisher(BlogPublisher):
    """Markdownファイルとしてエクスポートするクライアント。

    Args:
        export_dir: エクスポート先ディレクトリ。
        service_label: サービス名ラベル。
    """

    def __init__(
        self,
        export_dir: Path | str = DEFAULT_EXPORT_DIR,
        service_label: str = "markdown_export",
    ) -> None:
        self._export_dir = Path(export_dir)
        self._service_label = service_label

    @property
    def service_name(self) -> str:
        """サービス名を返す。"""
        return self._service_label

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Markdownファイルとしてエクスポートする。

        Args:
            request: 投稿リクエスト。

        Returns:
            投稿結果。
        """
        try:
            self._export_dir.mkdir(parents=True, exist_ok=True)

            # ファイル名生成
            safe_title = "".join(
                c if c.isalnum() or c in ("-", "_", " ") else "_" for c in request.title
            )[:100]
            file_path = self._export_dir / f"{safe_title}.md"

            # フロントマター付きMarkdown生成
            tags_str = ", ".join(request.tags)
            content = f"""---
title: "{request.title}"
tags: [{tags_str}]
---

{request.body}
"""
            file_path.write_text(content, encoding="utf-8")

            logger.info("Markdownをエクスポートしました: %s", file_path)
            return PublishResult(
                success=True,
                service_name=self.service_name,
                article_url=str(file_path),
            )
        except Exception as e:
            logger.error("Markdownエクスポートに失敗しました: %s", e)
            return PublishResult(
                success=False,
                service_name=self.service_name,
                error_message=str(e),
            )

    async def test_connection(self) -> bool:
        """接続テスト（エクスポートディレクトリの書き込み確認）。"""
        try:
            self._export_dir.mkdir(parents=True, exist_ok=True)
            test_file = self._export_dir / ".connection_test"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
            return True
        except Exception:
            logger.exception("Markdownエクスポート接続テストに失敗しました")
            return False
