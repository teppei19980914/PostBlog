"""ホーム画面コントローラ。

下書き一覧と投稿履歴一覧を管理する。
"""

import logging

from postblog.services.draft_service import DraftService
from postblog.services.history_service import HistoryService


logger = logging.getLogger(__name__)


class HomeController:
    """ホーム画面のロジックを管理するコントローラ。

    Args:
        draft_service: 下書き管理サービス。
        history_service: 投稿履歴管理サービス。
    """

    def __init__(
        self,
        draft_service: DraftService,
        history_service: HistoryService,
    ) -> None:
        self._draft_service = draft_service
        self._history_service = history_service

    def get_recent_drafts(self, limit: int = 10) -> list[dict[str, str]]:
        """最近の下書き一覧を取得する。

        Args:
            limit: 取得件数上限。

        Returns:
            下書き情報の辞書リスト。
        """
        try:
            drafts = self._draft_service.get_all()
            sorted_drafts = sorted(drafts, key=lambda d: d.updated_at, reverse=True)[
                :limit
            ]
            return [
                {
                    "id": str(d.id) if d.id is not None else "",
                    "title": d.title or "無題",
                    "blog_type_id": d.blog_type_id,
                    "updated_at": d.updated_at.isoformat(),
                    "preview": d.body[:100] + "..." if len(d.body) > 100 else d.body,
                }
                for d in sorted_drafts
            ]
        except Exception:
            logger.exception("下書き一覧の取得に失敗しました")
            return []

    def get_recent_history(self, limit: int = 10) -> list[dict[str, str | None]]:
        """最近の投稿履歴一覧を取得する。

        Args:
            limit: 取得件数上限。

        Returns:
            投稿履歴情報の辞書リスト。
        """
        try:
            records = self._history_service.get_all()
            return [
                {
                    "id": str(r.id) if r.id is not None else "",
                    "title": r.title or "無題",
                    "service_name": r.service_name,
                    "status": r.status,
                    "article_url": r.article_url,
                    "published_at": r.published_at.isoformat(),
                }
                for r in records[:limit]
            ]
        except Exception:
            logger.exception("投稿履歴の取得に失敗しました")
            return []

    def delete_draft(self, draft_id: int) -> bool:
        """下書きを削除する。

        Args:
            draft_id: 削除する下書きID。

        Returns:
            削除成功の場合True。
        """
        try:
            return self._draft_service.delete(draft_id)
        except Exception:
            logger.exception("下書きの削除に失敗しました: id=%s", draft_id)
            return False

    def delete_history(self, record_id: int) -> bool:
        """投稿履歴を削除する。

        Args:
            record_id: 削除するレコードID。

        Returns:
            削除成功の場合True。
        """
        try:
            return self._history_service.delete(record_id)
        except Exception:
            logger.exception("投稿履歴の削除に失敗しました: id=%s", record_id)
            return False
