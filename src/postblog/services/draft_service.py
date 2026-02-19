"""下書き管理サービス。"""

import logging

from postblog.infrastructure.storage.draft_repository import DraftRepository
from postblog.models.draft import Draft


logger = logging.getLogger(__name__)


class DraftService:
    """下書きの管理を提供するサービス。

    Args:
        repository: 下書きリポジトリ。
    """

    def __init__(self, repository: DraftRepository) -> None:
        self._repo = repository

    def save(self, draft: Draft) -> Draft:
        """下書きを保存する。

        Args:
            draft: 保存する下書き。

        Returns:
            保存後の下書き。
        """
        saved = self._repo.save(draft)
        logger.info("下書きを保存しました: id=%s, title=%s", saved.id, saved.title)
        return saved

    def get(self, draft_id: int) -> Draft | None:
        """下書きを取得する。

        Args:
            draft_id: 下書きID。

        Returns:
            下書き。見つからない場合はNone。
        """
        return self._repo.find_by_id(draft_id)

    def get_all(self) -> list[Draft]:
        """全ての下書きを取得する。

        Returns:
            下書きのリスト。
        """
        return self._repo.find_all()

    def delete(self, draft_id: int) -> bool:
        """下書きを削除する。

        Args:
            draft_id: 削除する下書きID。

        Returns:
            削除に成功した場合True。
        """
        result = self._repo.delete(draft_id)
        if result:
            logger.info("下書きを削除しました: id=%s", draft_id)
        return result
