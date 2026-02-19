"""投稿履歴サービス。"""

import logging

from postblog.infrastructure.storage.history_repository import (
    HistoryRecord,
    HistoryRepository,
)


logger = logging.getLogger(__name__)


class HistoryService:
    """投稿履歴の管理を提供するサービス。

    Args:
        repository: 投稿履歴リポジトリ。
    """

    def __init__(self, repository: HistoryRepository) -> None:
        self._repo = repository

    def save(self, record: HistoryRecord) -> HistoryRecord:
        """投稿履歴を保存する。

        Args:
            record: 保存する投稿履歴。

        Returns:
            保存後のレコード。
        """
        saved = self._repo.save(record)
        logger.info(
            "投稿履歴を保存しました: id=%s, service=%s", saved.id, saved.service_name
        )
        return saved

    def get_all(self) -> list[HistoryRecord]:
        """全ての投稿履歴を取得する。

        Returns:
            投稿履歴のリスト。
        """
        return self._repo.find_all()

    def get(self, record_id: int) -> HistoryRecord | None:
        """投稿履歴を取得する。

        Args:
            record_id: レコードID。

        Returns:
            投稿履歴。見つからない場合はNone。
        """
        return self._repo.find_by_id(record_id)

    def delete(self, record_id: int) -> bool:
        """投稿履歴を削除する。

        Args:
            record_id: 削除するレコードID。

        Returns:
            削除に成功した場合True。
        """
        result = self._repo.delete(record_id)
        if result:
            logger.info("投稿履歴を削除しました: id=%s", record_id)
        return result
