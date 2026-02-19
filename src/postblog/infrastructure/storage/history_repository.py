"""投稿履歴リポジトリモジュール。

SQLiteを使用した投稿履歴のCRUD操作を提供する。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from postblog.infrastructure.storage.database import Database


logger = logging.getLogger(__name__)


@dataclass
class HistoryRecord:
    """投稿履歴レコード。

    Args:
        id: レコードID。
        title: 記事タイトル。
        body_preview: 本文プレビュー（先頭200文字）。
        blog_type_id: ブログ種別ID。
        service_name: サービス名。
        article_url: 記事URL。
        status: ステータス（"published" | "draft" | "failed"）。
        published_at: 投稿日時。
        created_at: レコード作成日時。
    """

    id: int | None = None
    title: str = ""
    body_preview: str = ""
    blog_type_id: str = ""
    service_name: str = ""
    article_url: str | None = None
    status: str = "published"
    published_at: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


class HistoryRepository:
    """投稿履歴のCRUD操作を提供する。

    Args:
        database: データベース接続管理オブジェクト。
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    def save(self, record: HistoryRecord) -> HistoryRecord:
        """投稿履歴を保存する。

        Args:
            record: 保存する投稿履歴レコード。

        Returns:
            保存後のレコード（IDが設定される）。
        """
        conn = self._db.get_connection()
        now = datetime.now().isoformat()

        cursor = conn.execute(
            """INSERT INTO publish_history
               (title, body_preview, blog_type_id, service_name, article_url, status, published_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.title,
                record.body_preview,
                record.blog_type_id,
                record.service_name,
                record.article_url,
                record.status,
                record.published_at.isoformat() if record.published_at else now,
                now,
            ),
        )
        conn.commit()
        record.id = cursor.lastrowid
        logger.info(
            "投稿履歴を保存しました: id=%s, service=%s", record.id, record.service_name
        )
        return record

    def find_all(self) -> list[HistoryRecord]:
        """全ての投稿履歴を取得する（投稿日時の降順）。

        Returns:
            投稿履歴レコードのリスト。
        """
        conn = self._db.get_connection()
        rows = conn.execute(
            "SELECT * FROM publish_history ORDER BY published_at DESC"
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_by_id(self, record_id: int) -> HistoryRecord | None:
        """IDで投稿履歴を検索する。

        Args:
            record_id: レコードID。

        Returns:
            投稿履歴レコード。見つからない場合はNone。
        """
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT * FROM publish_history WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def delete(self, record_id: int) -> bool:
        """投稿履歴を削除する。

        Args:
            record_id: 削除するレコードのID。

        Returns:
            削除に成功した場合True。
        """
        conn = self._db.get_connection()
        cursor = conn.execute("DELETE FROM publish_history WHERE id = ?", (record_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("投稿履歴を削除しました: id=%s", record_id)
        return deleted

    @staticmethod
    def _row_to_record(row: object) -> HistoryRecord:
        """データベースの行をHistoryRecordオブジェクトに変換する。

        Args:
            row: sqlite3.Rowオブジェクト。

        Returns:
            HistoryRecordインスタンス。
        """
        published_at = (
            datetime.fromisoformat(row["published_at"])  # type: ignore[index]
            if row["published_at"]  # type: ignore[index]
            else datetime.now()
        )
        created_at = (
            datetime.fromisoformat(row["created_at"])  # type: ignore[index]
            if row["created_at"]  # type: ignore[index]
            else datetime.now()
        )

        return HistoryRecord(
            id=row["id"],  # type: ignore[index]
            title=row["title"] or "",  # type: ignore[index]
            body_preview=row["body_preview"] or "",  # type: ignore[index]
            blog_type_id=row["blog_type_id"] or "",  # type: ignore[index]
            service_name=row["service_name"] or "",  # type: ignore[index]
            article_url=row["article_url"],  # type: ignore[index]
            status=row["status"] or "published",  # type: ignore[index]
            published_at=published_at,
            created_at=created_at,
        )
