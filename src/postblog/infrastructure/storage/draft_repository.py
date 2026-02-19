"""下書きリポジトリモジュール。

SQLiteを使用した下書きのCRUD操作を提供する。
"""

import json
import logging
from datetime import datetime

from postblog.infrastructure.storage.database import Database
from postblog.models.draft import Draft


logger = logging.getLogger(__name__)


class DraftRepository:
    """下書きのCRUD操作を提供する。

    Args:
        database: データベース接続管理オブジェクト。
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    def save(self, draft: Draft) -> Draft:
        """下書きを保存する。新規の場合はINSERT、既存の場合はUPDATE。

        Args:
            draft: 保存する下書き。

        Returns:
            保存後の下書き（IDが設定される）。
        """
        conn = self._db.get_connection()
        now = datetime.now().isoformat()
        tags_json = json.dumps(draft.tags, ensure_ascii=False)

        if draft.id is None:
            cursor = conn.execute(
                """INSERT INTO drafts (title, body, tags, blog_type_id, hearing_data, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    draft.title,
                    draft.body,
                    tags_json,
                    draft.blog_type_id,
                    draft.hearing_data,
                    now,
                    now,
                ),
            )
            conn.commit()
            draft.id = cursor.lastrowid
            logger.info("下書きを新規作成しました: id=%s", draft.id)
        else:
            conn.execute(
                """UPDATE drafts SET title=?, body=?, tags=?, blog_type_id=?, hearing_data=?, updated_at=?
                   WHERE id=?""",
                (
                    draft.title,
                    draft.body,
                    tags_json,
                    draft.blog_type_id,
                    draft.hearing_data,
                    now,
                    draft.id,
                ),
            )
            conn.commit()
            logger.info("下書きを更新しました: id=%s", draft.id)

        return draft

    def find_by_id(self, draft_id: int) -> Draft | None:
        """IDで下書きを検索する。

        Args:
            draft_id: 下書きID。

        Returns:
            下書き。見つからない場合はNone。
        """
        conn = self._db.get_connection()
        row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_draft(row)

    def find_all(self) -> list[Draft]:
        """全ての下書きを取得する（更新日時の降順）。

        Returns:
            下書きのリスト。
        """
        conn = self._db.get_connection()
        rows = conn.execute("SELECT * FROM drafts ORDER BY updated_at DESC").fetchall()
        return [self._row_to_draft(row) for row in rows]

    def delete(self, draft_id: int) -> bool:
        """下書きを削除する。

        Args:
            draft_id: 削除する下書きのID。

        Returns:
            削除に成功した場合True。
        """
        conn = self._db.get_connection()
        cursor = conn.execute("DELETE FROM drafts WHERE id = ?", (draft_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("下書きを削除しました: id=%s", draft_id)
        return deleted

    @staticmethod
    def _row_to_draft(row: object) -> Draft:
        """データベースの行をDraftオブジェクトに変換する。

        Args:
            row: sqlite3.Rowオブジェクト。

        Returns:
            Draftインスタンス。
        """
        tags = json.loads(row["tags"]) if row["tags"] else []  # type: ignore[index]
        created_at = (
            datetime.fromisoformat(row["created_at"])  # type: ignore[index]
            if row["created_at"]  # type: ignore[index]
            else datetime.now()
        )
        updated_at = (
            datetime.fromisoformat(row["updated_at"])  # type: ignore[index]
            if row["updated_at"]  # type: ignore[index]
            else datetime.now()
        )

        return Draft(
            id=row["id"],  # type: ignore[index]
            title=row["title"] or "",  # type: ignore[index]
            body=row["body"] or "",  # type: ignore[index]
            tags=tags,
            blog_type_id=row["blog_type_id"] or "",  # type: ignore[index]
            hearing_data=row["hearing_data"] or "",  # type: ignore[index]
            created_at=created_at,
            updated_at=updated_at,
        )
