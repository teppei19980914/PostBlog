"""SQLiteデータベース接続管理モジュール。

~/.postblog/postblog.db にSQLiteデータベースを作成・管理する。
"""

import logging
import sqlite3
from pathlib import Path


logger = logging.getLogger(__name__)

# デフォルトDBパス
DEFAULT_DB_PATH = Path.home() / ".postblog" / "postblog.db"

# スキーマ定義
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    blog_type_id TEXT NOT NULL DEFAULT '',
    hearing_data TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS publish_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    body_preview TEXT NOT NULL DEFAULT '',
    blog_type_id TEXT NOT NULL DEFAULT '',
    service_name TEXT NOT NULL DEFAULT '',
    article_url TEXT DEFAULT NULL,
    status TEXT NOT NULL DEFAULT 'published',
    published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    """SQLiteデータベース接続管理クラス。

    Args:
        db_path: データベースファイルのパス（":memory:" でインメモリDB）。
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._db_path = str(db_path)
        self._connection: sqlite3.Connection | None = None

    @property
    def db_path(self) -> str:
        """データベースファイルのパスを返す。"""
        return self._db_path

    def connect(self) -> sqlite3.Connection:
        """データベースに接続する。

        Returns:
            SQLite接続オブジェクト。
        """
        if self._connection is not None:
            return self._connection

        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self._db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        logger.info("データベースに接続しました: %s", self._db_path)
        return self._connection

    def initialize(self) -> None:
        """スキーマを初期化する。"""
        conn = self.connect()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("データベーススキーマを初期化しました")

    def close(self) -> None:
        """データベース接続を閉じる。"""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("データベース接続を閉じました")

    def get_connection(self) -> sqlite3.Connection:
        """現在の接続を取得する。未接続の場合は接続する。

        Returns:
            SQLite接続オブジェクト。
        """
        if self._connection is None:
            return self.connect()
        return self._connection
