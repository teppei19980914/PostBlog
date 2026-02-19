"""データベース接続管理のテスト。"""

from postblog.infrastructure.storage.database import Database


class TestDatabase:
    """Databaseのテスト。"""

    def test_connect_in_memory(self) -> None:
        """インメモリDBに接続できることを確認する。"""
        db = Database(":memory:")
        conn = db.connect()

        assert conn is not None

        db.close()

    def test_initialize_creates_tables(self) -> None:
        """スキーマ初期化でテーブルが作成されることを確認する。"""
        db = Database(":memory:")
        db.initialize()

        conn = db.get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [row["name"] for row in tables]

        assert "drafts" in table_names
        assert "publish_history" in table_names

        db.close()

    def test_get_connection_auto_connects(self) -> None:
        """get_connectionが自動接続することを確認する。"""
        db = Database(":memory:")
        conn = db.get_connection()

        assert conn is not None

        db.close()

    def test_close_and_reconnect(self) -> None:
        """接続を閉じて再接続できることを確認する。"""
        db = Database(":memory:")
        db.connect()
        db.close()

        conn = db.connect()
        assert conn is not None

        db.close()

    def test_connect_returns_same_connection(self) -> None:
        """同じ接続が返されることを確認する。"""
        db = Database(":memory:")
        conn1 = db.connect()
        conn2 = db.connect()

        assert conn1 is conn2

        db.close()

    def test_db_path(self) -> None:
        """DBパスが正しいことを確認する。"""
        db = Database(":memory:")
        assert db.db_path == ":memory:"

    def test_file_db_creates_directory(self, tmp_dir) -> None:
        """ファイルDBがディレクトリを自動作成することを確認する。"""
        db_path = tmp_dir / "nested" / "dir" / "test.db"
        db = Database(db_path)
        db.connect()

        assert db_path.parent.exists()

        db.close()
