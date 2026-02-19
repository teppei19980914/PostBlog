"""設定管理モジュールのテスト。"""

from pathlib import Path

from postblog.config import AppConfig, ConfigManager


class TestAppConfig:
    """AppConfigのテスト。"""

    def test_default_values(self) -> None:
        """デフォルト値が正しいことを確認する。"""
        config = AppConfig()

        assert config.theme == "dark"
        assert config.font_size == 14
        assert config.auto_save_interval == 30
        assert config.model == "gpt-4o"
        assert config.preview_position == "right"

    def test_custom_values(self) -> None:
        """カスタム値で作成できることを確認する。"""
        config = AppConfig(
            theme="light",
            font_size=16,
            auto_save_interval=60,
            model="gpt-4o-mini",
            preview_position="bottom",
        )

        assert config.theme == "light"
        assert config.font_size == 16
        assert config.auto_save_interval == 60
        assert config.model == "gpt-4o-mini"
        assert config.preview_position == "bottom"

    def test_to_dict(self) -> None:
        """辞書変換が正しいことを確認する。"""
        config = AppConfig(theme="light", font_size=16, model="gpt-4o-mini")
        data = config.to_dict()

        assert data["app"]["theme"] == "light"
        assert data["app"]["font_size"] == 16
        assert data["openai"]["model"] == "gpt-4o-mini"
        assert data["editor"]["preview_position"] == "right"

    def test_from_dict(self) -> None:
        """辞書からの生成が正しいことを確認する。"""
        data = {
            "app": {"theme": "light", "font_size": 18, "auto_save_interval": 45},
            "openai": {"model": "gpt-4o-mini"},
            "editor": {"preview_position": "bottom"},
        }
        config = AppConfig.from_dict(data)

        assert config.theme == "light"
        assert config.font_size == 18
        assert config.auto_save_interval == 45
        assert config.model == "gpt-4o-mini"
        assert config.preview_position == "bottom"

    def test_from_dict_partial(self) -> None:
        """部分的な辞書からデフォルト値が使われることを確認する。"""
        data = {"app": {"theme": "light"}}
        config = AppConfig.from_dict(data)

        assert config.theme == "light"
        assert config.font_size == 14  # default
        assert config.model == "gpt-4o"  # default

    def test_from_dict_empty(self) -> None:
        """空の辞書からデフォルト設定が作られることを確認する。"""
        config = AppConfig.from_dict({})

        assert config.theme == "dark"
        assert config.font_size == 14

    def test_roundtrip(self) -> None:
        """to_dict -> from_dict のラウンドトリップを確認する。"""
        original = AppConfig(theme="light", font_size=20, model="gpt-3.5-turbo")
        restored = AppConfig.from_dict(original.to_dict())

        assert restored.theme == original.theme
        assert restored.font_size == original.font_size
        assert restored.auto_save_interval == original.auto_save_interval
        assert restored.model == original.model
        assert restored.preview_position == original.preview_position


class TestConfigManager:
    """ConfigManagerのテスト。"""

    def test_default_config(self, tmp_dir: Path) -> None:
        """デフォルト設定が返されることを確認する。"""
        manager = ConfigManager(config_path=tmp_dir / "config.toml")

        assert manager.config.theme == "dark"
        assert manager.config.model == "gpt-4o"

    def test_config_path(self, tmp_dir: Path) -> None:
        """設定ファイルのパスが正しいことを確認する。"""
        path = tmp_dir / "config.toml"
        manager = ConfigManager(config_path=path)

        assert manager.config_path == path

    def test_save_and_load(self, tmp_dir: Path) -> None:
        """設定の保存と読み込みが正しいことを確認する。"""
        path = tmp_dir / "config.toml"

        # 保存
        manager1 = ConfigManager(config_path=path)
        config = AppConfig(theme="light", font_size=18, model="gpt-4o-mini")
        manager1.save(config)

        assert path.exists()

        # 読み込み
        manager2 = ConfigManager(config_path=path)
        loaded = manager2.load()

        assert loaded.theme == "light"
        assert loaded.font_size == 18
        assert loaded.model == "gpt-4o-mini"

    def test_load_nonexistent_returns_default(self, tmp_dir: Path) -> None:
        """存在しないファイルの読み込みでデフォルト設定が返されることを確認する。"""
        manager = ConfigManager(config_path=tmp_dir / "nonexistent.toml")
        config = manager.load()

        assert config.theme == "dark"
        assert config.font_size == 14

    def test_save_creates_directory(self, tmp_dir: Path) -> None:
        """保存時にディレクトリが自動作成されることを確認する。"""
        path = tmp_dir / "nested" / "dir" / "config.toml"
        manager = ConfigManager(config_path=path)
        manager.save()

        assert path.exists()

    def test_save_current_config(self, tmp_dir: Path) -> None:
        """現在の設定が保存されることを確認する。"""
        path = tmp_dir / "config.toml"
        manager = ConfigManager(config_path=path)
        manager.update(theme="light")
        manager.save()

        manager2 = ConfigManager(config_path=path)
        loaded = manager2.load()
        assert loaded.theme == "light"

    def test_update(self, tmp_dir: Path) -> None:
        """設定を部分更新できることを確認する。"""
        manager = ConfigManager(config_path=tmp_dir / "config.toml")

        updated = manager.update(theme="light", font_size=20)

        assert updated.theme == "light"
        assert updated.font_size == 20
        assert updated.model == "gpt-4o"  # 変更なし

    def test_update_unknown_key_ignored(self, tmp_dir: Path) -> None:
        """不明なキーの更新が無視されることを確認する。"""
        manager = ConfigManager(config_path=tmp_dir / "config.toml")

        updated = manager.update(unknown_key="value")

        assert updated.theme == "dark"  # 変更なし

    def test_load_corrupted_file_returns_default(self, tmp_dir: Path) -> None:
        """破損したファイルの読み込みでデフォルト設定が返されることを確認する。"""
        path = tmp_dir / "config.toml"
        path.write_text("invalid toml {{{", encoding="utf-8")

        manager = ConfigManager(config_path=path)
        config = manager.load()

        assert config.theme == "dark"
        assert config.font_size == 14
