"""設定コントローラのテスト。"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from postblog.config import AppConfig, ConfigManager
from postblog.controllers.settings_controller import SettingsController
from postblog.exceptions import ValidationError


class TestGetAppSettings:
    """get_app_settings メソッドのテスト。"""

    def _create_controller(self, config: AppConfig | None = None) -> SettingsController:
        """テスト用コントローラを作成する。"""
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.config = config or AppConfig()
        config_manager.config_path = Path("~/.postblog/config.toml")
        credential_manager = MagicMock()
        publish_service = MagicMock()
        async_runner = MagicMock()
        return SettingsController(
            config_manager, credential_manager, publish_service, async_runner
        )

    def test_returns_default_settings(self) -> None:
        """デフォルト設定が返されることを確認する。"""
        controller = self._create_controller()

        result = controller.get_app_settings()

        assert result["theme"] == "dark"
        assert result["font_size"] == 14
        assert result["auto_save_interval"] == 30
        assert result["model"] == "gpt-4o"
        assert result["preview_position"] == "right"

    def test_returns_custom_settings(self) -> None:
        """カスタム設定が返されることを確認する。"""
        config = AppConfig(
            theme="light",
            font_size=16,
            auto_save_interval=60,
            model="gpt-4o-mini",
            preview_position="bottom",
        )
        controller = self._create_controller(config)

        result = controller.get_app_settings()

        assert result["theme"] == "light"
        assert result["font_size"] == 16
        assert result["auto_save_interval"] == 60


class TestUpdateAppSettings:
    """update_app_settings メソッドのテスト。"""

    def _create_controller(self) -> tuple[SettingsController, MagicMock]:
        """テスト用コントローラと設定管理のモックを返す。"""
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.config = AppConfig()
        config_manager.config_path = Path("~/.postblog/config.toml")
        config_manager.update.return_value = AppConfig()
        credential_manager = MagicMock()
        publish_service = MagicMock()
        async_runner = MagicMock()
        controller = SettingsController(
            config_manager, credential_manager, publish_service, async_runner
        )
        return controller, config_manager

    def test_update_valid_settings(self) -> None:
        """有効な設定更新が成功することを確認する。"""
        controller, config_manager = self._create_controller()

        controller.update_app_settings(theme="light")

        config_manager.update.assert_called_once_with(theme="light")
        config_manager.save.assert_called_once()

    def test_update_invalid_font_size_raises_error(self) -> None:
        """不正なフォントサイズでValidationErrorが発生することを確認する。"""
        controller, _ = self._create_controller()

        with pytest.raises(ValidationError, match="フォントサイズは8-32"):
            controller.update_app_settings(font_size=5)

    def test_update_invalid_font_size_too_large(self) -> None:
        """大きすぎるフォントサイズでValidationErrorが発生することを確認する。"""
        controller, _ = self._create_controller()

        with pytest.raises(ValidationError, match="フォントサイズは8-32"):
            controller.update_app_settings(font_size=50)

    def test_update_invalid_auto_save_interval(self) -> None:
        """不正な自動保存間隔でValidationErrorが発生することを確認する。"""
        controller, _ = self._create_controller()

        with pytest.raises(ValidationError, match="自動保存間隔は10-300秒"):
            controller.update_app_settings(auto_save_interval=5)

    def test_update_invalid_theme(self) -> None:
        """不正なテーマでValidationErrorが発生することを確認する。"""
        controller, _ = self._create_controller()

        with pytest.raises(ValidationError, match="テーマは"):
            controller.update_app_settings(theme="blue")

    def test_update_invalid_preview_position(self) -> None:
        """不正なプレビュー位置でValidationErrorが発生することを確認する。"""
        controller, _ = self._create_controller()

        with pytest.raises(ValidationError, match="プレビュー位置は"):
            controller.update_app_settings(preview_position="left")

    def test_update_valid_font_size_boundary(self) -> None:
        """境界値のフォントサイズが通ることを確認する。"""
        controller, config_manager = self._create_controller()

        controller.update_app_settings(font_size=8)
        config_manager.update.assert_called_with(font_size=8)

        controller.update_app_settings(font_size=32)


class TestApiKeyManagement:
    """APIキー管理のテスト。"""

    def _create_controller(self) -> tuple[SettingsController, MagicMock]:
        """テスト用コントローラとcredential_managerのモックを返す。"""
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.config = AppConfig()
        config_manager.config_path = Path("~/.postblog/config.toml")
        credential_manager = MagicMock()
        publish_service = MagicMock()
        async_runner = MagicMock()
        controller = SettingsController(
            config_manager, credential_manager, publish_service, async_runner
        )
        return controller, credential_manager

    def test_get_api_key_display_when_set(self) -> None:
        """APIキー設定時にマスク表示されることを確認する。"""
        controller, credential_manager = self._create_controller()
        credential_manager.retrieve.return_value = "sk-abc123456789"

        result = controller.get_api_key_display()

        assert "..." in result
        assert "sk-" not in result or len(result) < len("sk-abc123456789")

    def test_get_api_key_display_when_not_set(self) -> None:
        """APIキー未設定時に空文字列が返されることを確認する。"""
        controller, credential_manager = self._create_controller()
        credential_manager.retrieve.return_value = None

        result = controller.get_api_key_display()

        assert result == ""

    def test_save_api_key(self) -> None:
        """APIキー保存が成功することを確認する。"""
        controller, credential_manager = self._create_controller()

        controller.save_api_key("sk-test123")

        credential_manager.store.assert_called_once_with(
            "openai", "api_key", "sk-test123"
        )

    def test_save_empty_api_key_raises_error(self) -> None:
        """空のAPIキーでValidationErrorが発生することを確認する。"""
        controller, _ = self._create_controller()

        with pytest.raises(ValidationError, match="APIキーを入力してください"):
            controller.save_api_key("")

    def test_save_whitespace_api_key_raises_error(self) -> None:
        """空白のみのAPIキーでValidationErrorが発生することを確認する。"""
        controller, _ = self._create_controller()

        with pytest.raises(ValidationError, match="APIキーを入力してください"):
            controller.save_api_key("   ")

    def test_delete_api_key(self) -> None:
        """APIキー削除が成功することを確認する。"""
        controller, credential_manager = self._create_controller()
        credential_manager.delete.return_value = True

        result = controller.delete_api_key()

        assert result is True
        credential_manager.delete.assert_called_once_with("openai", "api_key")


class TestServiceCredentialManagement:
    """サービス認証情報管理のテスト。"""

    def _create_controller(self) -> tuple[SettingsController, MagicMock]:
        """テスト用コントローラとcredential_managerのモックを返す。"""
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.config = AppConfig()
        config_manager.config_path = Path("~/.postblog/config.toml")
        credential_manager = MagicMock()
        publish_service = MagicMock()
        async_runner = MagicMock()
        controller = SettingsController(
            config_manager, credential_manager, publish_service, async_runner
        )
        return controller, credential_manager

    def test_save_service_credential(self) -> None:
        """サービス認証情報の保存が成功することを確認する。"""
        controller, credential_manager = self._create_controller()

        controller.save_service_credential("qiita", "api_token", "token123")

        credential_manager.store.assert_called_once_with(
            "qiita", "api_token", "token123"
        )

    def test_save_empty_credential_raises_error(self) -> None:
        """空の認証情報でValidationErrorが発生することを確認する。"""
        controller, _ = self._create_controller()

        with pytest.raises(ValidationError, match="認証情報を入力してください"):
            controller.save_service_credential("qiita", "api_token", "")

    def test_get_service_credential_display(self) -> None:
        """サービス認証情報のマスク表示を確認する。"""
        controller, credential_manager = self._create_controller()
        credential_manager.retrieve.return_value = "long-token-value-12345"

        result = controller.get_service_credential_display("qiita", "api_token")

        assert "..." in result

    def test_get_service_credential_display_when_not_set(self) -> None:
        """未設定時に空文字列が返されることを確認する。"""
        controller, credential_manager = self._create_controller()
        credential_manager.retrieve.return_value = None

        result = controller.get_service_credential_display("qiita", "api_token")

        assert result == ""

    def test_delete_service_credential(self) -> None:
        """サービス認証情報の削除が成功することを確認する。"""
        controller, credential_manager = self._create_controller()
        credential_manager.delete.return_value = True

        result = controller.delete_service_credential("qiita", "api_token")

        assert result is True


class TestTestConnection:
    """test_connection メソッドのテスト。"""

    def test_test_connection_calls_async_runner(self) -> None:
        """接続テストがAsyncRunnerを経由することを確認する。"""
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.config = AppConfig()
        config_manager.config_path = Path("~/.postblog/config.toml")
        credential_manager = MagicMock()
        publish_service = MagicMock()
        async_runner = MagicMock()
        controller = SettingsController(
            config_manager, credential_manager, publish_service, async_runner
        )

        controller.test_connection("Qiita")

        async_runner.run.assert_called_once()


class TestResetToDefaults:
    """reset_to_defaults メソッドのテスト。"""

    def test_reset_to_defaults(self) -> None:
        """デフォルト値にリセットされることを確認する。"""
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.config = AppConfig(theme="light", font_size=20)
        config_manager.config_path = Path("~/.postblog/config.toml")
        # loadの後にconfigプロパティがデフォルトを返すようにする
        config_manager.load.side_effect = lambda: setattr(
            config_manager, "config", AppConfig()
        )
        credential_manager = MagicMock()
        publish_service = MagicMock()
        async_runner = MagicMock()
        controller = SettingsController(
            config_manager, credential_manager, publish_service, async_runner
        )

        result = controller.reset_to_defaults()

        config_manager.save.assert_called_once()
        config_manager.load.assert_called_once()
        assert result["theme"] == "dark"
        assert result["font_size"] == 14


class TestGetDataLocation:
    """get_data_location メソッドのテスト。"""

    def test_returns_parent_of_config_path(self) -> None:
        """設定ファイルの親ディレクトリが返されることを確認する。"""
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.config = AppConfig()
        config_manager.config_path = Path("/home/user/.postblog/config.toml")
        credential_manager = MagicMock()
        publish_service = MagicMock()
        async_runner = MagicMock()
        controller = SettingsController(
            config_manager, credential_manager, publish_service, async_runner
        )

        result = controller.get_data_location()

        assert ".postblog" in result


class TestValidateSettings:
    """_validate_settings 静的メソッドのテスト。"""

    def test_valid_settings(self) -> None:
        """有効な設定値が通ることを確認する。"""
        SettingsController._validate_settings(
            {
                "font_size": 14,
                "auto_save_interval": 30,
                "theme": "dark",
                "preview_position": "right",
            }
        )

    def test_font_size_string_raises_error(self) -> None:
        """フォントサイズが文字列の場合にValidationErrorが発生することを確認する。"""
        with pytest.raises(ValidationError, match="フォントサイズは8-32"):
            SettingsController._validate_settings({"font_size": "large"})  # type: ignore[dict-item]

    def test_auto_save_interval_too_large(self) -> None:
        """大きすぎる自動保存間隔でValidationErrorが発生することを確認する。"""
        with pytest.raises(ValidationError, match="自動保存間隔は10-300秒"):
            SettingsController._validate_settings({"auto_save_interval": 500})

    def test_empty_settings_passes(self) -> None:
        """空の設定辞書が通ることを確認する。"""
        SettingsController._validate_settings({})
