"""設定コントローラ。

アプリケーション設定・APIキー管理・サービス接続テストを管理する。
"""

import logging
from typing import Any

from postblog.config import AppConfig, ConfigManager
from postblog.exceptions import ValidationError
from postblog.infrastructure.async_runner import AsyncRunner
from postblog.infrastructure.credential.credential_manager import CredentialManager
from postblog.services.publish_service import PublishService


logger = logging.getLogger(__name__)


class SettingsController:
    """アプリケーション設定を管理するコントローラ。

    Args:
        config_manager: 設定管理。
        credential_manager: 認証情報管理。
        publish_service: 投稿サービス（接続テスト用）。
        async_runner: 非同期ランナー。
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        credential_manager: CredentialManager,
        publish_service: PublishService,
        async_runner: AsyncRunner,
    ) -> None:
        self._config_manager = config_manager
        self._credential_manager = credential_manager
        self._publish_service = publish_service
        self._async_runner = async_runner

    def get_app_settings(self) -> dict[str, str | int]:
        """アプリケーション設定を取得する。

        Returns:
            設定値の辞書。
        """
        config = self._config_manager.config
        return {
            "theme": config.theme,
            "font_size": config.font_size,
            "auto_save_interval": config.auto_save_interval,
            "model": config.model,
            "preview_position": config.preview_position,
        }

    def update_app_settings(self, **kwargs: str | int) -> dict[str, str | int]:
        """アプリケーション設定を更新する。

        Args:
            **kwargs: 更新する設定値。

        Returns:
            更新後の設定値の辞書。

        Raises:
            ValidationError: 不正な設定値の場合。
        """
        self._validate_settings(kwargs)
        self._config_manager.update(**kwargs)
        self._config_manager.save()
        logger.info("アプリケーション設定を更新しました")
        return self.get_app_settings()

    def get_api_key_display(self) -> str:
        """OpenAI APIキーの表示用文字列を取得する。

        Returns:
            マスクされたAPIキー。未設定の場合は空文字列。
        """
        key = self._credential_manager.retrieve("openai", "api_key")
        if key is None:
            return ""
        return CredentialManager.mask(key)

    def save_api_key(self, api_key: str) -> None:
        """OpenAI APIキーを保存する。

        Args:
            api_key: 保存するAPIキー。

        Raises:
            ValidationError: APIキーが空の場合。
        """
        if not api_key.strip():
            raise ValidationError("APIキーを入力してください。")
        self._credential_manager.store("openai", "api_key", api_key.strip())
        logger.info("OpenAI APIキーを保存しました")

    def delete_api_key(self) -> bool:
        """OpenAI APIキーを削除する。

        Returns:
            削除成功の場合True。
        """
        result = self._credential_manager.delete("openai", "api_key")
        if result:
            logger.info("OpenAI APIキーを削除しました")
        return result

    def save_service_credential(
        self, service_name: str, credential_type: str, value: str
    ) -> None:
        """サービスの認証情報を保存する。

        Args:
            service_name: サービス名。
            credential_type: 認証情報の種類。
            value: 認証情報の値。

        Raises:
            ValidationError: 値が空の場合。
        """
        if not value.strip():
            raise ValidationError("認証情報を入力してください。")
        self._credential_manager.store(service_name, credential_type, value.strip())
        logger.info(
            "サービス認証情報を保存しました: service=%s, type=%s",
            service_name,
            credential_type,
        )

    def get_service_credential_display(
        self, service_name: str, credential_type: str
    ) -> str:
        """サービスの認証情報の表示用文字列を取得する。

        Args:
            service_name: サービス名。
            credential_type: 認証情報の種類。

        Returns:
            マスクされた認証情報。未設定の場合は空文字列。
        """
        value = self._credential_manager.retrieve(service_name, credential_type)
        if value is None:
            return ""
        return CredentialManager.mask(value)

    def delete_service_credential(
        self, service_name: str, credential_type: str
    ) -> bool:
        """サービスの認証情報を削除する。

        Args:
            service_name: サービス名。
            credential_type: 認証情報の種類。

        Returns:
            削除成功の場合True。
        """
        return self._credential_manager.delete(service_name, credential_type)

    def test_connection(
        self,
        service_name: str,
        on_success: Any = None,
        on_error: Any = None,
    ) -> None:
        """サービスの接続テストを実行する（非同期）。

        Args:
            service_name: サービス名。
            on_success: 成功時コールバック。
            on_error: 失敗時コールバック。
        """

        async def _test() -> bool:
            return await self._publish_service.test_connection(service_name)

        self._async_runner.run(_test(), on_success=on_success, on_error=on_error)

    def reset_to_defaults(self) -> dict[str, str | int]:
        """設定をデフォルト値にリセットする。

        Returns:
            リセット後の設定値の辞書。
        """
        default_config = AppConfig()
        self._config_manager.save(default_config)
        # 現在のconfig_managerを再読み込み
        self._config_manager.load()
        logger.info("アプリケーション設定をデフォルトにリセットしました")
        return self.get_app_settings()

    def get_data_location(self) -> str:
        """データ保存ディレクトリのパスを取得する。

        Returns:
            データ保存ディレクトリの文字列。
        """
        return str(self._config_manager.config_path.parent)

    @staticmethod
    def _validate_settings(settings: dict[str, str | int]) -> None:
        """設定値を検証する。

        Args:
            settings: 検証する設定値。

        Raises:
            ValidationError: 不正な設定値の場合。
        """
        if "font_size" in settings:
            font_size = settings["font_size"]
            if not isinstance(font_size, int) or not (8 <= font_size <= 32):
                raise ValidationError("フォントサイズは8-32の範囲で指定してください。")

        if "auto_save_interval" in settings:
            interval = settings["auto_save_interval"]
            if not isinstance(interval, int) or not (10 <= interval <= 300):
                raise ValidationError(
                    "自動保存間隔は10-300秒の範囲で指定してください。"
                )

        if "theme" in settings:
            theme = settings["theme"]
            if theme not in ("light", "dark"):
                raise ValidationError(
                    "テーマは 'light' または 'dark' を指定してください。"
                )

        if "preview_position" in settings:
            pos = settings["preview_position"]
            if pos not in ("right", "bottom"):
                raise ValidationError(
                    "プレビュー位置は 'right' または 'bottom' を指定してください。"
                )
