"""認証情報管理モジュール。

OS標準のキーチェーン（keyring）を使用して、
APIキーやトークンなどの機密情報を安全に管理する。
"""

import logging

import keyring


logger = logging.getLogger(__name__)

# keyringサービス名のプレフィックス
SERVICE_PREFIX = "postblog"


class CredentialManager:
    """認証情報の保存・取得・削除を管理する。

    keyringライブラリを使用してOS標準のキーチェーンに保存する。
    """

    def store(self, service_name: str, credential_type: str, value: str) -> None:
        """認証情報を保存する。

        Args:
            service_name: サービス名（例: "qiita"）。
            credential_type: 認証情報の種類（例: "api_token"）。
            value: 認証情報の値。
        """
        key = self._build_key(service_name, credential_type)
        keyring.set_password(SERVICE_PREFIX, key, value)
        logger.info(
            "認証情報を保存しました: service=%s, type=%s", service_name, credential_type
        )

    def retrieve(self, service_name: str, credential_type: str) -> str | None:
        """認証情報を取得する。

        Args:
            service_name: サービス名（例: "qiita"）。
            credential_type: 認証情報の種類（例: "api_token"）。

        Returns:
            認証情報の値。存在しない場合はNone。
        """
        key = self._build_key(service_name, credential_type)
        value = keyring.get_password(SERVICE_PREFIX, key)
        if value is None:
            logger.debug(
                "認証情報が見つかりません: service=%s, type=%s",
                service_name,
                credential_type,
            )
        return value

    def delete(self, service_name: str, credential_type: str) -> bool:
        """認証情報を削除する。

        Args:
            service_name: サービス名（例: "qiita"）。
            credential_type: 認証情報の種類（例: "api_token"）。

        Returns:
            削除に成功した場合True。
        """
        key = self._build_key(service_name, credential_type)
        try:
            keyring.delete_password(SERVICE_PREFIX, key)
            logger.info(
                "認証情報を削除しました: service=%s, type=%s",
                service_name,
                credential_type,
            )
            return True
        except keyring.errors.PasswordDeleteError:
            logger.warning(
                "認証情報の削除に失敗しました（存在しない可能性があります）: service=%s, type=%s",
                service_name,
                credential_type,
            )
            return False

    @staticmethod
    def mask(value: str, visible_chars: int = 4) -> str:
        """認証情報をマスク表示用に変換する。

        Args:
            value: マスクする値。
            visible_chars: 表示する末尾の文字数。

        Returns:
            マスクされた文字列（例: "sk-...xxxx"）。
        """
        if len(value) <= visible_chars:
            return "*" * len(value)
        prefix = value[:3] if len(value) > 3 else ""
        suffix = value[-visible_chars:]
        return f"{prefix}...{suffix}"

    @staticmethod
    def _build_key(service_name: str, credential_type: str) -> str:
        """keyring用のキーを構築する。

        Args:
            service_name: サービス名。
            credential_type: 認証情報の種類。

        Returns:
            キー文字列。
        """
        return f"{service_name}_{credential_type}"
