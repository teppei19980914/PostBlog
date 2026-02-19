"""認証情報管理モジュールのテスト。"""

from unittest.mock import patch

import keyring.errors

from postblog.infrastructure.credential.credential_manager import (
    SERVICE_PREFIX,
    CredentialManager,
)


class TestCredentialManager:
    """CredentialManagerのテスト。"""

    def test_store(self) -> None:
        """認証情報が保存されることを確認する。"""
        manager = CredentialManager()

        with patch.object(keyring, "set_password") as mock_set:
            manager.store("qiita", "api_token", "test_token_123")

            mock_set.assert_called_once_with(
                SERVICE_PREFIX, "qiita_api_token", "test_token_123"
            )

    def test_retrieve_existing(self) -> None:
        """存在する認証情報が取得できることを確認する。"""
        manager = CredentialManager()

        with patch.object(keyring, "get_password", return_value="test_token_123"):
            result = manager.retrieve("qiita", "api_token")

            assert result == "test_token_123"

    def test_retrieve_nonexistent(self) -> None:
        """存在しない認証情報でNoneが返されることを確認する。"""
        manager = CredentialManager()

        with patch.object(keyring, "get_password", return_value=None):
            result = manager.retrieve("qiita", "api_token")

            assert result is None

    def test_delete_existing(self) -> None:
        """存在する認証情報が削除できることを確認する。"""
        manager = CredentialManager()

        with patch.object(keyring, "delete_password") as mock_delete:
            result = manager.delete("qiita", "api_token")

            assert result is True
            mock_delete.assert_called_once_with(SERVICE_PREFIX, "qiita_api_token")

    def test_delete_nonexistent(self) -> None:
        """存在しない認証情報の削除でFalseが返されることを確認する。"""
        manager = CredentialManager()

        with patch.object(
            keyring,
            "delete_password",
            side_effect=keyring.errors.PasswordDeleteError(),
        ):
            result = manager.delete("qiita", "api_token")

            assert result is False

    def test_mask_long_value(self) -> None:
        """長い値が正しくマスクされることを確認する。"""
        assert CredentialManager.mask("sk-abc123456789xxxx") == "sk-...xxxx"

    def test_mask_short_value(self) -> None:
        """短い値が全てマスクされることを確認する。"""
        assert CredentialManager.mask("abc") == "***"

    def test_mask_exact_visible_chars(self) -> None:
        """visible_chars と同じ長さの値が全てマスクされることを確認する。"""
        assert CredentialManager.mask("abcd") == "****"

    def test_mask_custom_visible_chars(self) -> None:
        """カスタムvisible_chars数でマスクされることを確認する。"""
        result = CredentialManager.mask("sk-abc123456789xxxx", visible_chars=6)
        assert result == "sk-...89xxxx"

    def test_build_key(self) -> None:
        """キーの構築が正しいことを確認する。"""
        key = CredentialManager._build_key("qiita", "api_token")
        assert key == "qiita_api_token"

    def test_build_key_different_services(self) -> None:
        """異なるサービスのキーが正しく構築されることを確認する。"""
        assert (
            CredentialManager._build_key("wordpress", "password")
            == "wordpress_password"
        )
        assert CredentialManager._build_key("hatena", "api_key") == "hatena_api_key"
        assert CredentialManager._build_key("openai", "api_key") == "openai_api_key"

    def test_store_and_retrieve_flow(self) -> None:
        """保存→取得の一連のフローを確認する。"""
        manager = CredentialManager()
        stored_values: dict[str, str] = {}

        def mock_set(service: str, key: str, value: str) -> None:
            stored_values[f"{service}:{key}"] = value

        def mock_get(service: str, key: str) -> str | None:
            return stored_values.get(f"{service}:{key}")

        with (
            patch.object(keyring, "set_password", side_effect=mock_set),
            patch.object(keyring, "get_password", side_effect=mock_get),
        ):
            manager.store("qiita", "api_token", "my_token")
            result = manager.retrieve("qiita", "api_token")

            assert result == "my_token"
