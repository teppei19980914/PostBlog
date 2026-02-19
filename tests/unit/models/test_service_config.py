"""ServiceConfigモデルのテスト。"""

from datetime import datetime

from postblog.models.service_config import ServiceConfig


class TestServiceConfig:
    """ServiceConfigのテスト。"""

    def test_create_config_minimal(self) -> None:
        """最小限のパラメータでServiceConfigが作成されることを確認する。"""
        config = ServiceConfig(service_name="qiita")

        assert config.service_name == "qiita"
        assert config.enabled is False
        assert config.credentials == {}
        assert config.connection_status == "unconfigured"
        assert config.last_checked is None

    def test_create_config_full(self) -> None:
        """全パラメータ指定でServiceConfigが作成されることを確認する。"""
        now = datetime(2024, 6, 1, 12, 0, 0)
        config = ServiceConfig(
            service_name="qiita",
            enabled=True,
            credentials={"api_token": "dummy_token_123"},
            connection_status="connected",
            last_checked=now,
        )

        assert config.service_name == "qiita"
        assert config.enabled is True
        assert config.credentials["api_token"] == "dummy_token_123"
        assert config.connection_status == "connected"
        assert config.last_checked == now

    def test_config_is_mutable(self) -> None:
        """ServiceConfigがミュータブルであることを確認する。"""
        config = ServiceConfig(service_name="qiita")

        config.enabled = True
        config.connection_status = "connected"
        config.last_checked = datetime.now()

        assert config.enabled is True
        assert config.connection_status == "connected"
        assert config.last_checked is not None

    def test_config_credentials_are_independent(self) -> None:
        """デフォルトのcredentials辞書が独立していることを確認する。"""
        config1 = ServiceConfig(service_name="qiita")
        config2 = ServiceConfig(service_name="zenn")

        config1.credentials["token"] = "abc"

        assert len(config2.credentials) == 0

    def test_config_connection_statuses(self) -> None:
        """各接続状態を設定できることを確認する。"""
        config = ServiceConfig(service_name="wordpress")

        statuses = ["unconfigured", "connected", "auth_error", "connection_error"]
        for status in statuses:
            config.connection_status = status
            assert config.connection_status == status

    def test_config_different_services(self) -> None:
        """異なるサービスの設定が作成できることを確認する。"""
        services = ["qiita", "zenn", "wordpress", "hatena", "ameba"]

        configs = [ServiceConfig(service_name=name) for name in services]

        assert len(configs) == 5
        assert configs[0].service_name == "qiita"
        assert configs[4].service_name == "ameba"
