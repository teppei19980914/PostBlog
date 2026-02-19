"""サービス設定のデータモデル。"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ServiceConfig:
    """投稿サービスの設定。

    Args:
        service_name: サービス名（例: "qiita"）。
        enabled: 有効フラグ。
        credentials: 認証情報のキーと値のマップ。
        connection_status: 接続状態
            （"unconfigured" | "connected" | "auth_error" | "connection_error"）。
        last_checked: 最終接続確認日時。
    """

    service_name: str
    enabled: bool = False
    credentials: dict[str, str] = field(default_factory=dict)
    connection_status: str = "unconfigured"
    last_checked: datetime | None = None
