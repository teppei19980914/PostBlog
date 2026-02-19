"""サービス管理画面（SCR-008）。

投稿サービスの認証設定と接続テストを管理する。
"""

from __future__ import annotations

import logging

import customtkinter as ctk

from postblog.gui.navigation import BaseView, NavigationManager


logger = logging.getLogger(__name__)

# サービスごとの認証フィールド定義
SERVICE_FIELDS: dict[str, list[tuple[str, str, bool]]] = {
    "Qiita": [("api_token", "API Token", True)],
    "Zenn": [
        ("github_token", "GitHub Token", True),
        ("repo", "Repository", False),
    ],
    "WordPress": [
        ("site_url", "Site URL", False),
        ("username", "Username", False),
        ("app_password", "App Password", True),
    ],
    "Hatena": [
        ("hatena_id", "Hatena ID", False),
        ("blog_id", "Blog ID", False),
        ("api_key", "API Key", True),
    ],
    "Ameba": [
        ("sender_email", "From Email", False),
        ("recipient_email", "To Email", False),
    ],
}


class ServiceView(BaseView):
    """サービス管理画面。"""

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        super().__init__(parent, navigation)
        self._entries: dict[str, dict[str, ctk.CTkEntry]] = {}

    def build(self) -> None:
        """画面を構築する。"""
        self.frame = ctk.CTkFrame(self.parent, fg_color="transparent")

        # ヘッダー
        ctk.CTkLabel(
            self.frame,
            text="Service Management",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(pady=(20, 10))

        # サービス一覧
        scroll_frame = ctk.CTkScrollableFrame(self.frame)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

        for service_name, fields in SERVICE_FIELDS.items():
            self._create_service_card(scroll_frame, service_name, fields)

        # 戻るボタン
        ctk.CTkButton(
            self.frame,
            text="Back",
            fg_color="transparent",
            border_width=1,
            command=self.navigation.go_back,
        ).pack(pady=20)

    def _create_service_card(
        self,
        parent: ctk.CTkFrame,
        service_name: str,
        fields: list[tuple[str, str, bool]],
    ) -> None:
        """サービスカードを作成する。"""
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.pack(fill="x", pady=8, padx=4)

        # ヘッダー
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            header,
            text=service_name,
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side="left")

        # 接続状態表示
        settings_controller = self.navigation.context.get("settings_controller")
        status_text = "Not configured"
        if settings_controller:
            first_field = fields[0][0]
            display = settings_controller.get_service_credential_display(
                service_name.lower(), first_field
            )
            if display:
                status_text = "Configured"

        ctk.CTkLabel(
            header,
            text=status_text,
            font=ctk.CTkFont(size=12),
            text_color="#4CAF50" if status_text == "Configured" else "gray60",
        ).pack(side="right")

        # 入力フィールド
        self._entries[service_name] = {}
        for field_key, field_label, is_password in fields:
            field_frame = ctk.CTkFrame(card, fg_color="transparent")
            field_frame.pack(fill="x", padx=10, pady=2)

            ctk.CTkLabel(
                field_frame, text=f"{field_label}:", width=120, anchor="w"
            ).pack(side="left")

            entry = ctk.CTkEntry(field_frame, show="*" if is_password else "")
            entry.pack(side="left", fill="x", expand=True)
            self._entries[service_name][field_key] = entry

        # ボタン
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=(5, 10))

        ctk.CTkButton(
            btn_frame,
            text="Test",
            width=80,
            command=lambda sn=service_name: self._on_test(sn),
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            btn_frame,
            text="Save",
            width=80,
            command=lambda sn=service_name: self._on_save(sn),
        ).pack(side="left")

    def _on_test(self, service_name: str) -> None:
        """接続テストを実行する。"""
        settings_controller = self.navigation.context.get("settings_controller")
        if settings_controller is None:
            return

        settings_controller.test_connection(
            service_name.lower(),
            on_success=lambda result: logger.info(
                "接続テスト結果: %s=%s", service_name, result
            ),
            on_error=lambda err: logger.error(
                "接続テストエラー: %s=%s", service_name, err
            ),
        )

    def _on_save(self, service_name: str) -> None:
        """認証情報を保存する。"""
        settings_controller = self.navigation.context.get("settings_controller")
        if settings_controller is None:
            return

        entries = self._entries.get(service_name, {})
        for field_key, entry in entries.items():
            value = entry.get().strip()
            if value:
                try:
                    settings_controller.save_service_credential(
                        service_name.lower(), field_key, value
                    )
                except Exception:
                    logger.exception("認証情報の保存に失敗しました: %s", field_key)
