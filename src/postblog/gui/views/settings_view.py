"""設定画面（SCR-009）。

アプリケーション設定とAI設定を管理する。
"""

from __future__ import annotations

import logging

import customtkinter as ctk

from postblog.gui.navigation import BaseView, NavigationManager


logger = logging.getLogger(__name__)


class SettingsView(BaseView):
    """設定画面。"""

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        super().__init__(parent, navigation)

    def build(self) -> None:
        """画面を構築する。"""
        self.frame = ctk.CTkFrame(self.parent, fg_color="transparent")

        # ヘッダー
        ctk.CTkLabel(
            self.frame,
            text="Settings",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(pady=(20, 10))

        scroll_frame = ctk.CTkScrollableFrame(self.frame)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

        settings_controller = self.navigation.context.get("settings_controller")
        settings = {}
        if settings_controller:
            settings = settings_controller.get_app_settings()

        # AI設定セクション
        self._create_section_header(scroll_frame, "AI Settings")

        # APIキー
        api_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        api_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(api_frame, text="OpenAI API Key:", width=150, anchor="w").pack(
            side="left"
        )
        self._api_key_entry = ctk.CTkEntry(api_frame, show="*")
        self._api_key_entry.pack(side="left", fill="x", expand=True)

        if settings_controller:
            display = settings_controller.get_api_key_display()
            if display:
                self._api_key_entry.insert(0, display)

        ctk.CTkButton(
            api_frame, text="Save", width=60, command=self._save_api_key
        ).pack(side="left", padx=(5, 0))

        # モデル選択
        model_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        model_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(model_frame, text="Model:", width=150, anchor="w").pack(
            side="left"
        )
        self._model_var = ctk.StringVar(value=str(settings.get("model", "gpt-4o")))
        ctk.CTkOptionMenu(
            model_frame,
            values=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            variable=self._model_var,
        ).pack(side="left")

        # 表示設定セクション
        self._create_section_header(scroll_frame, "Display Settings")

        # テーマ
        theme_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        theme_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(theme_frame, text="Theme:", width=150, anchor="w").pack(
            side="left"
        )
        self._theme_var = ctk.StringVar(value=str(settings.get("theme", "dark")))
        ctk.CTkOptionMenu(
            theme_frame,
            values=["dark", "light"],
            variable=self._theme_var,
        ).pack(side="left")

        # フォントサイズ
        font_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        font_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(font_frame, text="Font Size:", width=150, anchor="w").pack(
            side="left"
        )
        self._font_var = ctk.StringVar(value=str(settings.get("font_size", 14)))
        ctk.CTkOptionMenu(
            font_frame,
            values=["12", "14", "16", "18", "20"],
            variable=self._font_var,
        ).pack(side="left")

        # エディタ設定セクション
        self._create_section_header(scroll_frame, "Editor Settings")

        # 自動保存間隔
        save_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        save_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(save_frame, text="Auto Save:", width=150, anchor="w").pack(
            side="left"
        )
        self._save_var = ctk.StringVar(
            value=str(settings.get("auto_save_interval", 30))
        )
        ctk.CTkOptionMenu(
            save_frame,
            values=["15", "30", "60", "120"],
            variable=self._save_var,
        ).pack(side="left")

        # プレビュー位置
        preview_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        preview_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(
            preview_frame, text="Preview Position:", width=150, anchor="w"
        ).pack(side="left")
        self._preview_var = ctk.StringVar(
            value=str(settings.get("preview_position", "right"))
        )
        ctk.CTkOptionMenu(
            preview_frame,
            values=["right", "bottom"],
            variable=self._preview_var,
        ).pack(side="left")

        # データ管理セクション
        self._create_section_header(scroll_frame, "Data Management")

        if settings_controller:
            location = settings_controller.get_data_location()
            ctk.CTkLabel(
                scroll_frame,
                text=f"Data Location: {location}",
                font=ctk.CTkFont(size=12),
                text_color="gray60",
            ).pack(anchor="w", padx=10)

        # ボタンエリア
        btn_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=15)

        ctk.CTkButton(
            btn_frame,
            text="Reset to Defaults",
            fg_color="transparent",
            border_width=1,
            command=self._on_reset,
        ).pack(side="left")

        ctk.CTkButton(btn_frame, text="Save", command=self._on_save).pack(side="right")

        ctk.CTkButton(
            btn_frame,
            text="Back",
            fg_color="transparent",
            border_width=1,
            command=self.navigation.go_back,
        ).pack(side="right", padx=5)

    @staticmethod
    def _create_section_header(parent: ctk.CTkFrame, title: str) -> None:
        """セクションヘッダーを作成する。"""
        ctk.CTkLabel(
            parent,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).pack(anchor="w", pady=(15, 5))

    def _save_api_key(self) -> None:
        """APIキーを保存する。"""
        settings_controller = self.navigation.context.get("settings_controller")
        if settings_controller is None:
            return

        key = self._api_key_entry.get().strip()
        if key:
            try:
                settings_controller.save_api_key(key)
                logger.info("APIキーを保存しました")
            except Exception:
                logger.exception("APIキーの保存に失敗しました")

    def _on_save(self) -> None:
        """設定を保存する。"""
        settings_controller = self.navigation.context.get("settings_controller")
        if settings_controller is None:
            return

        try:
            settings_controller.update_app_settings(
                model=self._model_var.get(),
                theme=self._theme_var.get(),
                font_size=int(self._font_var.get()),
                auto_save_interval=int(self._save_var.get()),
                preview_position=self._preview_var.get(),
            )
            # テーマ適用
            ctk.set_appearance_mode(self._theme_var.get())
            logger.info("設定を保存しました")
        except Exception:
            logger.exception("設定の保存に失敗しました")

    def _on_reset(self) -> None:
        """設定をリセットする。"""
        settings_controller = self.navigation.context.get("settings_controller")
        if settings_controller is None:
            return

        try:
            settings_controller.reset_to_defaults()
            # 画面を再構築
            self.show()
            logger.info("設定をリセットしました")
        except Exception:
            logger.exception("設定のリセットに失敗しました")
