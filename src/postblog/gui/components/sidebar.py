"""サイドバーコンポーネント。

アプリケーションのナビゲーションメニューを提供する。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import customtkinter as ctk


logger = logging.getLogger(__name__)


class Sidebar(ctk.CTkFrame):
    """サイドバーナビゲーション。

    Args:
        parent: 親ウィジェット。
        on_navigate: 画面遷移コールバック。
    """

    WIDTH = 200

    def __init__(
        self, parent: ctk.CTkFrame, on_navigate: Callable[[str], None]
    ) -> None:
        super().__init__(parent, width=self.WIDTH, corner_radius=0)
        self._on_navigate = on_navigate
        self._buttons: dict[str, ctk.CTkButton] = {}
        self._active_name: str = ""
        self._build()

    def _build(self) -> None:
        """サイドバーを構築する。"""
        # ロゴ
        logo_label = ctk.CTkLabel(
            self,
            text="PostBlog",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        logo_label.pack(pady=(20, 30), padx=20)

        # メニュー項目
        menu_items = [
            ("home", "Home"),
            ("new_article", "New Article"),
            ("services", "Services"),
            ("settings", "Settings"),
        ]

        for name, label in menu_items:
            btn = ctk.CTkButton(
                self,
                text=label,
                anchor="w",
                height=40,
                corner_radius=8,
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                command=lambda n=name: self._on_click(n),
            )
            btn.pack(fill="x", padx=10, pady=2)
            self._buttons[name] = btn

    def _on_click(self, name: str) -> None:
        """メニュー項目クリックハンドラ。

        Args:
            name: クリックされた画面名。
        """
        self.set_active(name)
        self._on_navigate(name)

    def set_active(self, name: str) -> None:
        """アクティブなメニュー項目を設定する。

        Args:
            name: アクティブにする画面名。
        """
        for btn_name, btn in self._buttons.items():
            if btn_name == name:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")
        self._active_name = name
