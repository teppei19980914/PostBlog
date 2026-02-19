"""ステータスバーコンポーネント。

接続状態と最終保存時刻を表示する。
"""

from __future__ import annotations

import logging

import customtkinter as ctk


logger = logging.getLogger(__name__)


class StatusBar(ctk.CTkFrame):
    """ステータスバー。

    Args:
        parent: 親ウィジェット。
    """

    def __init__(self, parent: ctk.CTkFrame) -> None:
        super().__init__(parent, height=30, corner_radius=0)
        self._status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self._status_label.pack(side="left", padx=10, fill="x", expand=True)

        self._save_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12),
            anchor="e",
        )
        self._save_label.pack(side="right", padx=10)

    def set_status(self, message: str) -> None:
        """ステータスメッセージを設定する。

        Args:
            message: 表示するメッセージ。
        """
        self._status_label.configure(text=message)

    def set_last_saved(self, time_str: str) -> None:
        """最終保存時刻を設定する。

        Args:
            time_str: 表示する時刻文字列。
        """
        self._save_label.configure(text=f"Last Save: {time_str}")
