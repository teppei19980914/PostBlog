"""チャット吹き出しコンポーネント。

AI/ユーザーの表示を切り替えるチャットメッセージを提供する。
"""

from __future__ import annotations

import customtkinter as ctk


class ChatBubble(ctk.CTkFrame):
    """チャット吹き出し。

    Args:
        parent: 親ウィジェット。
        message: メッセージ内容。
        is_user: ユーザーメッセージの場合True。
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        message: str,
        is_user: bool = False,
    ) -> None:
        super().__init__(parent, corner_radius=12)
        self._is_user = is_user
        self._message = message
        self._build()

    def _build(self) -> None:
        """吹き出しを構築する。"""
        if self._is_user:
            self.configure(fg_color=("#0084FF", "#0066CC"))
            text_color = "white"
            anchor = "e"
            padx = (60, 10)
        else:
            self.configure(fg_color=("gray85", "gray25"))
            text_color = ("gray10", "gray90")
            anchor = "w"
            padx = (10, 60)

        label = ctk.CTkLabel(
            self,
            text=self._message,
            wraplength=500,
            justify="left",
            text_color=text_color,
            font=ctk.CTkFont(size=14),
        )
        label.pack(padx=12, pady=8)

        self.pack(anchor=anchor, padx=padx, pady=4, fill="x")

    def update_message(self, message: str) -> None:
        """メッセージを更新する（ストリーミング用）。

        Args:
            message: 新しいメッセージ内容。
        """
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkLabel):
                widget.configure(text=message)
                break
