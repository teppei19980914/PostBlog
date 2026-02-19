"""Markdownエディタコンポーネント。

テキスト編集と変更イベント通知を提供する。
"""

from __future__ import annotations

from collections.abc import Callable

import customtkinter as ctk


class MarkdownEditor(ctk.CTkFrame):
    """Markdownエディタ。

    Args:
        parent: 親ウィジェット。
        on_change: テキスト変更時コールバック。
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_change = on_change
        self._debounce_id: str | None = None

        self._textbox = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(family="Consolas", size=14),
            wrap="word",
        )
        self._textbox.pack(fill="both", expand=True)

        # テキスト変更イベント
        self._textbox.bind("<KeyRelease>", self._on_key_release)

    def _on_key_release(self, event: object) -> None:
        """キーリリースイベントハンドラ（300msデバウンス）。"""
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(300, self._fire_change)

    def _fire_change(self) -> None:
        """変更イベントを発火する。"""
        if self._on_change is not None:
            self._on_change(self.get_text())

    def get_text(self) -> str:
        """テキストを取得する。

        Returns:
            エディタ内のテキスト。
        """
        return self._textbox.get("1.0", "end-1c")

    def set_text(self, text: str) -> None:
        """テキストを設定する。

        Args:
            text: 設定するテキスト。
        """
        self._textbox.delete("1.0", "end")
        self._textbox.insert("1.0", text)

    def set_editable(self, editable: bool) -> None:
        """編集可否を設定する。

        Args:
            editable: 編集可能にする場合True。
        """
        self._textbox.configure(state="normal" if editable else "disabled")
