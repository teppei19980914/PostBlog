"""ダイアログコンポーネント。

確認・エラー・未保存変更ダイアログを提供する。
"""

from __future__ import annotations

from collections.abc import Callable

import customtkinter as ctk


class ConfirmDialog(ctk.CTkToplevel):
    """確認ダイアログ。

    Args:
        parent: 親ウィジェット。
        title: ダイアログタイトル。
        message: 確認メッセージ。
        on_confirm: 確認時コールバック。
        on_cancel: キャンセル時コールバック。
    """

    def __init__(
        self,
        parent: ctk.CTk,
        title: str,
        message: str,
        on_confirm: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel

        self.title(title)
        self.geometry("400x200")
        self.resizable(False, False)
        self.grab_set()

        msg_label = ctk.CTkLabel(
            self, text=message, wraplength=350, font=ctk.CTkFont(size=14)
        )
        msg_label.pack(pady=(30, 20), padx=20)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=10)

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            fg_color="transparent",
            border_width=1,
            command=self._cancel,
        ).pack(side="left", padx=10)

        ctk.CTkButton(btn_frame, text="OK", command=self._confirm).pack(
            side="left", padx=10
        )

        self.bind("<Escape>", lambda e: self._cancel())

    def _confirm(self) -> None:
        """確認ボタン押下時の処理。"""
        self.destroy()
        if self._on_confirm is not None:
            self._on_confirm()

    def _cancel(self) -> None:
        """キャンセルボタン押下時の処理。"""
        self.destroy()
        if self._on_cancel is not None:
            self._on_cancel()


class ErrorDialog(ctk.CTkToplevel):
    """エラーダイアログ。

    Args:
        parent: 親ウィジェット。
        title: ダイアログタイトル。
        message: エラーメッセージ。
        details: 詳細情報。
    """

    def __init__(
        self,
        parent: ctk.CTk,
        title: str,
        message: str,
        details: str = "",
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.geometry("450x250")
        self.resizable(False, False)
        self.grab_set()

        msg_label = ctk.CTkLabel(
            self, text=message, wraplength=400, font=ctk.CTkFont(size=14)
        )
        msg_label.pack(pady=(20, 10), padx=20)

        if details:
            details_label = ctk.CTkLabel(
                self,
                text=details,
                wraplength=400,
                font=ctk.CTkFont(size=12),
                text_color="gray60",
            )
            details_label.pack(pady=(0, 10), padx=20)

        ctk.CTkButton(self, text="Close", command=self.destroy).pack(pady=10)

        self.bind("<Escape>", lambda e: self.destroy())


class UnsavedChangesDialog(ctk.CTkToplevel):
    """未保存変更確認ダイアログ。

    Args:
        parent: 親ウィジェット。
        on_save: 保存して移動コールバック。
        on_discard: 破棄して移動コールバック。
        on_cancel: キャンセルコールバック。
    """

    def __init__(
        self,
        parent: ctk.CTk,
        on_save: Callable[[], None] | None = None,
        on_discard: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_save = on_save
        self._on_discard = on_discard
        self._on_cancel = on_cancel

        self.title("Unsaved Changes")
        self.geometry("400x180")
        self.resizable(False, False)
        self.grab_set()

        msg_label = ctk.CTkLabel(
            self,
            text="Save changes before leaving?",
            wraplength=350,
            font=ctk.CTkFont(size=14),
        )
        msg_label.pack(pady=(30, 20), padx=20)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=10)

        ctk.CTkButton(
            btn_frame,
            text="Discard",
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90"),
            command=self._discard,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90"),
            command=self._cancel,
        ).pack(side="left", padx=5)

        ctk.CTkButton(btn_frame, text="Save & Go", command=self._save).pack(
            side="left", padx=5
        )

        self.bind("<Escape>", lambda e: self._cancel())

    def _save(self) -> None:
        """保存して移動。"""
        self.destroy()
        if self._on_save is not None:
            self._on_save()

    def _discard(self) -> None:
        """破棄して移動。"""
        self.destroy()
        if self._on_discard is not None:
            self._on_discard()

    def _cancel(self) -> None:
        """キャンセル。"""
        self.destroy()
        if self._on_cancel is not None:
            self._on_cancel()
