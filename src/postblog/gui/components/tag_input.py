"""タグ入力コンポーネント。

チップ形式のタグ入力・削除を提供する。
"""

from __future__ import annotations

from collections.abc import Callable

import customtkinter as ctk


class TagInput(ctk.CTkFrame):
    """タグ入力ウィジェット。

    Args:
        parent: 親ウィジェット。
        on_change: タグ変更時コールバック。
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        on_change: Callable[[list[str]], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._tags: list[str] = []
        self._on_change = on_change
        self._chips_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._chips_frame.pack(fill="x", side="top")

        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(fill="x", side="top", pady=(4, 0))

        self._entry = ctk.CTkEntry(input_frame, placeholder_text="Add tag...")
        self._entry.pack(side="left", fill="x", expand=True)
        self._entry.bind("<Return>", self._on_enter)

        self._add_btn = ctk.CTkButton(
            input_frame, text="+", width=30, command=self._add_from_entry
        )
        self._add_btn.pack(side="left", padx=(4, 0))

    def _on_enter(self, event: object) -> None:
        """Enterキーハンドラ。"""
        self._add_from_entry()

    def _add_from_entry(self) -> None:
        """入力フィールドからタグを追加する。"""
        text = self._entry.get().strip()
        if text:
            self.add_tag(text)
            self._entry.delete(0, "end")

    def add_tag(self, tag: str) -> None:
        """タグを追加する。

        Args:
            tag: 追加するタグ文字列。
        """
        if tag and tag not in self._tags:
            self._tags.append(tag)
            self._rebuild_chips()
            self._fire_change()

    def remove_tag(self, tag: str) -> None:
        """タグを削除する。

        Args:
            tag: 削除するタグ文字列。
        """
        if tag in self._tags:
            self._tags.remove(tag)
            self._rebuild_chips()
            self._fire_change()

    def get_tags(self) -> list[str]:
        """全タグを取得する。

        Returns:
            タグ文字列のリスト。
        """
        return list(self._tags)

    def set_tags(self, tags: list[str]) -> None:
        """タグを設定する。

        Args:
            tags: 設定するタグリスト。
        """
        self._tags = list(tags)
        self._rebuild_chips()

    def _rebuild_chips(self) -> None:
        """チップ表示を再構築する。"""
        for widget in self._chips_frame.winfo_children():
            widget.destroy()

        for tag in self._tags:
            chip = ctk.CTkFrame(self._chips_frame, corner_radius=12)
            chip.pack(side="left", padx=2, pady=2)

            label = ctk.CTkLabel(chip, text=tag, font=ctk.CTkFont(size=12))
            label.pack(side="left", padx=(8, 2), pady=2)

            close_btn = ctk.CTkButton(
                chip,
                text="x",
                width=20,
                height=20,
                font=ctk.CTkFont(size=10),
                corner_radius=10,
                fg_color="transparent",
                hover_color=("gray70", "gray30"),
                command=lambda t=tag: self.remove_tag(t),
            )
            close_btn.pack(side="left", padx=(0, 4), pady=2)

    def _fire_change(self) -> None:
        """変更イベントを発火する。"""
        if self._on_change is not None:
            self._on_change(list(self._tags))
