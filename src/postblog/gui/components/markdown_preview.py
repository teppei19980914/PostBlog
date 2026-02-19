"""Markdownプレビューコンポーネント。

Markdownテキストのプレビュー表示を提供する。
"""

from __future__ import annotations

import customtkinter as ctk


class MarkdownPreview(ctk.CTkFrame):
    """Markdownプレビュー表示。

    Args:
        parent: 親ウィジェット。
    """

    def __init__(self, parent: ctk.CTkFrame) -> None:
        super().__init__(parent)
        self._textbox = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(size=14),
            wrap="word",
            state="disabled",
        )
        self._textbox.pack(fill="both", expand=True)

    def update_preview(self, markdown_text: str) -> None:
        """プレビューを更新する。

        Args:
            markdown_text: Markdownテキスト。
        """
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        # 簡易プレビュー: Markdown記法を一部変換して表示
        preview_text = self._simple_render(markdown_text)
        self._textbox.insert("1.0", preview_text)
        self._textbox.configure(state="disabled")

    @staticmethod
    def _simple_render(markdown_text: str) -> str:
        """Markdownをプレーンテキストとして簡易レンダリングする。

        Args:
            markdown_text: Markdownテキスト。

        Returns:
            レンダリングされたテキスト。
        """
        lines = []
        for line in markdown_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                lines.append(stripped[2:].upper())
                lines.append("=" * len(stripped[2:]))
            elif stripped.startswith("## "):
                lines.append(stripped[3:])
                lines.append("-" * len(stripped[3:]))
            elif stripped.startswith("### "):
                lines.append(stripped[4:])
            else:
                lines.append(line)
        return "\n".join(lines)
