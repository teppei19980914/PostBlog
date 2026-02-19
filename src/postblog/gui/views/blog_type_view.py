"""ブログ種別選択画面（SCR-002）。

5種類のブログ種別から1つを選択する。
"""

from __future__ import annotations

import customtkinter as ctk

from postblog.gui.navigation import BaseView, NavigationManager
from postblog.templates.hearing_templates import get_all_blog_types


class BlogTypeView(BaseView):
    """ブログ種別選択画面。"""

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        super().__init__(parent, navigation)

    def build(self) -> None:
        """画面を構築する。"""
        self.frame = ctk.CTkFrame(self.parent, fg_color="transparent")

        # ヘッダー
        header = ctk.CTkLabel(
            self.frame,
            text="Select Blog Type",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        header.pack(pady=(30, 20))

        # カードグリッド
        grid_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        grid_frame.pack(padx=40, pady=10, fill="both", expand=True)

        blog_types = get_all_blog_types()
        icons = {
            "tech": "Tech",
            "diary": "Diary",
            "review": "Review",
            "news": "News",
            "howto": "HowTo",
        }

        for i, bt in enumerate(blog_types):
            row, col = divmod(i, 2)
            card = ctk.CTkFrame(grid_frame, corner_radius=12, height=120)
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

            grid_frame.grid_columnconfigure(col, weight=1)
            grid_frame.grid_rowconfigure(row, weight=1)

            icon_text = icons.get(bt.id, bt.id.upper())
            title_label = ctk.CTkLabel(
                card,
                text=f"{icon_text}\n{bt.name}",
                font=ctk.CTkFont(size=16, weight="bold"),
            )
            title_label.pack(pady=(15, 5))

            desc_label = ctk.CTkLabel(
                card,
                text=bt.description,
                font=ctk.CTkFont(size=12),
                text_color="gray60",
                wraplength=200,
            )
            desc_label.pack(pady=(0, 15))

            # カード全体をクリック可能にする
            for widget in [card, title_label, desc_label]:
                widget.bind(
                    "<Button-1>",
                    lambda e, type_id=bt.id: self._on_select(type_id),
                )

        # 戻るボタン
        back_btn = ctk.CTkButton(
            self.frame,
            text="Back",
            fg_color="transparent",
            border_width=1,
            command=self.navigation.go_back,
        )
        back_btn.pack(pady=20)

    def _on_select(self, blog_type_id: str) -> None:
        """ブログ種別を選択して次へ進む。"""
        self.navigation.context["blog_type_id"] = blog_type_id
        self.navigation.navigate("hearing")
