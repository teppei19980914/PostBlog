"""ホーム画面（SCR-001）。

下書き一覧と投稿履歴一覧を表示する。
"""

from __future__ import annotations

from typing import Any

import customtkinter as ctk

from postblog.gui.navigation import BaseView, NavigationManager


class HomeView(BaseView):
    """ホーム画面。"""

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        super().__init__(parent, navigation)

    def build(self) -> None:
        """画面を構築する。"""
        self.frame = ctk.CTkFrame(self.parent, fg_color="transparent")

        # ヘッダー
        header = ctk.CTkLabel(
            self.frame,
            text="PostBlog",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        header.pack(pady=(20, 10))

        # 新規記事ボタン
        new_btn = ctk.CTkButton(
            self.frame,
            text="Create New Article",
            height=50,
            font=ctk.CTkFont(size=16),
            command=lambda: self.navigation.navigate("blog_type"),
        )
        new_btn.pack(pady=20, padx=40, fill="x")

        # 下書き一覧セクション
        drafts_label = ctk.CTkLabel(
            self.frame,
            text="Recent Drafts",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        drafts_label.pack(padx=20, pady=(20, 5), anchor="w")

        self._drafts_frame = ctk.CTkScrollableFrame(self.frame, height=200)
        self._drafts_frame.pack(fill="x", padx=20, pady=(0, 10))

        # 投稿履歴セクション
        history_label = ctk.CTkLabel(
            self.frame,
            text="Recent Posts",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        history_label.pack(padx=20, pady=(10, 5), anchor="w")

        self._history_frame = ctk.CTkScrollableFrame(self.frame, height=200)
        self._history_frame.pack(fill="x", padx=20, pady=(0, 20))

        self._load_data()

    def _load_data(self) -> None:
        """データを読み込む。"""
        controller = self.navigation.context.get("home_controller")
        if controller is None:
            return

        # 下書き一覧
        drafts = controller.get_recent_drafts()
        if not drafts:
            ctk.CTkLabel(
                self._drafts_frame, text="No drafts yet", text_color="gray60"
            ).pack(pady=10)
        for draft in drafts:
            self._create_draft_card(draft)

        # 投稿履歴
        history = controller.get_recent_history()
        if not history:
            ctk.CTkLabel(
                self._history_frame, text="No posts yet", text_color="gray60"
            ).pack(pady=10)
        for record in history:
            self._create_history_card(record)

    def _create_draft_card(self, draft: dict[str, str]) -> None:
        """下書きカードを作成する。"""
        card = ctk.CTkFrame(self._drafts_frame, corner_radius=8)
        card.pack(fill="x", pady=4, padx=4)

        title = ctk.CTkLabel(
            card,
            text=draft.get("title", "Untitled"),
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        title.pack(padx=10, pady=(8, 2), anchor="w")

        info = ctk.CTkLabel(
            card,
            text=draft.get("updated_at", ""),
            font=ctk.CTkFont(size=11),
            text_color="gray60",
            anchor="w",
        )
        info.pack(padx=10, pady=(0, 8), anchor="w")

        card.bind(
            "<Button-1>",
            lambda e, d=draft: self._on_draft_click(d),
        )

    def _create_history_card(self, record: dict[str, Any]) -> None:
        """投稿履歴カードを作成する。"""
        card = ctk.CTkFrame(self._history_frame, corner_radius=8)
        card.pack(fill="x", pady=4, padx=4)

        title = ctk.CTkLabel(
            card,
            text=record.get("title", "Untitled"),
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        title.pack(padx=10, pady=(8, 2), anchor="w")

        service_name = record.get("service_name", "")
        status = record.get("status", "")
        info = ctk.CTkLabel(
            card,
            text=f"{service_name} - {status}",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
            anchor="w",
        )
        info.pack(padx=10, pady=(0, 8), anchor="w")

    def _on_draft_click(self, draft: dict[str, str]) -> None:
        """下書きカードクリック時の処理。"""
        draft_id = draft.get("id", "")
        if draft_id:
            self.navigation.context["load_draft_id"] = int(draft_id)
            self.navigation.navigate("editor")
