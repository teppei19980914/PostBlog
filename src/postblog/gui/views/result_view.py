"""投稿結果画面（SCR-007）。

投稿結果の成功・失敗を表示する。
"""

from __future__ import annotations

import logging
from typing import Any

import customtkinter as ctk

from postblog.gui.navigation import BaseView, NavigationManager


logger = logging.getLogger(__name__)


class ResultView(BaseView):
    """投稿結果画面。"""

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        super().__init__(parent, navigation)

    def build(self) -> None:
        """画面を構築する。"""
        self.frame = ctk.CTkFrame(self.parent, fg_color="transparent")

        # ヘッダー
        ctk.CTkLabel(
            self.frame,
            text="Publication Results",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(pady=(20, 10))

        # 結果表示エリア
        results_frame = ctk.CTkScrollableFrame(self.frame)
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)

        results = self.navigation.context.get("publish_results", [])
        publish_controller = self.navigation.context.get("publish_controller")

        if results and publish_controller:
            summary = publish_controller.summarize_results(results)

            # 全体サマリー
            total = summary.get("total", 0)
            success_count = summary.get("success_count", 0)
            status_text = (
                "All Published Successfully!"
                if success_count == total
                else f"{success_count}/{total} Published"
            )
            status_color = "#4CAF50" if success_count == total else "#FFC107"

            ctk.CTkLabel(
                results_frame,
                text=status_text,
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=status_color,
            ).pack(pady=10)

            # 成功結果
            for item in summary.get("successes", []):
                self._create_success_card(results_frame, item)

            # 失敗結果
            for item in summary.get("failures", []):
                self._create_failure_card(results_frame, item)
        else:
            ctk.CTkLabel(
                results_frame,
                text="Publishing in progress...",
                font=ctk.CTkFont(size=16),
                text_color="gray60",
            ).pack(pady=30)

        # ホームに戻るボタン
        ctk.CTkButton(
            self.frame,
            text="Return to Home",
            command=lambda: self.navigation.navigate("home"),
        ).pack(pady=20)

    def _create_success_card(self, parent: ctk.CTkFrame, item: dict[str, Any]) -> None:
        """成功カードを作成する。"""
        card = ctk.CTkFrame(parent, corner_radius=8)
        card.pack(fill="x", pady=4)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(
            header,
            text=f"OK {item.get('service_name', '')}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4CAF50",
        ).pack(side="left")

        url = item.get("article_url", "")
        if url:
            ctk.CTkLabel(
                card,
                text=url,
                font=ctk.CTkFont(size=12),
                text_color=("blue", "#4DA6FF"),
                anchor="w",
            ).pack(padx=10, pady=(0, 8), anchor="w")

    def _create_failure_card(self, parent: ctk.CTkFrame, item: dict[str, Any]) -> None:
        """失敗カードを作成する。"""
        card = ctk.CTkFrame(parent, corner_radius=8)
        card.pack(fill="x", pady=4)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(
            header,
            text=f"NG {item.get('service_name', '')}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#F44336",
        ).pack(side="left")

        error_msg = item.get("error_message", "Unknown error")
        ctk.CTkLabel(
            card,
            text=f"Error: {error_msg}",
            font=ctk.CTkFont(size=12),
            text_color="gray60",
            anchor="w",
        ).pack(padx=10, pady=(0, 8), anchor="w")
