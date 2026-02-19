"""サマリー確認画面（SCR-004）。

ヒアリング結果のサマリーを確認・編集する。
"""

from __future__ import annotations

import logging

import customtkinter as ctk

from postblog.gui.navigation import BaseView, NavigationManager


logger = logging.getLogger(__name__)


class SummaryView(BaseView):
    """サマリー確認画面。"""

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        super().__init__(parent, navigation)

    def build(self) -> None:
        """画面を構築する。"""
        self.frame = ctk.CTkFrame(self.parent, fg_color="transparent")

        # ヘッダー
        ctk.CTkLabel(
            self.frame,
            text="Hearing Summary",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(pady=(20, 10))

        # サマリー表示エリア
        scroll_frame = ctk.CTkScrollableFrame(self.frame)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

        hearing_controller = self.navigation.context.get("hearing_controller")
        if hearing_controller and hearing_controller.hearing_result:
            result = hearing_controller.hearing_result

            # サマリーテキスト
            if result.summary:
                ctk.CTkLabel(
                    scroll_frame,
                    text="Summary",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    anchor="w",
                ).pack(anchor="w", pady=(10, 5))

                summary_text = ctk.CTkTextbox(scroll_frame, height=100)
                summary_text.insert("1.0", result.summary)
                summary_text.pack(fill="x", pady=(0, 10))

            # 回答一覧
            if result.answers:
                ctk.CTkLabel(
                    scroll_frame,
                    text="Answers",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    anchor="w",
                ).pack(anchor="w", pady=(10, 5))

                for key, value in result.answers.items():
                    item_frame = ctk.CTkFrame(scroll_frame)
                    item_frame.pack(fill="x", pady=4)

                    ctk.CTkLabel(
                        item_frame,
                        text=f"{key}:",
                        font=ctk.CTkFont(weight="bold"),
                        anchor="w",
                    ).pack(padx=10, pady=(5, 0), anchor="w")

                    ctk.CTkLabel(
                        item_frame,
                        text=value,
                        anchor="w",
                        wraplength=600,
                    ).pack(padx=10, pady=(0, 5), anchor="w")

            # SEO情報
            if result.seo_keywords:
                ctk.CTkLabel(
                    scroll_frame,
                    text="SEO Information",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    anchor="w",
                ).pack(anchor="w", pady=(10, 5))

                seo_info = (
                    f"Keywords: {result.seo_keywords}\n"
                    f"Target Audience: {result.seo_target_audience}\n"
                    f"Search Intent: {result.seo_search_intent}"
                )
                ctk.CTkLabel(
                    scroll_frame,
                    text=seo_info,
                    anchor="w",
                    justify="left",
                ).pack(anchor="w", padx=10)

        # ボタンエリア
        btn_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkButton(
            btn_frame,
            text="Back to Hearing",
            fg_color="transparent",
            border_width=1,
            command=self.navigation.go_back,
        ).pack(side="left")

        ctk.CTkButton(
            btn_frame,
            text="Generate Article",
            command=self._on_generate,
        ).pack(side="right")

    def _on_generate(self) -> None:
        """記事生成へ進む。"""
        hearing_controller = self.navigation.context.get("hearing_controller")
        article_controller = self.navigation.context.get("article_controller")

        if not hearing_controller or not article_controller:
            return

        hearing_result = hearing_controller.hearing_result
        if hearing_result is None:
            return

        try:
            article_controller.generate_article(
                hearing_result,
                on_success=lambda result: self._on_generate_success(),
                on_error=lambda err: self._on_generate_error(err),
            )
            self.navigation.navigate("editor")
        except Exception:
            logger.exception("記事生成の開始に失敗しました")

    def _on_generate_success(self) -> None:
        """記事生成成功時。"""
        logger.info("記事生成が完了しました")

    def _on_generate_error(self, error: Exception) -> None:
        """記事生成エラー時。"""
        logger.error("記事生成エラー: %s", error)
