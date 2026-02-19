"""投稿確認画面（SCR-006）。

投稿先サービスの選択と投稿実行を管理する。
"""

from __future__ import annotations

import logging

import customtkinter as ctk

from postblog.gui.navigation import BaseView, NavigationManager


logger = logging.getLogger(__name__)


class PublishView(BaseView):
    """投稿確認画面。"""

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        super().__init__(parent, navigation)
        self._service_vars: dict[str, ctk.BooleanVar] = {}
        self._status_vars: dict[str, ctk.StringVar] = {}

    def build(self) -> None:
        """画面を構築する。"""
        self.frame = ctk.CTkFrame(self.parent, fg_color="transparent")

        # ヘッダー
        ctk.CTkLabel(
            self.frame,
            text="Publish Article",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(pady=(20, 10))

        # 記事プレビュー
        article_controller = self.navigation.context.get("article_controller")
        if article_controller and article_controller.current_article:
            article = article_controller.current_article
            info_frame = ctk.CTkFrame(self.frame)
            info_frame.pack(fill="x", padx=20, pady=10)

            ctk.CTkLabel(
                info_frame,
                text=f"Title: {article.title}",
                font=ctk.CTkFont(size=14),
                anchor="w",
            ).pack(padx=10, pady=(10, 2), anchor="w")

            tags_text = ", ".join(article.tags) if article.tags else "No tags"
            ctk.CTkLabel(
                info_frame,
                text=f"Tags: {tags_text}",
                font=ctk.CTkFont(size=12),
                text_color="gray60",
                anchor="w",
            ).pack(padx=10, pady=(0, 10), anchor="w")

        # サービス選択
        ctk.CTkLabel(
            self.frame,
            text="Select Destinations",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).pack(padx=20, pady=(10, 5), anchor="w")

        services_frame = ctk.CTkFrame(self.frame)
        services_frame.pack(fill="x", padx=20, pady=(0, 10))

        publish_controller = self.navigation.context.get("publish_controller")
        if publish_controller:
            services = publish_controller.get_available_services()
            for service in services:
                self._create_service_row(services_frame, service)

        if not self._service_vars:
            ctk.CTkLabel(
                services_frame,
                text="No services configured. Go to Settings to add services.",
                text_color="gray60",
            ).pack(pady=10)

        # ボタンエリア
        btn_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkButton(
            btn_frame,
            text="Back to Editor",
            fg_color="transparent",
            border_width=1,
            command=self.navigation.go_back,
        ).pack(side="left")

        ctk.CTkButton(btn_frame, text="Publish", command=self._on_publish).pack(
            side="right"
        )

    def _create_service_row(
        self, parent: ctk.CTkFrame, service: dict[str, str]
    ) -> None:
        """サービス選択行を作成する。"""
        name = service.get("name", "")
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=4)

        var = ctk.BooleanVar(value=False)
        self._service_vars[name] = var

        ctk.CTkCheckBox(row, text=name, variable=var).pack(side="left")

        status_var = ctk.StringVar(value="publish")
        self._status_vars[name] = status_var

        ctk.CTkOptionMenu(
            row, values=["publish", "draft"], variable=status_var, width=100
        ).pack(side="right")

    def _on_publish(self) -> None:
        """投稿を実行する。"""
        selected = [name for name, var in self._service_vars.items() if var.get()]

        if not selected:
            logger.warning("投稿先が選択されていません")
            return

        publish_controller = self.navigation.context.get("publish_controller")
        article_controller = self.navigation.context.get("article_controller")

        if not publish_controller or not article_controller:
            return

        article = article_controller.current_article
        if article is None:
            return

        try:
            publish_controller.publish(
                article,
                selected,
                on_success=lambda results: self._on_publish_success(results),
                on_error=lambda err: self._on_publish_error(err),
            )
            self.navigation.navigate("result")
        except Exception:
            logger.exception("投稿の開始に失敗しました")

    def _on_publish_success(self, results: list[object]) -> None:
        """投稿成功時。"""
        self.navigation.context["publish_results"] = results

    def _on_publish_error(self, error: Exception) -> None:
        """投稿エラー時。"""
        logger.error("投稿エラー: %s", error)
