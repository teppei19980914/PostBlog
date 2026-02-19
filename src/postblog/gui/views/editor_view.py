"""記事エディタ画面（SCR-005）。

Markdownエディタ+プレビュー+SEOパネルを提供する。
"""

from __future__ import annotations

import logging
from typing import Any

import customtkinter as ctk

from postblog.gui.components.markdown_editor import MarkdownEditor
from postblog.gui.components.markdown_preview import MarkdownPreview
from postblog.gui.components.tag_input import TagInput
from postblog.gui.navigation import BaseView, NavigationManager


logger = logging.getLogger(__name__)


class EditorView(BaseView):
    """記事エディタ画面。"""

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        super().__init__(parent, navigation)
        self._editor: MarkdownEditor | None = None
        self._preview: MarkdownPreview | None = None
        self._tag_input: TagInput | None = None
        self._title_entry: ctk.CTkEntry | None = None
        self._meta_textbox: ctk.CTkTextbox | None = None
        self._seo_score_label: ctk.CTkLabel | None = None
        self._seo_items_frame: ctk.CTkFrame | None = None
        self._preview_visible: bool = True
        self._seo_visible: bool = True

    def build(self) -> None:
        """画面を構築する。"""
        self.frame = ctk.CTkFrame(self.parent, fg_color="transparent")

        # メタデータエリア
        meta_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        meta_frame.pack(fill="x", padx=20, pady=(10, 5))

        # タイトル
        ctk.CTkLabel(meta_frame, text="Title:", anchor="w").pack(anchor="w")
        self._title_entry = ctk.CTkEntry(meta_frame, height=35)
        self._title_entry.pack(fill="x", pady=(2, 5))

        # タグ + キーワード
        tag_kw_frame = ctk.CTkFrame(meta_frame, fg_color="transparent")
        tag_kw_frame.pack(fill="x", pady=(0, 5))

        tag_frame = ctk.CTkFrame(tag_kw_frame, fg_color="transparent")
        tag_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(tag_frame, text="Tags:", anchor="w").pack(anchor="w")
        self._tag_input = TagInput(tag_frame)
        self._tag_input.pack(fill="x")

        # メタディスクリプション
        ctk.CTkLabel(meta_frame, text="Meta Description:", anchor="w").pack(
            anchor="w", pady=(5, 0)
        )
        self._meta_textbox = ctk.CTkTextbox(meta_frame, height=50)
        self._meta_textbox.pack(fill="x", pady=(2, 0))

        # メインコンテンツ（エディタ + プレビュー + SEO）
        content_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=5)

        # エディタ
        editor_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        editor_frame.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(editor_frame, text="Editor", anchor="w").pack(anchor="w")
        self._editor = MarkdownEditor(editor_frame, on_change=self._on_editor_change)
        self._editor.pack(fill="both", expand=True)

        # プレビュー
        self._preview_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        self._preview_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ctk.CTkLabel(self._preview_frame, text="Preview", anchor="w").pack(anchor="w")
        self._preview = MarkdownPreview(self._preview_frame)
        self._preview.pack(fill="both", expand=True)

        # SEOパネル
        self._seo_frame = ctk.CTkFrame(content_frame, width=250)
        self._seo_frame.pack(side="right", fill="y", padx=(10, 0))
        self._seo_frame.pack_propagate(False)

        ctk.CTkLabel(
            self._seo_frame, text="SEO Score", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))

        self._seo_score_label = ctk.CTkLabel(
            self._seo_frame, text="--/100", font=ctk.CTkFont(size=24, weight="bold")
        )
        self._seo_score_label.pack(pady=5)

        self._seo_items_frame = ctk.CTkScrollableFrame(self._seo_frame)
        self._seo_items_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # ボタンエリア
        btn_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(5, 15))

        ctk.CTkButton(
            btn_frame,
            text="Regenerate",
            fg_color="transparent",
            border_width=1,
            command=self._on_regenerate,
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(btn_frame, text="Save Draft", command=self._on_save_draft).pack(
            side="left", padx=5
        )

        ctk.CTkButton(btn_frame, text="Publish", command=self._on_publish).pack(
            side="right"
        )

        # 記事データの読み込み
        self._load_article_data()

    def _load_article_data(self) -> None:
        """記事データを読み込む。"""
        article_controller = self.navigation.context.get("article_controller")
        if article_controller is None:
            return

        # 下書き読み込みの場合
        draft_id = self.navigation.context.pop("load_draft_id", None)
        if draft_id is not None:
            try:
                article_controller.load_draft(draft_id)
            except Exception:
                logger.exception("下書き読み込みに失敗しました")

        article = article_controller.current_article
        if article is None:
            return

        if self._title_entry:
            self._title_entry.insert(0, article.title)
        if self._editor:
            self._editor.set_text(article.body)
        if self._tag_input:
            self._tag_input.set_tags(article.tags)
        if self._meta_textbox:
            self._meta_textbox.insert("1.0", article.meta_description)
        if self._preview:
            self._preview.update_preview(article.body)

        self._run_seo_analysis()

    def _on_editor_change(self, text: str) -> None:
        """エディタ変更時のハンドラ。"""
        if self._preview:
            self._preview.update_preview(text)
        self._run_seo_analysis()

    def _run_seo_analysis(self) -> None:
        """SEO分析を実行する。"""
        article_controller = self.navigation.context.get("article_controller")
        if article_controller is None or article_controller.current_article is None:
            return

        # 現在のエディタ内容で記事を更新
        try:
            title = self._title_entry.get() if self._title_entry else ""
            body = self._editor.get_text() if self._editor else ""
            meta = self._meta_textbox.get("1.0", "end-1c") if self._meta_textbox else ""
            tags = self._tag_input.get_tags() if self._tag_input else []

            article_controller.update_article(
                title=title or None,
                body=body or None,
                tags=tags or None,
                meta_description=meta or None,
            )

            result = article_controller.analyze_seo()
            self._display_seo_result(result)
        except Exception:
            pass  # SEO分析失敗は無視

    def _display_seo_result(self, result: Any) -> None:
        """SEO分析結果を表示する。"""
        if self._seo_score_label:
            score = result.score
            self._seo_score_label.configure(text=f"{score}/100")

            # 色分け
            if score >= 80:
                color = "#4CAF50"
            elif score >= 60:
                color = "#8BC34A"
            elif score >= 40:
                color = "#FFC107"
            else:
                color = "#F44336"
            self._seo_score_label.configure(text_color=color)

        if self._seo_items_frame:
            for widget in self._seo_items_frame.winfo_children():
                widget.destroy()

            for item in result.items:
                status_icon = {"pass": "OK", "warn": "!!", "fail": "NG"}.get(
                    item.status, "?"
                )
                status_color = {
                    "pass": "#4CAF50",
                    "warn": "#FFC107",
                    "fail": "#F44336",
                }.get(item.status, "gray")

                item_frame = ctk.CTkFrame(self._seo_items_frame, fg_color="transparent")
                item_frame.pack(fill="x", pady=1)

                ctk.CTkLabel(
                    item_frame,
                    text=status_icon,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=status_color,
                    width=30,
                ).pack(side="left")

                ctk.CTkLabel(
                    item_frame,
                    text=f"{item.category}: {item.name}",
                    font=ctk.CTkFont(size=11),
                    anchor="w",
                ).pack(side="left", fill="x", expand=True)

    def _on_regenerate(self) -> None:
        """記事再生成。"""
        article_controller = self.navigation.context.get("article_controller")
        if article_controller is None:
            return
        try:
            article_controller.regenerate_article(
                on_success=lambda result: self._on_regenerate_success(),
                on_error=lambda err: logger.error("再生成エラー: %s", err),
            )
        except Exception:
            logger.exception("再生成の開始に失敗しました")

    def _on_regenerate_success(self) -> None:
        """再生成成功時。"""
        if self.frame is not None:
            self.frame.after(0, self._load_article_data)

    def _on_save_draft(self) -> None:
        """下書き保存。"""
        article_controller = self.navigation.context.get("article_controller")
        if article_controller is None:
            return
        try:
            self._sync_article_from_ui()
            article_controller.save_draft()
            logger.info("下書きを保存しました")
        except Exception:
            logger.exception("下書き保存に失敗しました")

    def _on_publish(self) -> None:
        """投稿画面へ遷移。"""
        self._sync_article_from_ui()
        self.navigation.navigate("publish")

    def _sync_article_from_ui(self) -> None:
        """UI内容を記事に同期する。"""
        article_controller = self.navigation.context.get("article_controller")
        if article_controller is None or article_controller.current_article is None:
            return

        title = self._title_entry.get() if self._title_entry else ""
        body = self._editor.get_text() if self._editor else ""
        meta = self._meta_textbox.get("1.0", "end-1c") if self._meta_textbox else ""
        tags = self._tag_input.get_tags() if self._tag_input else []

        import contextlib

        with contextlib.suppress(Exception):
            article_controller.update_article(
                title=title or None,
                body=body or None,
                tags=tags or None,
                meta_description=meta or None,
            )
