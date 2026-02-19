"""ヒアリング画面（SCR-003）。

チャット形式でヒアリングを行う。
"""

from __future__ import annotations

import logging
from typing import Any

import customtkinter as ctk

from postblog.gui.components.chat_bubble import ChatBubble
from postblog.gui.navigation import BaseView, NavigationManager


logger = logging.getLogger(__name__)


class HearingView(BaseView):
    """ヒアリング画面。"""

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        super().__init__(parent, navigation)
        self._chat_area: ctk.CTkScrollableFrame | None = None
        self._input_textbox: ctk.CTkTextbox | None = None
        self._send_btn: ctk.CTkButton | None = None
        self._progress_bar: ctk.CTkProgressBar | None = None
        self._progress_label: ctk.CTkLabel | None = None

    def build(self) -> None:
        """画面を構築する。"""
        self.frame = ctk.CTkFrame(self.parent, fg_color="transparent")

        # ヘッダー
        header_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(10, 0))

        ctk.CTkLabel(
            header_frame,
            text="Hearing",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left")

        ctk.CTkButton(
            header_frame,
            text="End Hearing",
            width=120,
            command=self._on_finish,
        ).pack(side="right")

        # チャットエリア
        self._chat_area = ctk.CTkScrollableFrame(self.frame)
        self._chat_area.pack(fill="both", expand=True, padx=20, pady=10)

        # プログレスバー
        progress_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        progress_frame.pack(fill="x", padx=20, pady=(0, 5))

        self._progress_bar = ctk.CTkProgressBar(progress_frame)
        self._progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._progress_bar.set(0)

        self._progress_label = ctk.CTkLabel(
            progress_frame, text="0/0", font=ctk.CTkFont(size=12)
        )
        self._progress_label.pack(side="right")

        # 入力エリア
        input_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=20, pady=(0, 15))

        self._input_textbox = ctk.CTkTextbox(input_frame, height=60)
        self._input_textbox.pack(side="left", fill="x", expand=True)
        self._input_textbox.bind("<Control-Return>", lambda e: self._on_send())

        self._send_btn = ctk.CTkButton(
            input_frame, text="Send", width=80, command=self._on_send
        )
        self._send_btn.pack(side="right", padx=(10, 0))

        # ヒアリング開始
        self._start_hearing()

    def _start_hearing(self) -> None:
        """ヒアリングを開始する。"""
        controller = self.navigation.context.get("hearing_controller")
        blog_type_id = self.navigation.context.get("blog_type_id", "tech")

        if controller is None:
            return

        try:
            result = controller.start_hearing(blog_type_id)
            # 最初のAIメッセージを表示
            items = result.get("hearing_items", [])
            if items:
                first_q = items[0].get("question", "")
                self._add_ai_message(f"Let's start! {first_q}")
            self._update_progress()
        except Exception:
            logger.exception("ヒアリング開始に失敗しました")

    def _on_send(self) -> None:
        """メッセージ送信。"""
        if self._input_textbox is None or self._send_btn is None:
            return

        message = self._input_textbox.get("1.0", "end-1c").strip()
        if not message:
            return

        self._input_textbox.delete("1.0", "end")
        self._add_user_message(message)
        self._set_input_enabled(False)

        controller = self.navigation.context.get("hearing_controller")
        if controller is None:
            return

        try:
            controller.send_message(
                message,
                on_success=lambda resp: self._on_response(resp),
                on_error=lambda err: self._on_error(err),
            )
        except Exception as e:
            self._add_ai_message(f"Error: {e}")
            self._set_input_enabled(True)

    def _on_response(self, response: str) -> None:
        """AIレスポンス受信時。"""
        if self.frame is not None:
            self.frame.after(0, lambda: self._handle_response(response))

    def _handle_response(self, response: str) -> None:
        """AIレスポンスをUIに反映する。"""
        self._add_ai_message(response)
        self._set_input_enabled(True)
        self._update_progress()

    def _on_error(self, error: Exception) -> None:
        """エラー発生時。"""
        if self.frame is not None:
            self.frame.after(0, lambda: self._handle_error(str(error)))

    def _handle_error(self, error_msg: str) -> None:
        """エラーをUIに反映する。"""
        self._add_ai_message(f"Error occurred: {error_msg}")
        self._set_input_enabled(True)

    def _on_finish(self) -> None:
        """ヒアリング終了。"""
        controller = self.navigation.context.get("hearing_controller")
        if controller is None:
            return

        try:
            controller.finish_hearing(
                on_success=lambda result: self._on_finish_success(result),
                on_error=lambda err: self._on_error(err),
            )
        except Exception as e:
            logger.exception("ヒアリング終了に失敗しました")
            self._add_ai_message(f"Error: {e}")

    def _on_finish_success(self, result: Any) -> None:
        """ヒアリング終了成功時。"""
        if self.frame is not None:
            self.frame.after(0, lambda: self.navigation.navigate("summary"))

    def _add_ai_message(self, message: str) -> None:
        """AIメッセージを追加する。"""
        if self._chat_area is not None:
            ChatBubble(self._chat_area, message, is_user=False)

    def _add_user_message(self, message: str) -> None:
        """ユーザーメッセージを追加する。"""
        if self._chat_area is not None:
            ChatBubble(self._chat_area, message, is_user=True)

    def _set_input_enabled(self, enabled: bool) -> None:
        """入力エリアの有効/無効を切り替える。"""
        if self._input_textbox is not None:
            self._input_textbox.configure(state="normal" if enabled else "disabled")
        if self._send_btn is not None:
            self._send_btn.configure(state="normal" if enabled else "disabled")

    def _update_progress(self) -> None:
        """プログレスバーを更新する。"""
        controller = self.navigation.context.get("hearing_controller")
        if controller is None:
            return

        progress = controller.get_progress()
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        percentage = progress.get("percentage", 0)

        if self._progress_bar is not None:
            self._progress_bar.set(percentage / 100)
        if self._progress_label is not None:
            self._progress_label.configure(text=f"{completed}/{total}")
