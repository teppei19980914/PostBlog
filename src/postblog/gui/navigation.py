"""画面遷移管理。

BaseViewと NavigationManager による画面遷移と履歴管理を提供する。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:  # pragma: no cover
    import customtkinter as ctk

logger = logging.getLogger(__name__)


class BaseView:
    """全画面の基底クラス。

    Args:
        parent: 親ウィジェット。
        navigation: 画面遷移マネージャ。
    """

    def __init__(self, parent: ctk.CTkFrame, navigation: NavigationManager) -> None:
        self.parent = parent
        self.navigation = navigation
        self.frame: ctk.CTkFrame | None = None

    def build(self) -> None:
        """画面を構築する。サブクラスでオーバーライドする。"""

    def show(self, **kwargs: Any) -> None:
        """画面を表示する。

        Args:
            **kwargs: 画面に渡すパラメータ。
        """
        if self.frame is not None:
            self.frame.destroy()
        self.build()
        if self.frame is not None:
            self.frame.pack(fill="both", expand=True)

    def hide(self) -> None:
        """画面を非表示にする。"""
        if self.frame is not None:
            self.frame.pack_forget()

    def destroy(self) -> None:
        """画面を破棄する。"""
        if self.frame is not None:
            self.frame.destroy()
            self.frame = None


class NavigationManager:
    """画面遷移と履歴を管理する。

    Args:
        content_frame: 画面表示用のコンテナフレーム。
    """

    def __init__(self, content_frame: ctk.CTkFrame) -> None:
        self._content_frame = content_frame
        self._views: dict[str, type[BaseView]] = {}
        self._view_instances: dict[str, BaseView] = {}
        self._current_view: BaseView | None = None
        self._current_view_name: str = ""
        self._history: list[str] = []
        self._context: dict[str, Any] = {}

    @property
    def current_view_name(self) -> str:
        """現在の画面名。"""
        return self._current_view_name

    @property
    def context(self) -> dict[str, Any]:
        """画面間で共有するコンテキスト。"""
        return self._context

    def register(self, name: str, view_class: type[BaseView]) -> None:
        """画面を登録する（遅延読み込み）。

        Args:
            name: 画面名。
            view_class: 画面クラス。
        """
        self._views[name] = view_class

    def navigate(self, name: str, **kwargs: Any) -> None:
        """指定した画面に遷移する。

        Args:
            name: 遷移先画面名。
            **kwargs: 画面に渡すパラメータ。
        """
        if name not in self._views:
            logger.error("未登録の画面: %s", name)
            return

        if self._current_view is not None:
            self._current_view.hide()
            if self._current_view_name:
                self._history.append(self._current_view_name)

        if name not in self._view_instances:
            view_class = self._views[name]
            self._view_instances[name] = view_class(self._content_frame, self)

        view = self._view_instances[name]
        view.show(**kwargs)
        self._current_view = view
        self._current_view_name = name
        logger.info("画面遷移: %s", name)

    def go_back(self) -> None:
        """前の画面に戻る。"""
        if not self._history:
            return
        previous = self._history.pop()
        if self._current_view is not None:
            self._current_view.hide()

        if previous in self._view_instances:
            view = self._view_instances[previous]
            view.show()
            self._current_view = view
            self._current_view_name = previous
            logger.info("画面戻り: %s", previous)

    def can_go_back(self) -> bool:
        """前の画面に戻れるかどうか。"""
        return len(self._history) > 0
