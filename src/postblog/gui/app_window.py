"""メインウィンドウ。

アプリケーションのメインウィンドウとレイアウトを管理する。
"""

from __future__ import annotations

import logging

import customtkinter as ctk

from postblog.gui.components.sidebar import Sidebar
from postblog.gui.components.statusbar import StatusBar
from postblog.gui.navigation import NavigationManager
from postblog.gui.views.blog_type_view import BlogTypeView
from postblog.gui.views.editor_view import EditorView
from postblog.gui.views.hearing_view import HearingView
from postblog.gui.views.home_view import HomeView
from postblog.gui.views.publish_view import PublishView
from postblog.gui.views.result_view import ResultView
from postblog.gui.views.service_view import ServiceView
from postblog.gui.views.settings_view import SettingsView
from postblog.gui.views.summary_view import SummaryView


logger = logging.getLogger(__name__)


class AppWindow(ctk.CTk):
    """メインアプリケーションウィンドウ。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("PostBlog")
        self.geometry("1200x800")
        self.minsize(900, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._build_layout()
        self._register_views()
        self._navigation.navigate("home")

    def _build_layout(self) -> None:
        """レイアウトを構築する。"""
        # メインコンテナ
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True)

        # サイドバー
        self._sidebar = Sidebar(main_frame, on_navigate=self._on_sidebar_navigate)
        self._sidebar.pack(side="left", fill="y")

        # コンテンツエリア
        content_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        content_frame.pack(side="left", fill="both", expand=True)

        self._navigation = NavigationManager(content_frame)

        # ステータスバー
        self._statusbar = StatusBar(self)
        self._statusbar.pack(side="bottom", fill="x")

    def _register_views(self) -> None:
        """全画面を登録する。"""
        self._navigation.register("home", HomeView)
        self._navigation.register("blog_type", BlogTypeView)
        self._navigation.register("hearing", HearingView)
        self._navigation.register("summary", SummaryView)
        self._navigation.register("editor", EditorView)
        self._navigation.register("publish", PublishView)
        self._navigation.register("result", ResultView)
        self._navigation.register("services", ServiceView)
        self._navigation.register("settings", SettingsView)

    def _on_sidebar_navigate(self, name: str) -> None:
        """サイドバーナビゲーションハンドラ。

        Args:
            name: 選択されたメニュー名。
        """
        view_map = {
            "home": "home",
            "new_article": "blog_type",
            "services": "services",
            "settings": "settings",
        }
        view_name = view_map.get(name, name)
        self._navigation.navigate(view_name)
        self._sidebar.set_active(name)

    @property
    def navigation(self) -> NavigationManager:
        """NavigationManagerのプロパティ。"""
        return self._navigation

    @property
    def statusbar(self) -> StatusBar:
        """StatusBarのプロパティ。"""
        return self._statusbar
