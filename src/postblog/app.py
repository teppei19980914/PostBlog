"""PostBlogアプリケーションエントリーポイント。"""

import logging

from postblog.config import ConfigManager
from postblog.controllers.article_controller import ArticleController
from postblog.controllers.hearing_controller import HearingController
from postblog.controllers.home_controller import HomeController
from postblog.controllers.publish_controller import PublishController
from postblog.controllers.settings_controller import SettingsController
from postblog.gui.app_window import AppWindow
from postblog.infrastructure.async_runner import AsyncRunner
from postblog.infrastructure.credential.credential_manager import CredentialManager
from postblog.infrastructure.llm.openai_client import OpenAIClient
from postblog.infrastructure.storage.database import Database
from postblog.infrastructure.storage.draft_repository import DraftRepository
from postblog.infrastructure.storage.history_repository import HistoryRepository
from postblog.logging_config import setup_logging
from postblog.services.article_service import ArticleService
from postblog.services.draft_service import DraftService
from postblog.services.hearing_service import HearingService
from postblog.services.history_service import HistoryService
from postblog.services.publish_service import PublishService


logger = logging.getLogger(__name__)


def main() -> None:  # pragma: no cover
    """アプリケーションを起動する。"""
    setup_logging()
    logger.info("PostBlog を起動します")

    # Infrastructure
    config_manager = ConfigManager()
    config_manager.load()
    credential_manager = CredentialManager()
    database = Database()
    async_runner = AsyncRunner()
    async_runner.start()

    # Repositories
    draft_repo = DraftRepository(database)
    history_repo = HistoryRepository(database)

    # LLM Client
    api_key = credential_manager.retrieve("openai", "api_key") or ""
    llm_client = OpenAIClient(api_key=api_key, model=config_manager.config.model)

    # Services
    article_service = ArticleService(llm_client)
    hearing_service = HearingService(llm_client)
    draft_service = DraftService(draft_repo)
    publish_service = PublishService()
    history_service = HistoryService(history_repo)

    # Controllers
    home_controller = HomeController(draft_service, history_service)
    hearing_controller = HearingController(hearing_service, async_runner)
    article_controller = ArticleController(article_service, draft_service, async_runner)
    publish_controller = PublishController(
        publish_service, history_service, async_runner
    )
    settings_controller = SettingsController(
        config_manager, credential_manager, publish_service, async_runner
    )

    # GUI
    app = AppWindow()

    # コンテキストにコントローラを登録
    app.navigation.context["home_controller"] = home_controller
    app.navigation.context["hearing_controller"] = hearing_controller
    app.navigation.context["article_controller"] = article_controller
    app.navigation.context["publish_controller"] = publish_controller
    app.navigation.context["settings_controller"] = settings_controller

    try:
        app.mainloop()
    finally:
        async_runner.stop()
        logger.info("PostBlog を終了しました")


if __name__ == "__main__":  # pragma: no cover
    main()
