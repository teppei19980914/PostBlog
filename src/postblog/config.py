"""アプリケーション設定管理モジュール。

~/.postblog/config.toml を読み書きする。
APIキーなどの機密情報はここには保存しない（keyringを使用）。
"""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w


logger = logging.getLogger(__name__)

# デフォルト設定ディレクトリ
CONFIG_DIR = Path.home() / ".postblog"
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class AppConfig:
    """アプリケーション設定。

    Args:
        theme: UIテーマ（"dark" または "light"）。
        font_size: フォントサイズ。
        auto_save_interval: 自動保存間隔（秒）。
        model: LLMモデル名。
        preview_position: プレビュー表示位置（"right" または "bottom"）。
    """

    theme: str = "dark"
    font_size: int = 14
    auto_save_interval: int = 30
    model: str = "gpt-4o"
    preview_position: str = "right"

    def to_dict(self) -> dict[str, dict[str, str | int]]:
        """TOML書き出し用の辞書に変換する。

        Returns:
            セクションごとの辞書。
        """
        return {
            "app": {
                "theme": self.theme,
                "font_size": self.font_size,
                "auto_save_interval": self.auto_save_interval,
            },
            "openai": {
                "model": self.model,
            },
            "editor": {
                "preview_position": self.preview_position,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """辞書からAppConfigを生成する。

        Args:
            data: TOML読み込み結果の辞書。

        Returns:
            AppConfigインスタンス。
        """
        app: dict[str, Any] = data.get("app", {})
        openai: dict[str, Any] = data.get("openai", {})
        editor: dict[str, Any] = data.get("editor", {})

        kwargs: dict[str, Any] = {}
        if "theme" in app:
            kwargs["theme"] = str(app["theme"])
        if "font_size" in app:
            kwargs["font_size"] = int(app["font_size"])
        if "auto_save_interval" in app:
            kwargs["auto_save_interval"] = int(app["auto_save_interval"])
        if "model" in openai:
            kwargs["model"] = str(openai["model"])
        if "preview_position" in editor:
            kwargs["preview_position"] = str(editor["preview_position"])

        return cls(**kwargs)


class ConfigManager:
    """設定ファイルの読み書きを管理する。

    Args:
        config_path: 設定ファイルのパス（Noneの場合はデフォルト）。
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or CONFIG_FILE
        self._config: AppConfig = AppConfig()

    @property
    def config(self) -> AppConfig:
        """現在の設定を返す。"""
        return self._config

    @property
    def config_path(self) -> Path:
        """設定ファイルのパスを返す。"""
        return self._config_path

    def load(self) -> AppConfig:
        """設定ファイルを読み込む。

        ファイルが存在しない場合はデフォルト設定を返す。

        Returns:
            読み込んだ設定。
        """
        if not self._config_path.exists():
            logger.info(
                "設定ファイルが見つかりません。デフォルト設定を使用します: %s",
                self._config_path,
            )
            self._config = AppConfig()
            return self._config

        try:
            text = self._config_path.read_text(encoding="utf-8")
            data = tomllib.loads(text)
            self._config = AppConfig.from_dict(data)
            logger.info("設定ファイルを読み込みました: %s", self._config_path)
        except Exception:
            logger.exception(
                "設定ファイルの読み込みに失敗しました: %s", self._config_path
            )
            self._config = AppConfig()

        return self._config

    def save(self, config: AppConfig | None = None) -> None:
        """設定ファイルに保存する。

        Args:
            config: 保存する設定（Noneの場合は現在の設定）。
        """
        if config is not None:
            self._config = config

        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._config.to_dict()
        content = tomli_w.dumps(data)
        self._config_path.write_text(content, encoding="utf-8")
        logger.info("設定ファイルを保存しました: %s", self._config_path)

    def update(self, **kwargs: str | int) -> AppConfig:
        """設定を部分更新する。

        Args:
            **kwargs: 更新するフィールドとその値。

        Returns:
            更新後の設定。
        """
        valid_fields = {f.name for f in self._config.__dataclass_fields__.values()}
        for key, value in kwargs.items():
            if key not in valid_fields:
                logger.warning("不明な設定キー: %s", key)
                continue
            setattr(self._config, key, value)

        return self._config
