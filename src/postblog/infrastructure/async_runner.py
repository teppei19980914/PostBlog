"""非同期ランナーモジュール。

GUIスレッドからasync関数を安全に実行するためのブリッジ。
バックグラウンドスレッドでasyncioイベントループを管理する。
"""

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Callable, Coroutine
from typing import Any


logger = logging.getLogger(__name__)


class AsyncRunner:
    """バックグラウンドasyncioランナー。

    GUIスレッドから非同期処理を実行し、
    コールバックでUI更新を通知する。
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """バックグラウンドイベントループを開始する。"""
        if self._thread is not None and self._thread.is_alive():
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("AsyncRunnerを開始しました")

    def _run_loop(self) -> None:
        """バックグラウンドスレッドでイベントループを実行する。"""
        if self._loop is None:  # pragma: no cover
            return
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def stop(self) -> None:
        """イベントループを停止する。"""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("AsyncRunnerを停止しました")

    def run(
        self,
        coro: Coroutine[Any, Any, Any],
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """コルーチンをバックグラウンドで実行する。

        Args:
            coro: 実行するコルーチン。
            on_success: 成功時のコールバック。
            on_error: エラー時のコールバック。
        """
        if self._loop is None:
            self.start()

        assert self._loop is not None

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _done_callback(fut: concurrent.futures.Future[Any]) -> None:
            try:
                result = fut.result()
                if on_success:
                    on_success(result)
            except Exception as e:
                logger.error("非同期タスクでエラーが発生しました: %s", e)
                if on_error:
                    on_error(e)

        future.add_done_callback(_done_callback)

    @property
    def is_running(self) -> bool:
        """イベントループが実行中かどうかを返す。"""
        return self._loop is not None and self._loop.is_running()
