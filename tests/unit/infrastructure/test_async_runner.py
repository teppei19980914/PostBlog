"""非同期ランナーのテスト。"""

import time

from postblog.infrastructure.async_runner import AsyncRunner


class TestAsyncRunner:
    """AsyncRunnerのテスト。"""

    def test_start_and_stop(self) -> None:
        """開始と停止ができることを確認する。"""
        runner = AsyncRunner()
        runner.start()

        assert runner.is_running is True

        runner.stop()
        # stopの後は_loopがNoneになる
        assert runner._loop is None

    def test_start_twice(self) -> None:
        """二重開始しないことを確認する。"""
        runner = AsyncRunner()
        runner.start()
        runner.start()  # 二度目は無視される

        assert runner.is_running is True

        runner.stop()

    def test_stop_without_start(self) -> None:
        """開始前のstopがエラーにならないことを確認する。"""
        runner = AsyncRunner()
        runner.stop()  # エラーにならない

    def test_run_coroutine(self) -> None:
        """コルーチンを実行できることを確認する。"""
        runner = AsyncRunner()
        runner.start()
        result_holder: list[str] = []

        async def test_coro() -> str:
            return "hello"

        def on_success(result: str) -> None:
            result_holder.append(result)

        runner.run(test_coro(), on_success=on_success)
        time.sleep(0.5)  # コールバック実行を待つ

        assert result_holder == ["hello"]

        runner.stop()

    def test_run_coroutine_with_error(self) -> None:
        """エラーハンドリングが動作することを確認する。"""
        runner = AsyncRunner()
        runner.start()
        error_holder: list[str] = []

        async def failing_coro() -> str:
            msg = "test error"
            raise ValueError(msg)

        def on_error(e: Exception) -> None:
            error_holder.append(str(e))

        runner.run(failing_coro(), on_error=on_error)
        time.sleep(0.5)

        assert len(error_holder) == 1
        assert "test error" in error_holder[0]

        runner.stop()

    def test_run_auto_starts(self) -> None:
        """runが自動的にstartすることを確認する。"""
        runner = AsyncRunner()
        result_holder: list[int] = []

        async def test_coro() -> int:
            return 42

        def on_success(result: int) -> None:
            result_holder.append(result)

        runner.run(test_coro(), on_success=on_success)
        time.sleep(0.5)

        assert result_holder == [42]

        runner.stop()

    def test_is_running_before_start(self) -> None:
        """開始前はis_runningがFalseであることを確認する。"""
        runner = AsyncRunner()
        assert runner.is_running is False
