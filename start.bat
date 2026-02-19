@echo off
REM PostBlog 起動スクリプト（Windows用）
REM ダブルクリックでアプリを起動できます。

setlocal

REM プロジェクトルートに移動
cd /d "%~dp0"

REM 依存関係のインストール（初回 or 更新時のみ実行される）
echo PostBlog を起動しています...
uv sync --quiet
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] 依存関係のインストールに失敗しました。
    echo uv がインストールされているか確認してください。
    echo https://docs.astral.sh/uv/
    pause
    exit /b 1
)

REM アプリケーション起動
uv run postblog
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] アプリケーションの起動に失敗しました。
    pause
    exit /b 1
)
