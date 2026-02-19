# システムアーキテクチャ設計書

| 項目 | 内容 |
|------|------|
| プロジェクト名 | PostBlog - 高速ブログ投稿支援アプリ |
| バージョン | 1.0.0 |
| 作成日 | 2026-02-19 |
| 最終更新日 | 2026-02-19 |

---

## 1. アーキテクチャ概要

### 1.1 設計原則

| 原則 | 説明 |
|------|------|
| GUI/ロジック分離 | GUI層とビジネスロジック層を完全に分離し、テスト容易性を確保 |
| プラグイン方式 | ブログサービス連携を抽象化し、新サービスの追加を容易にする |
| 非同期処理 | LLM API呼び出し・投稿処理は非同期で実行し、UIをブロックしない |
| 遅延読み込み | 起動時間1秒以内を実現するため、必要時にのみモジュールを読み込む |

### 1.2 全体構成図

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PostBlog Application                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    Presentation Layer (GUI)                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │ │
│  │  │  Home    │ │ Hearing  │ │ Editor   │ │   Settings     │  │ │
│  │  │  View    │ │  View    │ │  View    │ │    View        │  │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬────────┘  │ │
│  └───────┼─────────────┼────────────┼───────────────┼───────────┘ │
│          │             │            │               │             │
│  ┌───────┼─────────────┼────────────┼───────────────┼───────────┐ │
│  │       ▼             ▼            ▼               ▼           │ │
│  │                   Controller Layer                            │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │ │
│  │  │ Home     │ │ Hearing  │ │ Article  │ │   Settings     │  │ │
│  │  │Controller│ │Controller│ │Controller│ │  Controller    │  │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬────────┘  │ │
│  └───────┼─────────────┼────────────┼───────────────┼───────────┘ │
│          │             │            │               │             │
│  ┌───────┼─────────────┼────────────┼───────────────┼───────────┐ │
│  │       ▼             ▼            ▼               ▼           │ │
│  │                    Service Layer                              │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │ │
│  │  │ Draft    │ │ Hearing  │ │ Article  │ │   Auth         │  │ │
│  │  │ Service  │ │ Service  │ │ Service  │ │   Service      │  │ │
│  │  └──────────┘ └────┬─────┘ └────┬─────┘ └───────┬────────┘  │ │
│  └─────────────────────┼────────────┼───────────────┼───────────┘ │
│                        │            │               │             │
│  ┌─────────────────────┼────────────┼───────────────┼───────────┐ │
│  │                     ▼            ▼               ▼           │ │
│  │               Infrastructure Layer                            │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │ │
│  │  │ OpenAI   │ │ Blog     │ │ Storage  │ │   Credential   │  │ │
│  │  │ Client   │ │ Publisher│ │ (SQLite) │ │   Manager      │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                    │            │               │
                    ▼            ▼               ▼
            ┌──────────┐ ┌──────────┐ ┌────────────────┐
            │ OpenAI   │ │ Blog     │ │ OS Keychain    │
            │ API      │ │ Services │ │ (keyring)      │
            └──────────┘ └──────────┘ └────────────────┘
```

---

## 2. レイヤー構成

### 2.1 Presentation Layer（GUI層）

CustomTkinterによるUI表示とユーザーインタラクションの処理。

**責務**:
- ウィジェットの配置と表示
- ユーザーイベントの受付
- Controller層への処理委譲
- 画面遷移管理

**禁止事項**:
- ビジネスロジックの記述
- 直接的なAPI呼び出し
- データベースアクセス

### 2.2 Controller Layer（コントローラ層）

GUI層とService層の仲介。入力のバリデーションとフロー制御。

**責務**:
- 入力値のバリデーション
- Service層の呼び出し
- GUI層へのデータ変換（ViewModel）
- エラーハンドリング

### 2.3 Service Layer（サービス層）

ビジネスロジックの実装。

**責務**:
- ヒアリングロジック
- 記事生成ロジック
- 記事変換ロジック
- 下書き管理

### 2.4 Infrastructure Layer（インフラ層）

外部サービスとの通信、データ永続化。

**責務**:
- OpenAI API通信
- ブログサービスAPI通信
- ローカルDB操作
- 認証情報管理

---

## 3. パッケージ構成

```
src/
└── postblog/
    ├── __init__.py
    ├── app.py                      # アプリケーションエントリーポイント
    ├── config.py                   # アプリ設定管理
    ├── logging_config.py           # ログ設定
    │
    ├── gui/                        # Presentation Layer
    │   ├── __init__.py
    │   ├── app_window.py           # メインウィンドウ
    │   ├── navigation.py           # 画面遷移マネージャー
    │   ├── components/             # 共通UIコンポーネント
    │   │   ├── __init__.py
    │   │   ├── sidebar.py          # サイドバー
    │   │   ├── statusbar.py        # ステータスバー
    │   │   ├── chat_bubble.py      # チャット吹き出し
    │   │   ├── markdown_editor.py  # Markdownエディタ
    │   │   ├── markdown_preview.py # Markdownプレビュー
    │   │   ├── tag_input.py        # タグ入力
    │   │   └── dialog.py           # 共通ダイアログ
    │   └── views/                  # 各画面
    │       ├── __init__.py
    │       ├── home_view.py        # SCR-001: ホーム画面
    │       ├── blog_type_view.py   # SCR-002: ブログ種別選択
    │       ├── hearing_view.py     # SCR-003: ヒアリング画面
    │       ├── summary_view.py     # SCR-004: サマリー確認
    │       ├── editor_view.py      # SCR-005: 記事編集
    │       ├── publish_view.py     # SCR-006: 投稿確認
    │       ├── result_view.py      # SCR-007: 投稿結果
    │       ├── service_view.py     # SCR-008: サービス管理
    │       └── settings_view.py    # SCR-009: アプリ設定
    │
    ├── controllers/                # Controller Layer
    │   ├── __init__.py
    │   ├── home_controller.py
    │   ├── hearing_controller.py
    │   ├── article_controller.py
    │   ├── publish_controller.py
    │   └── settings_controller.py
    │
    ├── services/                   # Service Layer
    │   ├── __init__.py
    │   ├── hearing_service.py      # ヒアリングロジック
    │   ├── article_service.py      # 記事生成・変換ロジック（SEO対策ポイント解説の解析含む）
    │   ├── seo_service.py          # SEO分析・最適化ロジック
    │   ├── draft_service.py        # 下書き管理
    │   ├── publish_service.py      # 投稿ロジック
    │   └── history_service.py      # 投稿履歴管理
    │
    ├── infrastructure/             # Infrastructure Layer
    │   ├── __init__.py
    │   ├── llm/                    # LLMクライアント
    │   │   ├── __init__.py
    │   │   ├── base.py             # 抽象ベースクラス
    │   │   └── openai_client.py    # OpenAI API実装
    │   ├── publishers/             # ブログ投稿クライアント
    │   │   ├── __init__.py
    │   │   ├── base.py             # 抽象ベースクラス
    │   │   ├── qiita.py            # Qiita API
    │   │   ├── zenn.py             # Zenn (GitHub連携)
    │   │   ├── wordpress.py        # WordPress REST API
    │   │   ├── hatena.py           # はてなブログ AtomPub API
    │   │   ├── ameba.py            # Ameba Blog（メール投稿）
    │   │   └── markdown_export.py  # Markdownエクスポート（note/Wantedly用）
    │   ├── storage/                # データ永続化
    │   │   ├── __init__.py
    │   │   ├── database.py         # SQLite接続管理
    │   │   ├── draft_repository.py # 下書きリポジトリ
    │   │   └── history_repository.py # 投稿履歴リポジトリ
    │   └── credential/             # 認証情報管理
    │       ├── __init__.py
    │       └── credential_manager.py # keyring連携
    │
    ├── models/                     # ドメインモデル
    │   ├── __init__.py
    │   ├── blog_type.py            # ブログ種別
    │   ├── hearing.py              # ヒアリングデータ
    │   ├── article.py              # 記事データ
    │   ├── seo.py                  # SEO分析結果データ + SEO対策ポイント解説データ
    │   ├── draft.py                # 下書きデータ
    │   ├── publish_result.py       # 投稿結果
    │   └── service_config.py       # サービス設定
    │
    └── templates/                  # ヒアリング・記事テンプレート
        ├── __init__.py
        ├── hearing_templates.py    # ヒアリングテンプレート定義
        └── prompts.py              # LLMプロンプトテンプレート
```

---

## 4. クラス設計

### 4.1 ブログ投稿クライアント（プラグインアーキテクチャ）

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PublishRequest:
    """投稿リクエスト。"""
    title: str
    body: str          # Markdown形式
    tags: list[str]
    status: str        # "publish" | "draft"
    blog_type: str


@dataclass
class PublishResult:
    """投稿結果。"""
    success: bool
    service_name: str
    article_url: str | None
    error_message: str | None


class BlogPublisher(ABC):
    """ブログ投稿クライアントの抽象基底クラス。"""

    @property
    @abstractmethod
    def service_name(self) -> str:
        """サービス名を返す。"""

    @abstractmethod
    async def publish(self, request: PublishRequest) -> PublishResult:
        """記事を投稿する。"""

    @abstractmethod
    async def test_connection(self) -> bool:
        """接続テストを実行する。"""

    @abstractmethod
    def convert_content(self, markdown: str) -> str:
        """Markdownをサービス固有のフォーマットに変換する。"""
```

#### サービス別実装

```python
class QiitaPublisher(BlogPublisher):
    """Qiita API v2 クライアント。"""
    # POST https://qiita.com/api/v2/items
    # 認証: Bearer Token
    # 形式: JSON (Markdown)

class ZennPublisher(BlogPublisher):
    """Zenn (GitHub連携) クライアント。"""
    # GitHub APIでMarkdownファイルをリポジトリにPush
    # 認証: GitHub Personal Access Token
    # 形式: Markdown + YAML Front Matter

class WordPressPublisher(BlogPublisher):
    """WordPress REST API クライアント。"""
    # POST /wp-json/wp/v2/posts
    # 認証: Application Passwords (Basic Auth)
    # 形式: HTML (Markdownから変換)

class HatenaPublisher(BlogPublisher):
    """はてなブログ AtomPub API クライアント。"""
    # POST /atom/entry
    # 認証: Basic Auth (はてなID + APIキー)
    # 形式: XML (AtomPub)

class AmebaPublisher(BlogPublisher):
    """Ameba Blog メール投稿クライアント。"""
    # SMTP経由でメール送信
    # 認証: 事前登録メールアドレス + 専用投稿先メールアドレス
    # 形式: HTML (Markdownから変換)
    # 制約: 公開投稿のみ（下書き不可）、テーマ1つのみ

class MarkdownExporter(BlogPublisher):
    """Markdownファイルエクスポート（note/Wantedly用）。"""
    # ローカルにMarkdownファイルとして保存
```

### 4.2 LLMクライアント（抽象化）

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMClient(ABC):
    """LLMクライアントの抽象基底クラス。"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """チャット応答を取得する。"""

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> AsyncIterator[str]:
        """ストリーミングでチャット応答を取得する。"""


class OpenAIClient(LLMClient):
    """OpenAI API クライアント。"""
    # openai ライブラリを使用
    # モデル: gpt-4o
```

### 4.3 画面遷移マネージャー

```python
class NavigationManager:
    """画面遷移を管理する。"""

    def __init__(self, container: ctk.CTkFrame) -> None:
        self._container = container
        self._views: dict[str, type[BaseView]] = {}
        self._current_view: BaseView | None = None
        self._history: list[str] = []

    def register(self, name: str, view_class: type[BaseView]) -> None:
        """画面を登録する。"""

    def navigate(self, name: str, **kwargs: object) -> None:
        """指定画面に遷移する。"""

    def back(self) -> None:
        """前の画面に戻る。"""
```

---

## 5. 非同期処理設計

### 5.1 UIスレッドとバックグラウンド処理

CustomTkinterのメインループはシングルスレッドのため、API通信等の長時間処理はバックグラウンドで実行する。

```
┌────────────────────┐     ┌────────────────────┐
│   Main Thread      │     │  Background Thread │
│   (UI/Tkinter)     │     │  (asyncio)         │
│                    │     │                    │
│  ユーザー操作       │────→│  API呼び出し        │
│                    │     │  DB操作             │
│  UI更新            │←────│  結果返却           │
│  (after()で反映)   │     │                    │
└────────────────────┘     └────────────────────┘
```

### 5.2 実装方針

```python
import asyncio
import threading
from concurrent.futures import Future


class AsyncRunner:
    """バックグラウンドでasyncioタスクを実行する。"""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine) -> Future:
        """コルーチンをバックグラウンドで実行する。"""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
```

### 5.3 ストリーミング表示

LLMからのストリーミング応答をリアルタイムでUIに反映する。

```python
async def on_hearing_response(self, user_message: str) -> None:
    """ヒアリング応答を処理する。"""
    # ストリーミングでLLM応答を取得
    async for chunk in self.llm_client.chat_stream(messages):
        # UIスレッドに反映（tkinterのafter()を使用）
        self.root.after(0, self.view.append_text, chunk)
```

---

## 6. データ設計

### 6.1 SQLiteデータベーススキーマ

```sql
-- 投稿履歴テーブル
CREATE TABLE publish_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    body_preview TEXT,          -- 本文の先頭200文字
    blog_type TEXT NOT NULL,
    service_name TEXT NOT NULL,
    article_url TEXT,
    status TEXT NOT NULL,       -- 'published' | 'draft' | 'failed'
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 下書きテーブル
CREATE TABLE drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    body TEXT NOT NULL,
    tags TEXT,                  -- JSON配列
    blog_type TEXT NOT NULL,
    hearing_data TEXT,          -- JSON (ヒアリング結果)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6.2 設定ファイル構造

保存先: `~/.postblog/config.toml`

```toml
[app]
theme = "dark"
font_size = 14
auto_save_interval = 30    # 秒

[openai]
model = "gpt-4o"
# APIキーはkeyringに保存（ここには保存しない）

[editor]
preview_position = "right"  # "right" | "bottom"
```

### 6.3 データ保存先一覧

| データ | 保存先 | 形式 |
|--------|--------|------|
| アプリ設定 | `~/.postblog/config.toml` | TOML |
| データベース | `~/.postblog/postblog.db` | SQLite |
| 下書き（バックアップ） | `~/.postblog/drafts/` | JSON |
| ログ | `~/.postblog/logs/` | テキスト |
| 認証情報 | OS Keychain（keyring） | 暗号化 |

---

## 7. 外部サービス連携

### 7.1 サービス別API仕様

| サービス | プロトコル | 認証方式 | コンテンツ形式 | 主要ライブラリ |
|----------|----------|----------|--------------|---------------|
| Qiita | REST/JSON | Bearer Token | Markdown | httpx |
| Zenn | Git/GitHub API | GitHub Token | Markdown + Front Matter | httpx + gitpython |
| WordPress | REST/JSON | Basic Auth (Application Passwords) | HTML | httpx |
| はてなブログ | AtomPub/XML | Basic Auth (API Key) | Markdown/XML | httpx |
| Ameba Blog | SMTP/メール | 登録メールアドレス | HTML | smtplib（標準ライブラリ） |
| note | N/A | N/A | Markdownエクスポートのみ | - |
| Wantedly | N/A | N/A | Markdownエクスポートのみ | - |

### 7.2 Markdown変換フロー

```
内部Markdown
    │
    ├──→ Qiita: そのまま送信（Qiita Markdown互換）
    │
    ├──→ Zenn: Front Matter付与 → GitHubリポジトリへPush
    │
    ├──→ WordPress: Markdown → HTML変換 → REST API送信
    │
    ├──→ はてなブログ: Markdown → XML(AtomPub)ラップ → API送信
    │
    ├──→ Ameba Blog: Markdown → HTML変換 → メール本文として送信(SMTP)
    │
    └──→ エクスポート: Front Matter付与 → .mdファイル保存
```

---

## 8. 起動時間最適化

### 8.1 起動シーケンス（目標: 1秒以内）

```
T=0ms    : Pythonインタープリタ起動
T=50ms   : メインモジュール読み込み（最小限のimport）
T=100ms  : CustomTkinter初期化
T=200ms  : メインウィンドウ表示（サイドバー + ホーム画面の骨格）
T=300ms  : ホーム画面のコンテンツ描画
T=500ms  : バックグラウンドで設定読み込み・DB接続
T=700ms  : バックグラウンドでサービス接続状態チェック
T=1000ms : 初期化完了
```

### 8.2 最適化手法

| 手法 | 対象 | 効果 |
|------|------|------|
| 遅延インポート | 各View、publishers | 起動時の読み込み量削減 |
| バックグラウンド初期化 | DB接続、サービス確認 | UI表示を優先 |
| キャッシュ | 設定、テンプレート | 再読み込み不要 |
| 事前コンパイル | .pyc ファイル活用 | モジュール読み込み高速化 |

---

## 9. エラーハンドリング設計

### 9.1 例外階層

```python
class PostBlogError(Exception):
    """PostBlogアプリの基底例外。"""

class AuthenticationError(PostBlogError):
    """認証エラー。"""

class NetworkError(PostBlogError):
    """ネットワークエラー。"""

class PublishError(PostBlogError):
    """投稿エラー。"""

class LLMError(PostBlogError):
    """LLM API エラー。"""

class StorageError(PostBlogError):
    """データ保存エラー。"""

class ValidationError(PostBlogError):
    """入力検証エラー。"""
```

### 9.2 エラー伝播フロー

```
Infrastructure Layer  →  Service Layer  →  Controller Layer  →  GUI Layer
(例外発生)              (ビジネスロジック     (エラーメッセージ     (ダイアログ
                        レベルで変換)        に変換)              表示)
```

---

## 10. 依存関係

### 10.1 主要ライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| customtkinter | >=5.2.0 | GUIフレームワーク |
| openai | >=1.0.0 | OpenAI API クライアント |
| httpx | >=0.27.0 | 非同期HTTPクライアント |
| keyring | >=25.0.0 | OS認証情報管理 |
| mistune | >=3.0.0 | Markdown→HTML変換 |
| tomli / tomllib | 標準ライブラリ(3.11+) | TOML設定ファイル読み込み |
| tomli-w | >=1.0.0 | TOML設定ファイル書き込み |
| gitpython | >=3.1.0 | Git操作（Zenn連携用） |
| Pillow | >=10.0.0 | 画像処理（UI用） |

### 10.2 開発用ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| pytest | テスト |
| pytest-cov | カバレッジ |
| pytest-asyncio | 非同期テスト |
| mypy | 型チェック |
| ruff | リンター/フォーマッター |

---

## 11. セキュリティ設計

### 11.1 認証情報の保護フロー

```
ユーザー入力 → バリデーション → keyring保存
                                    ↓
                              OS Keychain
                              (Windows: Credential Manager)
                              (macOS: Keychain Access)
                              (Linux: Secret Service)
```

### 11.2 APIキーの取り扱い

- APIキーは設定ファイルに保存しない
- メモリ上のAPIキーは使用後にクリアしない（keyringが管理）
- ログにAPIキーを出力しない
- 画面上ではマスク表示（`sk-...xxxx`）

---

## 12. テスト戦略

### 12.1 テスト構成

```
tests/
├── conftest.py                    # 共通フィクスチャ
├── unit/                          # ユニットテスト
│   ├── controllers/
│   ├── services/
│   │   ├── test_seo_service.py
│   │   ├── test_article_service.py   # SEO対策ポイント解説の解析テスト含む
│   │   └── ...
│   ├── infrastructure/
│   │   ├── test_openai_client.py
│   │   ├── test_qiita_publisher.py
│   │   ├── test_zenn_publisher.py
│   │   ├── test_wordpress_publisher.py
│   │   ├── test_hatena_publisher.py
│   │   ├── test_ameba_publisher.py
│   │   └── test_credential_manager.py
│   └── models/
├── integration/                   # 統合テスト
│   ├── test_hearing_flow.py
│   └── test_publish_flow.py
└── gui/                           # GUIロジックテスト
    ├── test_navigation.py
    └── test_view_models.py
```

### 12.2 テスト方針

| レイヤー | テスト対象 | モック対象 |
|---------|-----------|-----------|
| Controller | バリデーション、フロー制御 | Service層 |
| Service | ビジネスロジック（SEO分析含む） | Infrastructure層 |
| Service/SEO | ルールベース分析の全パターン | なし（純粋なロジック） |
| Service/Article | SEO対策ポイント解説の解析・データ変換 | LLM応答（モック） |
| Infrastructure | API通信、DB操作 | 外部API（httpxモック） |
| Model | データクラス、バリデーション | なし |

---

## 13. 変更履歴

| 日付 | バージョン | 変更内容 | 担当 |
|------|-----------|---------|------|
| 2026-02-19 | 1.0.0 | 初版作成 | - |
| 2026-02-19 | 1.1.0 | Ameba Blog（メール投稿方式）を追加 | - |
| 2026-02-19 | 1.2.0 | SEOサービス・モデルを追加、テスト構成にSEOテストを追加 | - |
| 2026-02-19 | 1.3.0 | SEO対策ポイント解説データをseo.pyに追加、article_serviceに解析責務を追加 | - |
