# PostBlog - 高速ブログ投稿支援アプリ

インプットした知識を最速でブログ記事としてアウトプットするためのデスクトップアプリケーション。

AIとの対話形式のヒアリングで記事の素材を収集し、SEO対策された記事を自動生成。
複数のブログサービスへの同時投稿をサポートします。

## 主な機能

- **AIヒアリング**: 対話形式で記事のテーマ・内容を深掘り（5種類のブログタイプ対応）
- **記事自動生成**: ヒアリング結果からSEO対策されたMarkdown記事を生成
- **SEO分析**: 13項目のルールベース分析で記事品質をスコアリング（100点満点）
- **SEO対策ポイント**: LLMによるSEO改善アドバイスを自動生成
- **Markdownエディタ**: リアルタイムプレビュー付きの記事編集
- **下書き保存**: 作業途中の記事をローカルDBに保存・復元
- **マルチサービス投稿**: Qiita, Zenn, WordPress, はてなブログ, Ameba, Markdownエクスポートに対応
- **投稿履歴管理**: 投稿結果の記録と閲覧

## 対応ブログタイプ

| タイプ | 説明 |
|--------|------|
| 技術ブログ | プログラミング、ツール、技術トピック |
| 日記・エッセイ | 日常の出来事や考え |
| レビュー記事 | 製品、サービス、書籍のレビュー |
| ニュース解説 | 最新ニュースやトレンドの解説 |
| ハウツー記事 | 手順や方法の解説 |

## 必要条件

- Python 3.12 以上
- [uv](https://docs.astral.sh/uv/)（推奨）または pip
- OpenAI API キー

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-username/PostBlog.git
cd PostBlog
```

### 2. 依存関係のインストール

**uv を使用する場合（推奨）:**

```bash
uv sync --dev
```

**pip を使用する場合:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

## 使い方

### アプリケーションの起動

```bash
uv run postblog
```

### 基本的なワークフロー

1. **ブログタイプを選択** — 技術ブログ、日記、レビューなど5種類から選択
2. **AIヒアリング** — 対話形式で記事の素材を収集
3. **サマリー確認** — ヒアリング内容のまとめを確認
4. **記事生成** — AIが記事を自動生成（SEO対策ポイント付き）
5. **エディタで編集** — Markdownエディタでプレビューしながら調整
6. **投稿先を選択** — Qiita、Zennなど複数サービスに同時投稿可能
7. **投稿結果確認** — 各サービスへの投稿結果を確認

### 設定

初回起動時にOpenAI APIキーの設定が必要です。
設定画面（Settings）から以下を設定できます：

- **AIモデル**: GPT-4o / GPT-4o-mini / GPT-4-turbo
- **テーマ**: ライト / ダーク
- **フォントサイズ**: 8〜32pt
- **自動保存間隔**: 10〜300秒
- **各ブログサービスの認証情報**

設定ファイルの保存先: `~/.postblog/config.toml`

## アーキテクチャ

4層アーキテクチャを採用しています：

```
┌─────────────────────────────────┐
│       GUI Layer (View)          │  CustomTkinter - 9画面
├─────────────────────────────────┤
│     Controller Layer            │  5コントローラ
├─────────────────────────────────┤
│       Service Layer             │  ビジネスロジック
├─────────────────────────────────┤
│    Infrastructure Layer         │  DB, LLM, Publisher
└─────────────────────────────────┘
```

## プロジェクト構造

```
src/postblog/
├── app.py                    # エントリーポイント
├── config.py                 # アプリケーション設定
├── exceptions.py             # カスタム例外
├── logging_config.py         # ログ設定
├── models/                   # データモデル（dataclass）
│   ├── article.py
│   ├── blog_type.py
│   ├── draft.py
│   ├── hearing.py
│   ├── publish_result.py
│   ├── seo.py
│   └── service_config.py
├── services/                 # ビジネスロジック
│   ├── article_service.py    # 記事生成
│   ├── draft_service.py      # 下書き管理
│   ├── hearing_service.py    # ヒアリング管理
│   ├── history_service.py    # 投稿履歴管理
│   ├── publish_service.py    # 投稿管理
│   └── seo_service.py        # SEO分析（13項目）
├── controllers/              # UIとサービスのブリッジ
│   ├── article_controller.py
│   ├── hearing_controller.py
│   ├── home_controller.py
│   ├── publish_controller.py
│   └── settings_controller.py
├── infrastructure/           # 外部リソースアクセス
│   ├── async_runner.py       # 非同期ブリッジ
│   ├── credential/           # 認証情報管理
│   ├── llm/                  # LLMクライアント（OpenAI）
│   ├── publishers/           # ブログ投稿クライアント
│   │   ├── qiita.py
│   │   ├── zenn.py
│   │   ├── wordpress.py
│   │   ├── hatena.py
│   │   ├── ameba.py
│   │   └── markdown_export.py
│   └── storage/              # SQLiteデータベース
├── templates/                # プロンプト・ヒアリングテンプレート
└── gui/                      # CustomTkinter GUI
    ├── app_window.py         # メインウィンドウ
    ├── navigation.py         # 画面遷移管理
    ├── components/           # 共通UIコンポーネント
    └── views/                # 9画面ビュー
```

## 開発コマンド

```bash
# リンター実行
uv run ruff check .

# フォーマッター実行
uv run ruff format .

# 型チェック
uv run mypy src/

# テスト実行
uv run pytest

# テスト（カバレッジ付き）
uv run pytest --cov=src/postblog --cov-report=term-missing

# 全チェック
uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest --cov=src/postblog
```

## テスト

テストは3層構造で整理されています：

| ディレクトリ | 内容 |
|-------------|------|
| `tests/unit/models/` | データモデルのテスト |
| `tests/unit/services/` | サービス層のテスト |
| `tests/unit/infrastructure/` | インフラ層のテスト |
| `tests/unit/controllers/` | コントローラ層のテスト |
| `tests/integration/` | ワークフロー統合テスト |

合計 353テスト、カバレッジ 98%+（GUI層を除く）

## ログ

ログファイルの保存先: `~/.postblog/logs/`

- ファイル名: `app_YYYY-MM-DD.log`（日付ローテーション）
- 保持期間: 30日間
- 最大サイズ: 10MB

## コーディング規約

- **PEP 8 準拠**: ruff による自動フォーマット
- **型ヒント必須**: すべての関数に型アノテーション
- **docstring**: Google スタイル
- **コミットメッセージ**: [Conventional Commits](https://www.conventionalcommits.org/)

詳細は [CLAUDE.md](CLAUDE.md) を参照してください。

## ライセンス

MIT License
