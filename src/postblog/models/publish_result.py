"""投稿リクエスト・投稿結果のデータモデル。"""

from dataclasses import dataclass, field


@dataclass
class PublishRequest:
    """投稿リクエスト。

    Args:
        title: 記事タイトル。
        body: 記事本文（Markdown形式）。
        tags: タグリスト。
        status: 投稿ステータス（"publish" または "draft"）。
        blog_type_id: ブログ種別ID。
    """

    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    status: str = "publish"
    blog_type_id: str = ""


@dataclass
class PublishResult:
    """投稿結果。

    Args:
        success: 投稿成功フラグ。
        service_name: サービス名。
        article_url: 記事URL（成功時）。
        error_message: エラーメッセージ（失敗時）。
    """

    success: bool
    service_name: str
    article_url: str | None = None
    error_message: str | None = None
