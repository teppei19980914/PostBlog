"""記事データモデル。"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Article:
    """記事データ。

    Args:
        title: 記事タイトル。
        body: 記事本文（Markdown形式）。
        tags: タグリスト。
        blog_type_id: ブログ種別ID。
        seo_keywords: SEOターゲットキーワード。
        seo_target_audience: 想定読者。
        seo_search_intent: 検索意図。
        meta_description: メタディスクリプション。
        created_at: 作成日時。
        updated_at: 更新日時。
    """

    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    blog_type_id: str = ""
    seo_keywords: str = ""
    seo_target_audience: str = ""
    seo_search_intent: str = ""
    meta_description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
