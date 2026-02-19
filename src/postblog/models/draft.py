"""下書きデータモデル。"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Draft:
    """下書きデータ。

    Args:
        id: 下書きID（DB自動採番、新規時はNone）。
        title: 記事タイトル。
        body: 記事本文（Markdown形式）。
        tags: タグリスト。
        blog_type_id: ブログ種別ID。
        hearing_data: ヒアリング結果のJSON文字列。
        created_at: 作成日時。
        updated_at: 更新日時。
    """

    id: int | None = None
    title: str = ""
    body: str = ""
    tags: list[str] = field(default_factory=list)
    blog_type_id: str = ""
    hearing_data: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
