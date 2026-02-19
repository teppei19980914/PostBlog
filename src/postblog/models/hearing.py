"""ヒアリングデータモデル。"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HearingMessage:
    """ヒアリングチャットの1メッセージ。

    Args:
        role: メッセージの送信者（"user" または "assistant"）。
        content: メッセージ内容。
        timestamp: 送信日時。
    """

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HearingResult:
    """ヒアリング結果。

    Args:
        blog_type_id: ブログ種別ID。
        messages: チャットメッセージ履歴。
        answers: ヒアリング項目キーと回答のマップ。
        seo_keywords: ターゲットキーワード。
        seo_target_audience: 想定読者。
        seo_search_intent: 検索意図。
        summary: ヒアリングサマリー。
        completed: ヒアリング完了フラグ。
    """

    blog_type_id: str
    messages: list[HearingMessage] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)
    seo_keywords: str = ""
    seo_target_audience: str = ""
    seo_search_intent: str = ""
    summary: str = ""
    completed: bool = False
