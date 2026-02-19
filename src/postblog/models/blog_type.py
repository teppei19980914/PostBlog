"""ブログ種別のデータモデル。"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HearingItem:
    """ヒアリング項目の定義。

    Args:
        order: 表示順序。
        key: 項目キー。
        question: 質問文。
        required: 必須フラグ。
        description: ヒント文。
    """

    order: int
    key: str
    question: str
    required: bool
    description: str


@dataclass(frozen=True)
class BlogType:
    """ブログ種別の定義。

    Args:
        id: 種別ID（例: "tech"）。
        name: 表示名（例: "技術ブログ"）。
        description: 説明文。
        hearing_policy: ヒアリング方針。
        hearing_items: ヒアリング項目リスト。
        system_prompt: LLMシステムプロンプト。
        article_template: 記事テンプレート。
    """

    id: str
    name: str
    description: str
    hearing_policy: str
    hearing_items: tuple[HearingItem, ...] = field(default_factory=tuple)
    system_prompt: str = ""
    article_template: str = ""
