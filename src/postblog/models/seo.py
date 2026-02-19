"""SEO分析・SEO対策ポイント解説のデータモデル。"""

from dataclasses import dataclass, field


@dataclass
class SeoCheckItem:
    """SEOチェック項目の結果。

    Args:
        category: カテゴリ。
        name: 項目名。
        status: 判定結果（"pass" | "warn" | "fail"）。
        score: 獲得スコア。
        max_score: 配点。
        message: 結果メッセージ。
        suggestion: 改善提案（failの場合）。
    """

    category: str
    name: str
    status: str
    score: int
    max_score: int
    message: str
    suggestion: str | None = None


@dataclass
class SeoAnalysisResult:
    """SEO分析結果。

    Args:
        score: 総合スコア（0〜100）。
        items: 各項目の分析結果。
        suggestions: 改善提案リスト。
    """

    score: int
    items: list[SeoCheckItem] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class SeoAdviceItem:
    """SEO対策ポイントの1項目。

    Args:
        category: カテゴリ（タイトル、見出し構造、キーワード配置等）。
        point: 施策内容（何をしたか）。
        reason: 理由・効果（なぜ効果的か）。
        edit_tip: 編集時の注意点（修正時に気をつけること）。
    """

    category: str
    point: str
    reason: str
    edit_tip: str


@dataclass
class SeoAdvice:
    """プロトタイプのSEO対策ポイント解説。

    Args:
        items: SEO対策ポイント一覧。
        summary: SEO対策の全体サマリー。
        target_keyword: ターゲットキーワード。
        generated_at: 生成日時（ISO 8601形式）。
    """

    items: list[SeoAdviceItem] = field(default_factory=list)
    summary: str = ""
    target_keyword: str = ""
    generated_at: str = ""
