"""ブログ種別ごとのヒアリングテンプレート定義。"""

from postblog.models.blog_type import BlogType, HearingItem


TECH_BLOG = BlogType(
    id="tech",
    name="技術ブログ",
    description="プログラミング、ツール、技術トピックに関する記事",
    hearing_policy="技術的な内容を深掘りし、読者が実践できるレベルまでヒアリングする。",
    hearing_items=(
        HearingItem(
            order=1,
            key="topic",
            question="どのような技術テーマについて書きたいですか？",
            required=True,
            description="言語、フレームワーク、ツールなど",
        ),
        HearingItem(
            order=2,
            key="target_reader",
            question="想定読者のスキルレベルは？",
            required=True,
            description="初心者/中級者/上級者",
        ),
        HearingItem(
            order=3,
            key="problem",
            question="この記事で解決する課題や問題は何ですか？",
            required=True,
            description="読者が抱えている課題",
        ),
        HearingItem(
            order=4,
            key="solution",
            question="具体的な解決方法やアプローチを教えてください。",
            required=True,
            description="コード例や手順",
        ),
        HearingItem(
            order=5,
            key="environment",
            question="実行環境やバージョン情報はありますか？",
            required=False,
            description="OS、言語バージョンなど",
        ),
    ),
    system_prompt="あなたは経験豊富な技術ライターです。読者が実践できる具体的で正確な技術記事を作成してください。",
)

DIARY_BLOG = BlogType(
    id="diary",
    name="日記・エッセイ",
    description="日常の出来事や考えを共有する記事",
    hearing_policy="筆者の体験や感情を自然に引き出し、読者が共感できるストーリーを構成する。",
    hearing_items=(
        HearingItem(
            order=1,
            key="topic",
            question="何について書きたいですか？",
            required=True,
            description="テーマや出来事",
        ),
        HearingItem(
            order=2,
            key="experience",
            question="具体的にどのような体験でしたか？",
            required=True,
            description="エピソードの詳細",
        ),
        HearingItem(
            order=3,
            key="feeling",
            question="その時どのように感じましたか？",
            required=False,
            description="感想や気づき",
        ),
        HearingItem(
            order=4,
            key="message",
            question="読者に伝えたいメッセージはありますか？",
            required=False,
            description="結論や学び",
        ),
    ),
    system_prompt="あなたは共感力の高いエッセイストです。読者の心に響く、温かみのある文章を作成してください。",
)

REVIEW_BLOG = BlogType(
    id="review",
    name="レビュー記事",
    description="製品、サービス、書籍などのレビュー記事",
    hearing_policy="対象の良い点・悪い点を公平に評価し、読者の購入判断に役立つ情報を収集する。",
    hearing_items=(
        HearingItem(
            order=1,
            key="product",
            question="レビュー対象は何ですか？",
            required=True,
            description="製品名やサービス名",
        ),
        HearingItem(
            order=2,
            key="usage_period",
            question="どのくらいの期間使用しましたか？",
            required=True,
            description="使用期間",
        ),
        HearingItem(
            order=3,
            key="pros",
            question="良かった点を教えてください。",
            required=True,
            description="メリット、優れている点",
        ),
        HearingItem(
            order=4,
            key="cons",
            question="改善してほしい点はありますか？",
            required=True,
            description="デメリット、不満点",
        ),
        HearingItem(
            order=5,
            key="recommendation",
            question="おすすめ度は？どんな人におすすめですか？",
            required=False,
            description="5段階評価、対象読者",
        ),
    ),
    system_prompt="あなたは公平なレビュアーです。読者の判断に役立つ、バランスの取れたレビュー記事を作成してください。",
)

NEWS_BLOG = BlogType(
    id="news",
    name="ニュース解説",
    description="最新ニュースやトレンドの解説記事",
    hearing_policy="事実に基づいた正確な情報と、独自の分析・考察を引き出す。",
    hearing_items=(
        HearingItem(
            order=1,
            key="topic",
            question="どのニュースやトレンドについて書きたいですか？",
            required=True,
            description="具体的なニュース",
        ),
        HearingItem(
            order=2,
            key="background",
            question="このニュースの背景を教えてください。",
            required=True,
            description="経緯や前提知識",
        ),
        HearingItem(
            order=3,
            key="analysis",
            question="あなたの分析や見解を教えてください。",
            required=True,
            description="独自の考察",
        ),
        HearingItem(
            order=4,
            key="impact",
            question="今後の影響や展望はありますか？",
            required=False,
            description="将来への影響",
        ),
    ),
    system_prompt="あなたは経験豊富なジャーナリストです。正確で分かりやすいニュース解説記事を作成してください。",
)

HOWTO_BLOG = BlogType(
    id="howto",
    name="ハウツー記事",
    description="手順や方法を解説する記事",
    hearing_policy="読者が迷わずに目標を達成できるよう、ステップバイステップの情報を収集する。",
    hearing_items=(
        HearingItem(
            order=1,
            key="goal",
            question="読者がこの記事で達成したいことは何ですか？",
            required=True,
            description="最終的なゴール",
        ),
        HearingItem(
            order=2,
            key="prerequisites",
            question="事前に必要な知識や準備はありますか？",
            required=True,
            description="前提条件",
        ),
        HearingItem(
            order=3,
            key="steps",
            question="具体的な手順を教えてください。",
            required=True,
            description="ステップバイステップの手順",
        ),
        HearingItem(
            order=4,
            key="tips",
            question="注意点やコツはありますか？",
            required=False,
            description="ヒントやよくある間違い",
        ),
    ),
    system_prompt="あなたは分かりやすい解説者です。初心者でも迷わない、ステップバイステップのハウツー記事を作成してください。",
)


# 全ブログ種別のマップ
BLOG_TYPES: dict[str, BlogType] = {
    TECH_BLOG.id: TECH_BLOG,
    DIARY_BLOG.id: DIARY_BLOG,
    REVIEW_BLOG.id: REVIEW_BLOG,
    NEWS_BLOG.id: NEWS_BLOG,
    HOWTO_BLOG.id: HOWTO_BLOG,
}


def get_blog_type(type_id: str) -> BlogType | None:
    """ブログ種別IDからBlogTypeを取得する。

    Args:
        type_id: ブログ種別ID。

    Returns:
        BlogType。見つからない場合はNone。
    """
    return BLOG_TYPES.get(type_id)


def get_all_blog_types() -> list[BlogType]:
    """全てのブログ種別を取得する。

    Returns:
        BlogTypeのリスト。
    """
    return list(BLOG_TYPES.values())
