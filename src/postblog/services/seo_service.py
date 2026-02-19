"""SEO分析サービス。

ルールベースのSEO分析（13項目、100点満点）を提供する。
"""

import logging
import re

from postblog.models.seo import SeoAnalysisResult, SeoCheckItem


logger = logging.getLogger(__name__)


def analyze_seo(
    title: str,
    body: str,
    keyword: str,
    meta_description: str = "",
) -> SeoAnalysisResult:
    """SEO分析を実行する。

    Args:
        title: 記事タイトル。
        body: 記事本文（Markdown）。
        keyword: ターゲットキーワード。
        meta_description: メタディスクリプション。

    Returns:
        SEO分析結果。
    """
    items: list[SeoCheckItem] = []
    keyword_lower = keyword.lower()

    # 1. タイトルキーワード含有（15点）
    items.append(_check_title_keyword(title, keyword_lower))

    # 2. タイトル文字数（5点）
    items.append(_check_title_length(title))

    # 3. メタディスクリプション存在（5点）
    items.append(_check_meta_description_exists(meta_description))

    # 4. メタディスクリプション文字数（5点）
    items.append(_check_meta_description_length(meta_description))

    # 5. メタディスクリプションキーワード含有（5点）
    items.append(_check_meta_description_keyword(meta_description, keyword_lower))

    # 6. 見出し階層チェック（10点）
    items.append(_check_heading_hierarchy(body))

    # 7. 見出しキーワード含有（5点）
    items.append(_check_heading_keyword(body, keyword_lower))

    # 8. 本文キーワード密度（10点）
    items.append(_check_keyword_density(body, keyword_lower))

    # 9. 本文文字数（10点）
    items.append(_check_body_length(body))

    # 10. 冒頭キーワード含有（5点）
    items.append(_check_first_paragraph_keyword(body, keyword_lower))

    # 11. 段落長チェック（5点）
    items.append(_check_paragraph_length(body))

    # 12. リスト・テーブル使用（5点）
    items.append(_check_lists_tables(body))

    # 13. 外部リンク（5点）— Markdown記法チェック
    items.append(_check_external_links(body))

    total_score = sum(item.score for item in items)
    suggestions = [item.suggestion for item in items if item.suggestion]

    logger.info("SEO分析完了: score=%d", total_score)
    return SeoAnalysisResult(score=total_score, items=items, suggestions=suggestions)


def _check_title_keyword(title: str, keyword: str) -> SeoCheckItem:
    title_lower = title.lower()
    if keyword in title_lower:
        return SeoCheckItem(
            category="タイトル",
            name="キーワード含有",
            status="pass",
            score=15,
            max_score=15,
            message="タイトルにキーワードが含まれています。",
        )
    return SeoCheckItem(
        category="タイトル",
        name="キーワード含有",
        status="fail",
        score=0,
        max_score=15,
        message="タイトルにキーワードが含まれていません。",
        suggestion="タイトルにターゲットキーワードを含めてください。",
    )


def _check_title_length(title: str) -> SeoCheckItem:
    length = len(title)
    if 30 <= length <= 60:
        return SeoCheckItem(
            category="タイトル",
            name="文字数",
            status="pass",
            score=5,
            max_score=5,
            message=f"タイトルの文字数が適切です（{length}文字）。",
        )
    status = "warn" if 20 <= length <= 70 else "fail"
    score = 3 if status == "warn" else 0
    return SeoCheckItem(
        category="タイトル",
        name="文字数",
        status=status,
        score=score,
        max_score=5,
        message=f"タイトルの文字数が{length}文字です（推奨: 30-60文字）。",
        suggestion="タイトルを30-60文字に調整してください。",
    )


def _check_meta_description_exists(meta: str) -> SeoCheckItem:
    if meta.strip():
        return SeoCheckItem(
            category="メタディスクリプション",
            name="存在チェック",
            status="pass",
            score=5,
            max_score=5,
            message="メタディスクリプションが設定されています。",
        )
    return SeoCheckItem(
        category="メタディスクリプション",
        name="存在チェック",
        status="fail",
        score=0,
        max_score=5,
        message="メタディスクリプションが設定されていません。",
        suggestion="120-160文字のメタディスクリプションを設定してください。",
    )


def _check_meta_description_length(meta: str) -> SeoCheckItem:
    length = len(meta.strip())
    if length == 0:
        return SeoCheckItem(
            category="メタディスクリプション",
            name="文字数",
            status="fail",
            score=0,
            max_score=5,
            message="メタディスクリプションがありません。",
            suggestion="120-160文字のメタディスクリプションを作成してください。",
        )
    if 120 <= length <= 160:
        return SeoCheckItem(
            category="メタディスクリプション",
            name="文字数",
            status="pass",
            score=5,
            max_score=5,
            message=f"メタディスクリプションの文字数が適切です（{length}文字）。",
        )
    score = 3 if 80 <= length <= 200 else 0
    return SeoCheckItem(
        category="メタディスクリプション",
        name="文字数",
        status="warn",
        score=score,
        max_score=5,
        message=f"メタディスクリプションが{length}文字です（推奨: 120-160文字）。",
        suggestion="メタディスクリプションを120-160文字に調整してください。",
    )


def _check_meta_description_keyword(meta: str, keyword: str) -> SeoCheckItem:
    if not meta.strip():
        return SeoCheckItem(
            category="メタディスクリプション",
            name="キーワード含有",
            status="fail",
            score=0,
            max_score=5,
            message="メタディスクリプションがないためチェックできません。",
            suggestion="メタディスクリプションにキーワードを含めてください。",
        )
    if keyword in meta.lower():
        return SeoCheckItem(
            category="メタディスクリプション",
            name="キーワード含有",
            status="pass",
            score=5,
            max_score=5,
            message="メタディスクリプションにキーワードが含まれています。",
        )
    return SeoCheckItem(
        category="メタディスクリプション",
        name="キーワード含有",
        status="fail",
        score=0,
        max_score=5,
        message="メタディスクリプションにキーワードが含まれていません。",
        suggestion="メタディスクリプションにターゲットキーワードを含めてください。",
    )


def _check_heading_hierarchy(body: str) -> SeoCheckItem:
    headings = re.findall(r"^(#{1,6})\s", body, re.MULTILINE)
    if not headings:
        return SeoCheckItem(
            category="見出し",
            name="見出し階層",
            status="fail",
            score=0,
            max_score=10,
            message="見出しが使用されていません。",
            suggestion="H2, H3の見出しを使用して記事を構造化してください。",
        )
    levels = [len(h) for h in headings]
    has_h2 = 2 in levels
    has_h3 = 3 in levels
    if has_h2 and has_h3:
        return SeoCheckItem(
            category="見出し",
            name="見出し階層",
            status="pass",
            score=10,
            max_score=10,
            message="H2, H3が適切に使用されています。",
        )
    if has_h2:
        return SeoCheckItem(
            category="見出し",
            name="見出し階層",
            status="warn",
            score=7,
            max_score=10,
            message="H2は使用されていますが、H3が不足しています。",
            suggestion="小見出し（H3）の追加を検討してください。",
        )
    return SeoCheckItem(
        category="見出し",
        name="見出し階層",
        status="warn",
        score=3,
        max_score=10,
        message="見出し構造が不十分です。",
        suggestion="H2, H3の見出しを使用して記事を構造化してください。",
    )


def _check_heading_keyword(body: str, keyword: str) -> SeoCheckItem:
    headings = re.findall(r"^#{1,6}\s+(.+)$", body, re.MULTILINE)
    if not headings:
        return SeoCheckItem(
            category="見出し",
            name="キーワード含有",
            status="fail",
            score=0,
            max_score=5,
            message="見出しがないためチェックできません。",
            suggestion="見出しにキーワードを含めてください。",
        )
    has_keyword = any(keyword in h.lower() for h in headings)
    if has_keyword:
        return SeoCheckItem(
            category="見出し",
            name="キーワード含有",
            status="pass",
            score=5,
            max_score=5,
            message="見出しにキーワードが含まれています。",
        )
    return SeoCheckItem(
        category="見出し",
        name="キーワード含有",
        status="fail",
        score=0,
        max_score=5,
        message="見出しにキーワードが含まれていません。",
        suggestion="少なくとも1つの見出しにキーワードを含めてください。",
    )


def _check_keyword_density(body: str, keyword: str) -> SeoCheckItem:
    # Markdownの記法を除去した本文で計算
    plain_text = re.sub(r"[#*_`\[\]()!|>-]", "", body).lower()
    total_chars = len(plain_text.replace(" ", "").replace("\n", ""))
    if total_chars == 0:
        return SeoCheckItem(
            category="本文",
            name="キーワード密度",
            status="fail",
            score=0,
            max_score=10,
            message="本文がありません。",
            suggestion="本文にキーワードを自然に含めてください。",
        )
    keyword_count = plain_text.count(keyword)
    keyword_chars = len(keyword) * keyword_count
    density = (keyword_chars / total_chars) * 100 if total_chars > 0 else 0

    if 2 <= density <= 5:
        return SeoCheckItem(
            category="本文",
            name="キーワード密度",
            status="pass",
            score=10,
            max_score=10,
            message=f"キーワード密度が適切です（{density:.1f}%）。",
        )
    if 1 <= density < 2 or 5 < density <= 7:
        return SeoCheckItem(
            category="本文",
            name="キーワード密度",
            status="warn",
            score=5,
            max_score=10,
            message=f"キーワード密度が{density:.1f}%です（推奨: 2-5%）。",
            suggestion="キーワード密度を2-5%に調整してください。",
        )
    return SeoCheckItem(
        category="本文",
        name="キーワード密度",
        status="fail",
        score=0,
        max_score=10,
        message=f"キーワード密度が{density:.1f}%です（推奨: 2-5%）。",
        suggestion="キーワードを本文中に自然に追加/削減してください。",
    )


def _check_body_length(body: str) -> SeoCheckItem:
    plain_text = re.sub(r"[#*_`\[\]()!|>-]", "", body)
    length = len(plain_text.replace(" ", "").replace("\n", ""))
    if length >= 1500:
        return SeoCheckItem(
            category="本文",
            name="文字数",
            status="pass",
            score=10,
            max_score=10,
            message=f"本文の文字数が十分です（{length}文字）。",
        )
    if length >= 1000:
        return SeoCheckItem(
            category="本文",
            name="文字数",
            status="warn",
            score=5,
            max_score=10,
            message=f"本文が{length}文字です（推奨: 1500文字以上）。",
            suggestion="もう少し内容を充実させてください（推奨: 1500文字以上）。",
        )
    return SeoCheckItem(
        category="本文",
        name="文字数",
        status="fail",
        score=0,
        max_score=10,
        message=f"本文が{length}文字です（推奨: 1500文字以上）。",
        suggestion="本文を1500文字以上に充実させてください。",
    )


def _check_first_paragraph_keyword(body: str, keyword: str) -> SeoCheckItem:
    first_100 = body[:100].lower()
    if keyword in first_100:
        return SeoCheckItem(
            category="本文",
            name="冒頭キーワード",
            status="pass",
            score=5,
            max_score=5,
            message="冒頭にキーワードが含まれています。",
        )
    return SeoCheckItem(
        category="本文",
        name="冒頭キーワード",
        status="fail",
        score=0,
        max_score=5,
        message="冒頭100文字以内にキーワードが含まれていません。",
        suggestion="記事の冒頭にキーワードを含めてください。",
    )


def _check_paragraph_length(body: str) -> SeoCheckItem:
    paragraphs = [
        p.strip()
        for p in body.split("\n\n")
        if p.strip() and not p.strip().startswith("#")
    ]
    if not paragraphs:
        return SeoCheckItem(
            category="本文",
            name="段落長",
            status="warn",
            score=3,
            max_score=5,
            message="段落が検出されませんでした。",
            suggestion="適切な段落分けを行ってください。",
        )
    # 3-4文を目安（句点で分割）
    long_paragraphs = sum(1 for p in paragraphs if p.count("。") > 5)
    if long_paragraphs == 0:
        return SeoCheckItem(
            category="本文",
            name="段落長",
            status="pass",
            score=5,
            max_score=5,
            message="段落の長さが適切です。",
        )
    return SeoCheckItem(
        category="本文",
        name="段落長",
        status="warn",
        score=3,
        max_score=5,
        message=f"{long_paragraphs}個の段落が長すぎます。",
        suggestion="長い段落は3-4文で区切ってください。",
    )


def _check_lists_tables(body: str) -> SeoCheckItem:
    has_list = bool(re.search(r"^[\s]*[-*+]\s", body, re.MULTILINE))
    has_numbered_list = bool(re.search(r"^[\s]*\d+\.\s", body, re.MULTILINE))
    has_table = "|" in body and "---" in body

    if has_list or has_numbered_list or has_table:
        return SeoCheckItem(
            category="構成要素",
            name="リスト・テーブル",
            status="pass",
            score=5,
            max_score=5,
            message="リストやテーブルが使用されています。",
        )
    return SeoCheckItem(
        category="構成要素",
        name="リスト・テーブル",
        status="warn",
        score=0,
        max_score=5,
        message="リストやテーブルが使用されていません。",
        suggestion="箇条書きリストやテーブルを活用してください。",
    )


def _check_external_links(body: str) -> SeoCheckItem:
    links = re.findall(r"\[.*?\]\(https?://.*?\)", body)
    if links:
        return SeoCheckItem(
            category="構成要素",
            name="外部リンク",
            status="pass",
            score=5,
            max_score=5,
            message=f"{len(links)}個の外部リンクがあります。",
        )
    return SeoCheckItem(
        category="構成要素",
        name="外部リンク",
        status="warn",
        score=0,
        max_score=5,
        message="外部リンクがありません。",
        suggestion="信頼性の高い外部サイトへのリンクを追加してください。",
    )
