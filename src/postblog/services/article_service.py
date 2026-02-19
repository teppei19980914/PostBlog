"""記事生成サービス。

LLMを使用した記事生成とSEO対策ポイント解説のパースを提供する。
"""

import json
import logging
from datetime import datetime

from postblog.infrastructure.llm.base import LLMClient
from postblog.models.article import Article
from postblog.models.hearing import HearingResult
from postblog.models.seo import SeoAdvice, SeoAdviceItem
from postblog.templates.prompts import (
    ARTICLE_GENERATION_PROMPT,
    SEO_ADVICE_END_MARKER,
    SEO_ADVICE_START_MARKER,
)


logger = logging.getLogger(__name__)


class ArticleService:
    """記事生成サービス。

    Args:
        llm_client: LLMクライアント。
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def generate(
        self, hearing_result: HearingResult
    ) -> tuple[Article, SeoAdvice]:
        """ヒアリング結果から記事とSEO対策ポイントを生成する。

        Args:
            hearing_result: ヒアリング結果。

        Returns:
            (Article, SeoAdvice) のタプル。
        """
        prompt = ARTICLE_GENERATION_PROMPT.format(
            hearing_summary=hearing_result.summary,
            seo_keywords=hearing_result.seo_keywords,
            seo_target_audience=hearing_result.seo_target_audience,
            seo_search_intent=hearing_result.seo_search_intent,
        )

        messages = [
            {
                "role": "system",
                "content": "あなたはSEO対策に詳しいプロのブログライターです。",
            },
            {"role": "user", "content": prompt},
        ]

        response = await self._llm.chat(messages)
        article_body, seo_advice = parse_article_response(response)

        article = Article(
            title=_extract_title(article_body),
            body=article_body,
            blog_type_id=hearing_result.blog_type_id,
            seo_keywords=hearing_result.seo_keywords,
            seo_target_audience=hearing_result.seo_target_audience,
            seo_search_intent=hearing_result.seo_search_intent,
        )

        logger.info("記事を生成しました: title=%s", article.title)
        return article, seo_advice


def parse_article_response(response: str) -> tuple[str, SeoAdvice]:
    """LLMレスポンスから記事本文とSEO対策ポイントを分離する。

    Args:
        response: LLMの応答全文。

    Returns:
        (記事本文, SeoAdvice) のタプル。
    """
    if SEO_ADVICE_START_MARKER in response and SEO_ADVICE_END_MARKER in response:
        start = response.index(SEO_ADVICE_START_MARKER)
        end = response.index(SEO_ADVICE_END_MARKER) + len(SEO_ADVICE_END_MARKER)

        article_body = response[:start].strip()
        advice_json = response[
            start + len(SEO_ADVICE_START_MARKER) : end - len(SEO_ADVICE_END_MARKER)
        ].strip()

        seo_advice = _parse_seo_advice(advice_json)
        return article_body, seo_advice

    # マーカーがない場合は全文を記事本文として扱う
    return response.strip(), SeoAdvice()


def _parse_seo_advice(json_str: str) -> SeoAdvice:
    """JSON文字列からSeoAdviceをパースする。

    Args:
        json_str: JSON文字列。

    Returns:
        SeoAdviceインスタンス。
    """
    try:
        data = json.loads(json_str)
        items = [
            SeoAdviceItem(
                category=item.get("category", ""),
                point=item.get("point", ""),
                reason=item.get("reason", ""),
                edit_tip=item.get("edit_tip", ""),
            )
            for item in data.get("items", [])
        ]
        return SeoAdvice(
            items=items,
            summary=data.get("summary", ""),
            target_keyword=data.get("target_keyword", ""),
            generated_at=data.get("generated_at", datetime.now().isoformat()),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("SEO対策ポイントのパースに失敗しました")
        return SeoAdvice()


def _extract_title(body: str) -> str:
    """Markdown本文からタイトル（H1）を抽出する。

    Args:
        body: Markdown本文。

    Returns:
        タイトル文字列。H1がない場合は空文字列。
    """
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return ""
