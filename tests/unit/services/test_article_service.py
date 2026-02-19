"""記事生成サービスのテスト。"""

import json

from postblog.services.article_service import (
    _extract_title,
    _parse_seo_advice,
    parse_article_response,
)
from postblog.templates.prompts import SEO_ADVICE_END_MARKER, SEO_ADVICE_START_MARKER


class TestParseArticleResponse:
    """parse_article_response関数のテスト。"""

    def test_parse_with_seo_advice(self) -> None:
        """SEO対策ポイント付きのレスポンスをパースできることを確認する。"""
        advice_json = json.dumps(
            {
                "items": [
                    {
                        "category": "タイトル",
                        "point": "キーワード配置",
                        "reason": "視認性向上",
                        "edit_tip": "先頭に保持",
                    }
                ],
                "summary": "SEO対策済み",
                "target_keyword": "Python入門",
                "generated_at": "2024-01-01T00:00:00",
            }
        )

        response = f"# テスト記事\n\n本文です。\n\n{SEO_ADVICE_START_MARKER}\n{advice_json}\n{SEO_ADVICE_END_MARKER}"

        article_body, seo_advice = parse_article_response(response)

        assert "# テスト記事" in article_body
        assert "本文です。" in article_body
        assert len(seo_advice.items) == 1
        assert seo_advice.items[0].category == "タイトル"
        assert seo_advice.summary == "SEO対策済み"

    def test_parse_without_seo_advice(self) -> None:
        """SEO対策ポイントなしのレスポンスをパースできることを確認する。"""
        response = "# テスト記事\n\n本文です。"

        article_body, seo_advice = parse_article_response(response)

        assert "# テスト記事" in article_body
        assert seo_advice.items == []

    def test_parse_with_invalid_json(self) -> None:
        """不正なJSONでもエラーにならないことを確認する。"""
        response = f"# テスト\n\n{SEO_ADVICE_START_MARKER}\n{{invalid json}}\n{SEO_ADVICE_END_MARKER}"

        article_body, seo_advice = parse_article_response(response)

        assert "# テスト" in article_body
        assert seo_advice.items == []


class TestParseSeoAdvice:
    """_parse_seo_advice関数のテスト。"""

    def test_parse_valid_json(self) -> None:
        """有効なJSONがパースされることを確認する。"""
        data = {
            "items": [
                {"category": "c1", "point": "p1", "reason": "r1", "edit_tip": "e1"},
                {"category": "c2", "point": "p2", "reason": "r2", "edit_tip": "e2"},
            ],
            "summary": "まとめ",
            "target_keyword": "kw",
            "generated_at": "2024-01-01",
        }
        result = _parse_seo_advice(json.dumps(data))

        assert len(result.items) == 2
        assert result.items[0].category == "c1"
        assert result.summary == "まとめ"

    def test_parse_invalid_json(self) -> None:
        """不正なJSONで空のSeoAdviceが返されることを確認する。"""
        result = _parse_seo_advice("invalid json")

        assert result.items == []
        assert result.summary == ""

    def test_parse_empty_items(self) -> None:
        """空のitems配列のパースを確認する。"""
        result = _parse_seo_advice('{"items": [], "summary": ""}')

        assert result.items == []


class TestExtractTitle:
    """_extract_title関数のテスト。"""

    def test_extract_h1(self) -> None:
        """H1タイトルが抽出されることを確認する。"""
        body = "# Python入門ガイド\n\n本文です。"
        assert _extract_title(body) == "Python入門ガイド"

    def test_extract_h1_skip_h2(self) -> None:
        """H2がH1として抽出されないことを確認する。"""
        body = "## サブタイトル\n\n本文です。"
        assert _extract_title(body) == ""

    def test_extract_no_heading(self) -> None:
        """見出しがない場合に空文字列が返されることを確認する。"""
        body = "見出しなしの本文です。"
        assert _extract_title(body) == ""

    def test_extract_first_h1(self) -> None:
        """複数のH1がある場合、最初のH1が抽出されることを確認する。"""
        body = "# 最初のタイトル\n\n## 中間\n\n# 二番目のタイトル"
        assert _extract_title(body) == "最初のタイトル"
