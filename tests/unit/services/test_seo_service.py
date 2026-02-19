"""SEO分析サービスのテスト。"""

from postblog.services.seo_service import analyze_seo


GOOD_ARTICLE = """# Python入門ガイド

Python入門として、基本的な文法と使い方を解説します。

## Pythonとは

Pythonは汎用プログラミング言語です。Python入門者にとって、
シンプルな文法は大きな魅力です。データ分析やWeb開発など幅広い用途があります。

### Pythonのインストール

Pythonをインストールするには、公式サイトからダウンロードします。
最新版のPythonを使用することをおすすめします。

## Pythonの基本文法

Pythonの変数宣言は非常にシンプルです。型宣言が不要なため、
初心者でも直感的にコードを書くことができます。

### 変数と型

- 整数型: `int`
- 浮動小数点型: `float`
- 文字列型: `str`

| 型 | 例 |
|---|---|
| int | 42 |
| str | "hello" |

## まとめ

Python入門の基本を学びました。さらに詳しくは
[公式ドキュメント](https://docs.python.org/)を参照してください。
Python入門者が次に学ぶべきは、リストや辞書などのデータ構造です。
Pythonの魅力は、そのシンプルさと強力なライブラリ群にあります。
初心者から上級者まで、Pythonは幅広い開発者に愛されています。
"""


class TestAnalyzeSeo:
    """analyze_seo関数のテスト。"""

    def test_good_article_high_score(self) -> None:
        """良い記事が高スコアを得ることを確認する。"""
        result = analyze_seo(
            title="Python入門ガイド - 初心者向け基本文法解説",
            body=GOOD_ARTICLE,
            keyword="python入門",
            meta_description="Python入門者向けのガイドです。基本的な文法やインストール方法を分かりやすく解説します。初心者でもすぐにPythonを始められます。",
        )

        assert result.score >= 60
        assert len(result.items) == 13

    def test_title_keyword_pass(self) -> None:
        """タイトルにキーワードが含まれる場合のテスト。"""
        result = analyze_seo(
            title="Python入門ガイド", body="本文", keyword="python入門"
        )

        title_item = next(
            i
            for i in result.items
            if i.name == "キーワード含有" and i.category == "タイトル"
        )
        assert title_item.status == "pass"
        assert title_item.score == 15

    def test_title_keyword_fail(self) -> None:
        """タイトルにキーワードが含まれない場合のテスト。"""
        result = analyze_seo(
            title="プログラミング入門", body="本文", keyword="python入門"
        )

        title_item = next(
            i
            for i in result.items
            if i.name == "キーワード含有" and i.category == "タイトル"
        )
        assert title_item.status == "fail"
        assert title_item.score == 0

    def test_title_length_pass(self) -> None:
        """タイトル文字数が適切な場合のテスト。"""
        result = analyze_seo(title="P" * 35, body="本文", keyword="p")

        title_length = next(
            i for i in result.items if i.name == "文字数" and i.category == "タイトル"
        )
        assert title_length.status == "pass"

    def test_title_length_too_short(self) -> None:
        """タイトルが短すぎる場合のテスト。"""
        result = analyze_seo(title="短い", body="本文", keyword="p")

        title_length = next(
            i for i in result.items if i.name == "文字数" and i.category == "タイトル"
        )
        assert title_length.status == "fail"

    def test_meta_description_missing(self) -> None:
        """メタディスクリプションがない場合のテスト。"""
        result = analyze_seo(title="テスト", body="本文", keyword="test")

        meta_exists = next(i for i in result.items if i.name == "存在チェック")
        assert meta_exists.status == "fail"

    def test_meta_description_good_length(self) -> None:
        """メタディスクリプションが適切な長さの場合のテスト。"""
        meta = "A" * 140
        result = analyze_seo(
            title="テスト", body="本文", keyword="test", meta_description=meta
        )

        meta_length = next(
            i
            for i in result.items
            if i.name == "文字数" and i.category == "メタディスクリプション"
        )
        assert meta_length.status == "pass"

    def test_heading_hierarchy_with_h2_and_h3(self) -> None:
        """H2とH3がある場合のテスト。"""
        body = "## 見出し2\n\n内容\n\n### 見出し3\n\n内容"
        result = analyze_seo(title="テスト", body=body, keyword="test")

        heading = next(i for i in result.items if i.name == "見出し階層")
        assert heading.status == "pass"
        assert heading.score == 10

    def test_heading_hierarchy_no_headings(self) -> None:
        """見出しがない場合のテスト。"""
        result = analyze_seo(title="テスト", body="見出しなしの本文", keyword="test")

        heading = next(i for i in result.items if i.name == "見出し階層")
        assert heading.status == "fail"

    def test_body_length_sufficient(self) -> None:
        """本文が十分な長さの場合のテスト。"""
        body = "あ" * 2000
        result = analyze_seo(title="テスト", body=body, keyword="test")

        body_length = next(
            i for i in result.items if i.name == "文字数" and i.category == "本文"
        )
        assert body_length.status == "pass"

    def test_body_length_short(self) -> None:
        """本文が短い場合のテスト。"""
        result = analyze_seo(title="テスト", body="短い本文", keyword="test")

        body_length = next(
            i for i in result.items if i.name == "文字数" and i.category == "本文"
        )
        assert body_length.status == "fail"

    def test_lists_present(self) -> None:
        """リストがある場合のテスト。"""
        body = "## 見出し\n\n- 項目1\n- 項目2\n- 項目3"
        result = analyze_seo(title="テスト", body=body, keyword="test")

        lists = next(i for i in result.items if i.name == "リスト・テーブル")
        assert lists.status == "pass"

    def test_external_links_present(self) -> None:
        """外部リンクがある場合のテスト。"""
        body = "本文 [参考](https://example.com)"
        result = analyze_seo(title="テスト", body=body, keyword="test")

        links = next(i for i in result.items if i.name == "外部リンク")
        assert links.status == "pass"

    def test_external_links_absent(self) -> None:
        """外部リンクがない場合のテスト。"""
        result = analyze_seo(title="テスト", body="リンクなし本文", keyword="test")

        links = next(i for i in result.items if i.name == "外部リンク")
        assert links.status == "warn"

    def test_first_paragraph_keyword(self) -> None:
        """冒頭にキーワードがある場合のテスト。"""
        body = "Python入門について解説します。" + "本文" * 500
        result = analyze_seo(title="テスト", body=body, keyword="python入門")

        first_para = next(i for i in result.items if i.name == "冒頭キーワード")
        assert first_para.status == "pass"

    def test_suggestions_generated_for_failures(self) -> None:
        """失敗項目に改善提案が生成されることを確認する。"""
        result = analyze_seo(title="短い", body="短い", keyword="ないキーワード")

        assert len(result.suggestions) > 0

    def test_score_range(self) -> None:
        """スコアが0-100の範囲内であることを確認する。"""
        result = analyze_seo(title="テスト", body="本文", keyword="test")

        assert 0 <= result.score <= 100

    def test_empty_body(self) -> None:
        """空の本文の場合のテスト。"""
        result = analyze_seo(title="テスト", body="", keyword="test")

        assert result.score >= 0
        body_item = next(i for i in result.items if i.name == "キーワード密度")
        assert body_item.status == "fail"
