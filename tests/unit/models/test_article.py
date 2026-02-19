"""Articleモデルのテスト。"""

from datetime import datetime

from postblog.models.article import Article


class TestArticle:
    """Articleのテスト。"""

    def test_create_article_required_fields(self) -> None:
        """必須フィールドのみでArticleが作成されることを確認する。"""
        article = Article(title="テスト記事", body="# テスト\n\n本文です。")

        assert article.title == "テスト記事"
        assert article.body == "# テスト\n\n本文です。"
        assert article.tags == []
        assert article.blog_type_id == ""
        assert article.seo_keywords == ""
        assert article.seo_target_audience == ""
        assert article.seo_search_intent == ""
        assert article.meta_description == ""
        assert isinstance(article.created_at, datetime)
        assert isinstance(article.updated_at, datetime)

    def test_create_article_full(self) -> None:
        """全フィールド指定でArticleが作成されることを確認する。"""
        now = datetime(2024, 6, 1, 12, 0, 0)
        article = Article(
            title="Python入門",
            body="# Python入門\n\nPythonについて解説します。",
            tags=["Python", "入門", "プログラミング"],
            blog_type_id="tech",
            seo_keywords="Python 入門 初心者",
            seo_target_audience="プログラミング初心者",
            seo_search_intent="Pythonの基本を学びたい",
            meta_description="Python入門者向けの解説記事です。",
            created_at=now,
            updated_at=now,
        )

        assert article.title == "Python入門"
        assert len(article.tags) == 3
        assert article.blog_type_id == "tech"
        assert article.seo_keywords == "Python 入門 初心者"
        assert article.meta_description == "Python入門者向けの解説記事です。"
        assert article.created_at == now

    def test_article_is_mutable(self) -> None:
        """Articleがミュータブルであることを確認する。"""
        article = Article(title="元のタイトル", body="元の本文")

        article.title = "変更後のタイトル"
        article.body = "変更後の本文"
        article.tags.append("新しいタグ")

        assert article.title == "変更後のタイトル"
        assert article.body == "変更後の本文"
        assert article.tags == ["新しいタグ"]

    def test_article_tags_are_independent(self) -> None:
        """デフォルトのタグリストが独立していることを確認する。"""
        article1 = Article(title="記事1", body="本文1")
        article2 = Article(title="記事2", body="本文2")

        article1.tags.append("tag1")

        assert len(article2.tags) == 0

    def test_article_timestamps_default_to_now(self) -> None:
        """タイムスタンプがデフォルトで現在時刻になることを確認する。"""
        before = datetime.now()
        article = Article(title="テスト", body="本文")
        after = datetime.now()

        assert before <= article.created_at <= after
        assert before <= article.updated_at <= after
