"""BlogType・HearingItemモデルのテスト。"""

import pytest

from postblog.models.blog_type import BlogType, HearingItem


class TestHearingItem:
    """HearingItemのテスト。"""

    def test_create_hearing_item(self) -> None:
        """HearingItemが正しく作成されることを確認する。"""
        item = HearingItem(
            order=1,
            key="topic",
            question="何について書きますか？",
            required=True,
            description="記事のメインテーマを入力してください。",
        )

        assert item.order == 1
        assert item.key == "topic"
        assert item.question == "何について書きますか？"
        assert item.required is True
        assert item.description == "記事のメインテーマを入力してください。"

    def test_hearing_item_is_frozen(self) -> None:
        """HearingItemがイミュータブルであることを確認する。"""
        item = HearingItem(
            order=1,
            key="topic",
            question="テスト質問",
            required=True,
            description="説明",
        )

        with pytest.raises(AttributeError):
            item.key = "changed"  # type: ignore[misc]

    def test_hearing_item_equality(self) -> None:
        """同じ値のHearingItemが等しいことを確認する。"""
        item1 = HearingItem(
            order=1, key="a", question="q", required=True, description="d"
        )
        item2 = HearingItem(
            order=1, key="a", question="q", required=True, description="d"
        )

        assert item1 == item2

    def test_hearing_item_inequality(self) -> None:
        """異なる値のHearingItemが等しくないことを確認する。"""
        item1 = HearingItem(
            order=1, key="a", question="q", required=True, description="d"
        )
        item2 = HearingItem(
            order=2, key="b", question="q2", required=False, description="d2"
        )

        assert item1 != item2


class TestBlogType:
    """BlogTypeのテスト。"""

    def test_create_blog_type_minimal(self) -> None:
        """最小限のパラメータでBlogTypeが作成されることを確認する。"""
        blog_type = BlogType(
            id="tech",
            name="技術ブログ",
            description="技術記事",
            hearing_policy="技術的な内容をヒアリングする",
        )

        assert blog_type.id == "tech"
        assert blog_type.name == "技術ブログ"
        assert blog_type.description == "技術記事"
        assert blog_type.hearing_policy == "技術的な内容をヒアリングする"
        assert blog_type.hearing_items == ()
        assert blog_type.system_prompt == ""
        assert blog_type.article_template == ""

    def test_create_blog_type_with_items(self) -> None:
        """ヒアリング項目付きでBlogTypeが作成されることを確認する。"""
        items = (
            HearingItem(
                order=1,
                key="topic",
                question="テーマは？",
                required=True,
                description="",
            ),
            HearingItem(
                order=2,
                key="target",
                question="読者は？",
                required=False,
                description="",
            ),
        )
        blog_type = BlogType(
            id="tech",
            name="技術ブログ",
            description="技術記事",
            hearing_policy="方針",
            hearing_items=items,
            system_prompt="あなたは技術ライターです。",
            article_template="# {title}",
        )

        assert len(blog_type.hearing_items) == 2
        assert blog_type.hearing_items[0].key == "topic"
        assert blog_type.hearing_items[1].key == "target"
        assert blog_type.system_prompt == "あなたは技術ライターです。"
        assert blog_type.article_template == "# {title}"

    def test_blog_type_is_frozen(self) -> None:
        """BlogTypeがイミュータブルであることを確認する。"""
        blog_type = BlogType(
            id="tech",
            name="技術ブログ",
            description="技術記事",
            hearing_policy="方針",
        )

        with pytest.raises(AttributeError):
            blog_type.name = "changed"  # type: ignore[misc]

    def test_blog_type_equality(self) -> None:
        """同じ値のBlogTypeが等しいことを確認する。"""
        bt1 = BlogType(id="tech", name="技術", description="d", hearing_policy="p")
        bt2 = BlogType(id="tech", name="技術", description="d", hearing_policy="p")

        assert bt1 == bt2
