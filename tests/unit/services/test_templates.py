"""テンプレートのテスト。"""

from postblog.templates.hearing_templates import (
    BLOG_TYPES,
    get_all_blog_types,
    get_blog_type,
)


class TestHearingTemplates:
    """ヒアリングテンプレートのテスト。"""

    def test_blog_types_count(self) -> None:
        """5種類のブログ種別が定義されていることを確認する。"""
        assert len(BLOG_TYPES) == 5

    def test_get_blog_type_tech(self) -> None:
        """techブログ種別が取得できることを確認する。"""
        bt = get_blog_type("tech")

        assert bt is not None
        assert bt.id == "tech"
        assert bt.name == "技術ブログ"
        assert len(bt.hearing_items) > 0

    def test_get_blog_type_diary(self) -> None:
        """diaryブログ種別が取得できることを確認する。"""
        bt = get_blog_type("diary")

        assert bt is not None
        assert bt.id == "diary"

    def test_get_blog_type_review(self) -> None:
        """reviewブログ種別が取得できることを確認する。"""
        bt = get_blog_type("review")

        assert bt is not None
        assert bt.id == "review"

    def test_get_blog_type_news(self) -> None:
        """newsブログ種別が取得できることを確認する。"""
        bt = get_blog_type("news")

        assert bt is not None
        assert bt.id == "news"

    def test_get_blog_type_howto(self) -> None:
        """howtoブログ種別が取得できることを確認する。"""
        bt = get_blog_type("howto")

        assert bt is not None
        assert bt.id == "howto"

    def test_get_blog_type_nonexistent(self) -> None:
        """存在しないIDでNoneが返されることを確認する。"""
        assert get_blog_type("nonexistent") is None

    def test_get_all_blog_types(self) -> None:
        """全てのブログ種別が取得できることを確認する。"""
        all_types = get_all_blog_types()

        assert len(all_types) == 5
        assert all(bt.id for bt in all_types)

    def test_all_types_have_hearing_items(self) -> None:
        """全ての種別にヒアリング項目があることを確認する。"""
        for bt in get_all_blog_types():
            assert len(bt.hearing_items) > 0, f"{bt.id}にヒアリング項目がありません"

    def test_all_types_have_system_prompt(self) -> None:
        """全ての種別にシステムプロンプトがあることを確認する。"""
        for bt in get_all_blog_types():
            assert bt.system_prompt, f"{bt.id}にシステムプロンプトがありません"

    def test_hearing_items_have_required_fields(self) -> None:
        """ヒアリング項目に必須フィールドがあることを確認する。"""
        for bt in get_all_blog_types():
            for item in bt.hearing_items:
                assert item.key, f"{bt.id}のヒアリング項目にkeyがありません"
                assert item.question, f"{bt.id}のヒアリング項目にquestionがありません"
