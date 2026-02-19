"""SEO関連モデルのテスト。"""

from postblog.models.seo import (
    SeoAdvice,
    SeoAdviceItem,
    SeoAnalysisResult,
    SeoCheckItem,
)


class TestSeoCheckItem:
    """SeoCheckItemのテスト。"""

    def test_create_check_item_pass(self) -> None:
        """合格のSeoCheckItemが作成されることを確認する。"""
        item = SeoCheckItem(
            category="タイトル",
            name="キーワード含有",
            status="pass",
            score=15,
            max_score=15,
            message="タイトルにキーワードが含まれています。",
        )

        assert item.category == "タイトル"
        assert item.name == "キーワード含有"
        assert item.status == "pass"
        assert item.score == 15
        assert item.max_score == 15
        assert item.message == "タイトルにキーワードが含まれています。"
        assert item.suggestion is None

    def test_create_check_item_fail_with_suggestion(self) -> None:
        """不合格のSeoCheckItemに改善提案が付与されることを確認する。"""
        item = SeoCheckItem(
            category="本文",
            name="キーワード密度",
            status="fail",
            score=0,
            max_score=10,
            message="キーワード密度が低すぎます（0.5%）。",
            suggestion="キーワードを本文中に自然に追加してください。",
        )

        assert item.status == "fail"
        assert item.score == 0
        assert item.suggestion == "キーワードを本文中に自然に追加してください。"

    def test_create_check_item_warn(self) -> None:
        """警告のSeoCheckItemが作成されることを確認する。"""
        item = SeoCheckItem(
            category="見出し",
            name="見出し階層",
            status="warn",
            score=5,
            max_score=10,
            message="H3が使用されていません。",
            suggestion="小見出しの追加を検討してください。",
        )

        assert item.status == "warn"
        assert item.score == 5


class TestSeoAnalysisResult:
    """SeoAnalysisResultのテスト。"""

    def test_create_result_minimal(self) -> None:
        """最小限のパラメータで作成されることを確認する。"""
        result = SeoAnalysisResult(score=75)

        assert result.score == 75
        assert result.items == []
        assert result.suggestions == []

    def test_create_result_with_items(self) -> None:
        """チェック項目付きで作成されることを確認する。"""
        items = [
            SeoCheckItem(
                category="タイトル",
                name="キーワード含有",
                status="pass",
                score=15,
                max_score=15,
                message="OK",
            ),
            SeoCheckItem(
                category="本文",
                name="文字数",
                status="fail",
                score=0,
                max_score=10,
                message="文字数不足",
                suggestion="1500文字以上にしてください。",
            ),
        ]
        result = SeoAnalysisResult(
            score=60,
            items=items,
            suggestions=["本文の文字数を増やしてください。"],
        )

        assert result.score == 60
        assert len(result.items) == 2
        assert len(result.suggestions) == 1

    def test_result_items_are_independent(self) -> None:
        """デフォルトのリストが独立していることを確認する。"""
        result1 = SeoAnalysisResult(score=80)
        result2 = SeoAnalysisResult(score=90)

        result1.suggestions.append("提案1")

        assert len(result2.suggestions) == 0


class TestSeoAdviceItem:
    """SeoAdviceItemのテスト。"""

    def test_create_advice_item(self) -> None:
        """SeoAdviceItemが正しく作成されることを確認する。"""
        item = SeoAdviceItem(
            category="タイトル",
            point="主要キーワードを先頭に配置",
            reason="検索結果での視認性が向上します。",
            edit_tip="タイトル変更時もキーワードを先頭付近に保持してください。",
        )

        assert item.category == "タイトル"
        assert item.point == "主要キーワードを先頭に配置"
        assert item.reason == "検索結果での視認性が向上します。"
        assert (
            item.edit_tip == "タイトル変更時もキーワードを先頭付近に保持してください。"
        )


class TestSeoAdvice:
    """SeoAdviceのテスト。"""

    def test_create_advice_minimal(self) -> None:
        """最小限のパラメータで作成されることを確認する。"""
        advice = SeoAdvice()

        assert advice.items == []
        assert advice.summary == ""
        assert advice.target_keyword == ""
        assert advice.generated_at == ""

    def test_create_advice_full(self) -> None:
        """全パラメータ指定で作成されることを確認する。"""
        items = [
            SeoAdviceItem(
                category="タイトル",
                point="キーワード配置",
                reason="視認性向上",
                edit_tip="先頭に保持",
            ),
            SeoAdviceItem(
                category="見出し構造",
                point="H2にキーワード含有",
                reason="構造化データ対応",
                edit_tip="見出し変更時も含有を維持",
            ),
        ]
        advice = SeoAdvice(
            items=items,
            summary="SEO対策済みの記事構成です。",
            target_keyword="Python 入門",
            generated_at="2024-06-01T12:00:00+09:00",
        )

        assert len(advice.items) == 2
        assert advice.summary == "SEO対策済みの記事構成です。"
        assert advice.target_keyword == "Python 入門"
        assert advice.generated_at == "2024-06-01T12:00:00+09:00"

    def test_advice_items_are_independent(self) -> None:
        """デフォルトのリストが独立していることを確認する。"""
        advice1 = SeoAdvice()
        advice2 = SeoAdvice()

        advice1.items.append(
            SeoAdviceItem(category="c", point="p", reason="r", edit_tip="e")
        )

        assert len(advice2.items) == 0
