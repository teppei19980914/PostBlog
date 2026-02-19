"""HearingMessage・HearingResultモデルのテスト。"""

from datetime import datetime

from postblog.models.hearing import HearingMessage, HearingResult


class TestHearingMessage:
    """HearingMessageのテスト。"""

    def test_create_message(self) -> None:
        """HearingMessageが正しく作成されることを確認する。"""
        msg = HearingMessage(role="user", content="こんにちは")

        assert msg.role == "user"
        assert msg.content == "こんにちは"
        assert isinstance(msg.timestamp, datetime)

    def test_create_message_with_timestamp(self) -> None:
        """タイムスタンプ付きでHearingMessageが作成されることを確認する。"""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        msg = HearingMessage(role="assistant", content="回答です", timestamp=ts)

        assert msg.timestamp == ts

    def test_message_role_user(self) -> None:
        """ユーザーロールのメッセージを確認する。"""
        msg = HearingMessage(role="user", content="テスト")
        assert msg.role == "user"

    def test_message_role_assistant(self) -> None:
        """アシスタントロールのメッセージを確認する。"""
        msg = HearingMessage(role="assistant", content="テスト")
        assert msg.role == "assistant"

    def test_message_is_mutable(self) -> None:
        """HearingMessageがミュータブルであることを確認する。"""
        msg = HearingMessage(role="user", content="元のメッセージ")
        msg.content = "変更後のメッセージ"
        assert msg.content == "変更後のメッセージ"


class TestHearingResult:
    """HearingResultのテスト。"""

    def test_create_result_minimal(self) -> None:
        """最小限のパラメータでHearingResultが作成されることを確認する。"""
        result = HearingResult(blog_type_id="tech")

        assert result.blog_type_id == "tech"
        assert result.messages == []
        assert result.answers == {}
        assert result.seo_keywords == ""
        assert result.seo_target_audience == ""
        assert result.seo_search_intent == ""
        assert result.summary == ""
        assert result.completed is False

    def test_create_result_full(self) -> None:
        """全パラメータ指定でHearingResultが作成されることを確認する。"""
        messages = [
            HearingMessage(role="assistant", content="テーマは？"),
            HearingMessage(role="user", content="Pythonです"),
        ]
        result = HearingResult(
            blog_type_id="tech",
            messages=messages,
            answers={"topic": "Python"},
            seo_keywords="Python 入門",
            seo_target_audience="プログラミング初心者",
            seo_search_intent="Python の基本を学びたい",
            summary="Pythonの入門記事",
            completed=True,
        )

        assert result.blog_type_id == "tech"
        assert len(result.messages) == 2
        assert result.answers["topic"] == "Python"
        assert result.seo_keywords == "Python 入門"
        assert result.seo_target_audience == "プログラミング初心者"
        assert result.seo_search_intent == "Python の基本を学びたい"
        assert result.summary == "Pythonの入門記事"
        assert result.completed is True

    def test_add_message(self) -> None:
        """メッセージを追加できることを確認する。"""
        result = HearingResult(blog_type_id="tech")
        result.messages.append(HearingMessage(role="user", content="テスト"))

        assert len(result.messages) == 1
        assert result.messages[0].content == "テスト"

    def test_add_answer(self) -> None:
        """回答を追加できることを確認する。"""
        result = HearingResult(blog_type_id="tech")
        result.answers["topic"] = "Python"
        result.answers["level"] = "初級"

        assert len(result.answers) == 2
        assert result.answers["topic"] == "Python"

    def test_mark_completed(self) -> None:
        """完了フラグを変更できることを確認する。"""
        result = HearingResult(blog_type_id="tech")
        assert result.completed is False

        result.completed = True
        assert result.completed is True

    def test_default_lists_are_independent(self) -> None:
        """デフォルトのリスト・辞書が独立していることを確認する。"""
        result1 = HearingResult(blog_type_id="tech")
        result2 = HearingResult(blog_type_id="diary")

        result1.messages.append(HearingMessage(role="user", content="msg1"))
        result1.answers["key"] = "value"

        assert len(result2.messages) == 0
        assert len(result2.answers) == 0
