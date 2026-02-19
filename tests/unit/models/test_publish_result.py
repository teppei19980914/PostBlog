"""PublishRequest・PublishResultモデルのテスト。"""

from postblog.models.publish_result import PublishRequest, PublishResult


class TestPublishRequest:
    """PublishRequestのテスト。"""

    def test_create_request_required_fields(self) -> None:
        """必須フィールドのみでPublishRequestが作成されることを確認する。"""
        request = PublishRequest(title="テスト記事", body="# テスト\n\n本文です。")

        assert request.title == "テスト記事"
        assert request.body == "# テスト\n\n本文です。"
        assert request.tags == []
        assert request.status == "publish"
        assert request.blog_type_id == ""

    def test_create_request_full(self) -> None:
        """全フィールド指定でPublishRequestが作成されることを確認する。"""
        request = PublishRequest(
            title="Python入門",
            body="# Python入門\n\n本文です。",
            tags=["Python", "入門"],
            status="draft",
            blog_type_id="tech",
        )

        assert request.title == "Python入門"
        assert request.tags == ["Python", "入門"]
        assert request.status == "draft"
        assert request.blog_type_id == "tech"

    def test_request_default_status_is_publish(self) -> None:
        """デフォルトのステータスがpublishであることを確認する。"""
        request = PublishRequest(title="t", body="b")
        assert request.status == "publish"

    def test_request_tags_are_independent(self) -> None:
        """デフォルトのタグリストが独立していることを確認する。"""
        req1 = PublishRequest(title="t1", body="b1")
        req2 = PublishRequest(title="t2", body="b2")

        req1.tags.append("tag")

        assert len(req2.tags) == 0


class TestPublishResult:
    """PublishResultのテスト。"""

    def test_create_result_success(self) -> None:
        """成功のPublishResultが作成されることを確認する。"""
        result = PublishResult(
            success=True,
            service_name="qiita",
            article_url="https://qiita.com/user/items/abc123",
        )

        assert result.success is True
        assert result.service_name == "qiita"
        assert result.article_url == "https://qiita.com/user/items/abc123"
        assert result.error_message is None

    def test_create_result_failure(self) -> None:
        """失敗のPublishResultが作成されることを確認する。"""
        result = PublishResult(
            success=False,
            service_name="wordpress",
            error_message="認証エラー: APIキーが無効です。",
        )

        assert result.success is False
        assert result.service_name == "wordpress"
        assert result.article_url is None
        assert result.error_message == "認証エラー: APIキーが無効です。"

    def test_create_result_minimal(self) -> None:
        """最小限のパラメータでPublishResultが作成されることを確認する。"""
        result = PublishResult(success=True, service_name="zenn")

        assert result.success is True
        assert result.service_name == "zenn"
        assert result.article_url is None
        assert result.error_message is None
