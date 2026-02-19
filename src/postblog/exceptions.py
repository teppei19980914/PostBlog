"""PostBlogアプリの例外階層。"""


class PostBlogError(Exception):
    """PostBlogアプリの基底例外。"""


class AuthenticationError(PostBlogError):
    """認証エラー。"""


class NetworkError(PostBlogError):
    """ネットワークエラー。"""


class PublishError(PostBlogError):
    """投稿エラー。"""


class LLMError(PostBlogError):
    """LLM API エラー。"""


class StorageError(PostBlogError):
    """データ保存エラー。"""


class ValidationError(PostBlogError):
    """入力検証エラー。"""
