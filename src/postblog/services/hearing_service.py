"""ヒアリングサービス。

LLMを使用したインタラクティブなヒアリングフローを管理する。
"""

import json
import logging

from postblog.infrastructure.llm.base import LLMClient
from postblog.models.blog_type import BlogType
from postblog.models.hearing import HearingMessage, HearingResult
from postblog.templates.prompts import HEARING_SUMMARY_PROMPT, HEARING_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class HearingService:
    """ヒアリングフローを管理するサービス。

    Args:
        llm_client: LLMクライアント。
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def start_hearing(self, blog_type: BlogType) -> HearingResult:
        """ヒアリングを開始する。

        Args:
            blog_type: ブログ種別。

        Returns:
            初期化されたHearingResult。
        """
        result = HearingResult(blog_type_id=blog_type.id)
        logger.info("ヒアリングを開始しました: blog_type=%s", blog_type.id)
        return result

    async def send_message(
        self, hearing_result: HearingResult, user_message: str, blog_type: BlogType
    ) -> str:
        """ユーザーメッセージを送信し、AIの応答を取得する。

        Args:
            hearing_result: 現在のヒアリング結果。
            user_message: ユーザーのメッセージ。
            blog_type: ブログ種別。

        Returns:
            AIの応答テキスト。
        """
        hearing_result.messages.append(
            HearingMessage(role="user", content=user_message)
        )

        # ヒアリング項目の文字列生成
        items_text = "\n".join(
            f"- {item.question}（{'必須' if item.required else '任意'}）"
            for item in blog_type.hearing_items
        )

        system_prompt = HEARING_SYSTEM_PROMPT.format(
            hearing_policy=blog_type.hearing_policy,
            hearing_items=items_text,
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {"role": msg.role, "content": msg.content}
            for msg in hearing_result.messages
        )

        response = await self._llm.chat(messages)
        hearing_result.messages.append(
            HearingMessage(role="assistant", content=response)
        )

        logger.debug("ヒアリングメッセージを送受信しました")
        return response

    async def generate_summary(self, hearing_result: HearingResult) -> HearingResult:
        """ヒアリング結果のサマリーを生成する。

        Args:
            hearing_result: ヒアリング結果。

        Returns:
            サマリーが設定されたHearingResult。
        """
        conversation = "\n".join(
            f"{msg.role}: {msg.content}" for msg in hearing_result.messages
        )

        prompt = HEARING_SUMMARY_PROMPT.format(conversation=conversation)
        messages = [{"role": "user", "content": prompt}]

        response = await self._llm.chat(messages, temperature=0.3)

        try:
            data = json.loads(response)
            hearing_result.summary = data.get("summary", "")
            hearing_result.answers.update(data.get("answers", {}))
            hearing_result.seo_keywords = data.get("seo_keywords", "")
            hearing_result.seo_target_audience = data.get("seo_target_audience", "")
            hearing_result.seo_search_intent = data.get("seo_search_intent", "")
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "サマリーのパースに失敗しました。応答全文をサマリーとして使用します。"
            )
            hearing_result.summary = response

        hearing_result.completed = True
        logger.info("ヒアリングサマリーを生成しました")
        return hearing_result
