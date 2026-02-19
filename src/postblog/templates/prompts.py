"""LLMプロンプトテンプレート。"""

# ヒアリング用プロンプト
HEARING_SYSTEM_PROMPT = """あなたはブログ記事作成のアシスタントです。
ユーザーから記事のテーマや内容についてヒアリングを行います。

以下のルールに従ってください:
1. 一度に1つの質問のみを行う
2. ユーザーの回答に基づいて適切なフォローアップ質問をする
3. 必要な情報が集まったら、ヒアリング完了を提案する
4. 丁寧で親しみやすいトーンで会話する

{hearing_policy}

ヒアリング項目:
{hearing_items}

SEO関連情報もヒアリング中に確認してください:
- ターゲットキーワード
- 想定読者
- 検索意図
"""

# ヒアリングサマリー生成プロンプト
HEARING_SUMMARY_PROMPT = """以下のヒアリング会話から、記事作成に必要な情報をJSON形式でサマリーしてください。

会話履歴:
{conversation}

以下のJSON形式で出力してください:
{{
    "summary": "ヒアリング内容の要約（200文字以内）",
    "answers": {{
        "キー": "回答内容"
    }},
    "seo_keywords": "ターゲットキーワード",
    "seo_target_audience": "想定読者",
    "seo_search_intent": "検索意図"
}}
"""

# 記事生成プロンプト
ARTICLE_GENERATION_PROMPT = """以下のヒアリング結果に基づいて、SEO対策済みのブログ記事を生成してください。

## ヒアリング結果
{hearing_summary}

## SEO情報
- ターゲットキーワード: {seo_keywords}
- 想定読者: {seo_target_audience}
- 検索意図: {seo_search_intent}

## 記事の要件
1. Markdown形式で出力する
2. タイトルにキーワードを含める
3. 見出し構造（H2, H3）を適切に使用する
4. キーワード密度2-5%を目安にする
5. メタディスクリプション（120-160文字）を記事冒頭に含める
6. 1500文字以上の本文を作成する

## 出力形式
記事本文（Markdown）の後に、SEO対策ポイント解説をJSON形式で出力してください。
区切りマーカーで分離してください。

[記事本文（Markdown）]
---SEO_ADVICE_START---
{{
    "items": [
        {{
            "category": "カテゴリ名",
            "point": "施策内容",
            "reason": "理由・効果",
            "edit_tip": "編集時の注意点"
        }}
    ],
    "summary": "SEO対策の全体サマリー",
    "target_keyword": "{seo_keywords}",
    "generated_at": "生成日時"
}}
---SEO_ADVICE_END---
"""

# SEO対策ポイント解説マーカー
SEO_ADVICE_START_MARKER = "---SEO_ADVICE_START---"
SEO_ADVICE_END_MARKER = "---SEO_ADVICE_END---"
