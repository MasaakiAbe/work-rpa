"""Gemini Vision OCRエンジン - 手書き日本語に最適"""

from __future__ import annotations

import io
import time
import logging

import numpy as np
from PIL import Image

from src.ocr_engines.base_engine import BaseOcrEngine
from src.models.data_models import OcrResult

logger = logging.getLogger(__name__)

# 試行するモデル（優先順。429/404なら即次へ）
MODELS = [
  'gemini-2.5-flash',
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite',
]

PROMPTS = {
  'テキスト読み取り': """この画像に含まれるテキストを正確に読み取ってください。

ルール:
- 画像内の文字をそのまま忠実にテキスト化する
- 手書き文字も含めて読み取る
- 余計な説明や補足は一切不要
- 読み取ったテキストのみを出力する
- 文字が読み取れない場合は「[判読不能]」と記載する
""",

  '項目抽出': """この画像はFAXで届いた書類です。以下の項目を抽出してください。

抽出項目:
- 宛名
- 会社名
- 担当者名
- 電話番号
- FAX番号
- 製品名/品番
- 数量
- 納期
- 備考

出力ルール:
- 上記のフォーマットで出力する（項目名: 値）
- 該当する情報がない項目は「-」と記載する
- 手書き文字も含めて読み取る
- 複数の製品がある場合は番号を振って列挙する
- 余計な説明は不要
""",
}


class GeminiEngine(BaseOcrEngine):
  """Gemini Vision OCRエンジン"""

  def __init__(self, apiKey: str):
    self.apiKey = apiKey
    self._client = None

  @property
  def engineName(self) -> str:
    return 'gemini'

  def _getClient(self):
    """遅延初期化でクライアントを取得"""
    if self._client is None:
      from google import genai
      self._client = genai.Client(api_key=self.apiKey)
    return self._client

  def recognize(self, image: np.ndarray, lang: str = 'jpn', promptMode: str = 'テキスト読み取り') -> OcrResult:
    """Gemini Visionで文字認識を実行（レート制限時は自動リトライ）"""
    startTime = time.time()

    # numpy → PIL → バイト列
    pilImage = Image.fromarray(image)
    if pilImage.mode != 'RGB':
      pilImage = pilImage.convert('RGB')
    buffer = io.BytesIO()
    pilImage.save(buffer, format='PNG')
    imageBytes = buffer.getvalue()

    prompt = PROMPTS.get(promptMode, PROMPTS['テキスト読み取り'])

    client = self._getClient()
    from google.genai import types

    contents = [
      types.Content(
        parts=[
          types.Part.from_bytes(data=imageBytes, mime_type='image/png'),
          types.Part.from_text(text=prompt),
        ],
      ),
    ]

    # モデルを順に試行し、429なら即次のモデルへ
    lastError = None
    for model in MODELS:
      try:
        logger.info(f'Gemini OCR: model={model}')
        response = client.models.generate_content(
          model=model,
          contents=contents,
        )
        resultText = response.text.strip() if response.text else ''
        elapsedMs = int((time.time() - startTime) * 1000)

        return OcrResult(
          text=resultText,
          confidence=0.95 if resultText else 0.0,
          engineUsed=f'gemini ({model})',
          rawResponse=None,
          processingTimeMs=elapsedMs,
        )

      except Exception as e:
        lastError = e
        errorStr = str(e)
        if '429' in errorStr or 'RESOURCE_EXHAUSTED' in errorStr:
          logger.warning(f'{model} レート制限。次のモデルへ切替。')
          continue
        elif '404' in errorStr or 'NOT_FOUND' in errorStr:
          logger.warning(f'{model} が利用不可。次のモデルへ切替。')
          continue
        else:
          raise

    raise RuntimeError(f'全モデルでレート制限です。しばらく待ってから再試行してください: {lastError}')

  def isAvailable(self) -> bool:
    """APIキーが設定されているか確認"""
    return bool(self.apiKey)
