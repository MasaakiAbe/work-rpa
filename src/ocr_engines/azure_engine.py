"""Azure AI Document Intelligence OCRエンジン"""

from __future__ import annotations

import io
import time
import logging

import numpy as np
from PIL import Image

from src.ocr_engines.base_engine import BaseOcrEngine
from src.models.data_models import OcrResult

logger = logging.getLogger(__name__)


class AzureEngine(BaseOcrEngine):
  """Azure AI Document Intelligence OCRエンジン"""

  def __init__(self, endpoint: str, apiKey: str):
    self.endpoint = endpoint
    self.apiKey = apiKey
    self._client = None

  @property
  def engineName(self) -> str:
    return 'azure'

  def _getClient(self):
    """遅延初期化でクライアントを取得"""
    if self._client is None:
      try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        self._client = DocumentIntelligenceClient(
          endpoint=self.endpoint,
          credential=AzureKeyCredential(self.apiKey),
        )
      except ImportError:
        raise RuntimeError(
          'azure-ai-documentintelligence パッケージが必要です。'
          'pip install azure-ai-documentintelligence でインストールしてください。'
        )
    return self._client

  def recognize(self, image: np.ndarray, lang: str = 'jpn', promptMode: str = 'テキスト読み取り') -> OcrResult:
    """Azure Document Intelligenceで文字認識を実行"""
    startTime = time.time()

    # numpy → バイトストリーム変換
    pilImage = Image.fromarray(image)
    buffer = io.BytesIO()
    pilImage.save(buffer, format='PNG')
    buffer.seek(0)

    client = self._getClient()

    # Read APIを使用（手書き対応）
    poller = client.begin_analyze_document(
      'prebuilt-read',
      analyze_request=buffer,
      content_type='application/octet-stream',
    )
    result = poller.result()

    # テキスト抽出
    texts: list[str] = []
    confidences: list[float] = []

    if result.content:
      texts.append(result.content)

    for page in (result.pages or []):
      for word in (page.words or []):
        if word.confidence is not None:
          confidences.append(word.confidence)

    resultText = result.content or ''
    avgConfidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    elapsedMs = int((time.time() - startTime) * 1000)

    return OcrResult(
      text=resultText,
      confidence=avgConfidence,
      engineUsed=self.engineName,
      rawResponse={'pageCount': len(result.pages or [])},
      processingTimeMs=elapsedMs,
    )

  def isAvailable(self) -> bool:
    """Azure認証情報が設定されているか確認"""
    return bool(self.endpoint and self.apiKey)
