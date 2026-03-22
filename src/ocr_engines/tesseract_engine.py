"""Tesseract OCRエンジン - ローカル実行"""

from __future__ import annotations

import time
import logging

import numpy as np
import pytesseract
from PIL import Image

from src.ocr_engines.base_engine import BaseOcrEngine
from src.models.data_models import OcrResult

logger = logging.getLogger(__name__)


class TesseractEngine(BaseOcrEngine):
  """Tesseract OCRエンジン"""

  def __init__(self, tesseractCmd: str | None = None, tessdataPrefix: str | None = None):
    if tesseractCmd:
      pytesseract.pytesseract.tesseract_cmd = tesseractCmd
    self.tessdataPrefix = tessdataPrefix

  @property
  def engineName(self) -> str:
    return 'tesseract'

  def recognize(self, image: np.ndarray, lang: str = 'jpn', promptMode: str = 'テキスト読み取り') -> OcrResult:
    """Tesseractで文字認識を実行"""
    startTime = time.time()

    pilImage = Image.fromarray(image)

    # TESSDATA_PREFIX設定（環境変数で渡す）
    import os
    if self.tessdataPrefix:
      os.environ['TESSDATA_PREFIX'] = self.tessdataPrefix

    # OCR実行（テキスト + 信頼度）
    data = pytesseract.image_to_data(
      pilImage,
      lang=lang,
      output_type=pytesseract.Output.DICT,
    )

    # テキスト結合と平均信頼度を算出
    texts: list[str] = []
    confidences: list[float] = []

    for i, text in enumerate(data['text']):
      text = text.strip()
      if text:
        texts.append(text)
        conf = float(data['conf'][i])
        if conf > 0:
          confidences.append(conf)

    resultText = ' '.join(texts)
    avgConfidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

    elapsedMs = int((time.time() - startTime) * 1000)

    return OcrResult(
      text=resultText,
      confidence=avgConfidence,
      engineUsed=self.engineName,
      rawResponse={'data': data},
      processingTimeMs=elapsedMs,
    )

  def isAvailable(self) -> bool:
    """Tesseractがインストールされているか確認"""
    try:
      pytesseract.get_tesseract_version()
      return True
    except Exception:
      logger.warning('Tesseractが見つかりません')
      return False
