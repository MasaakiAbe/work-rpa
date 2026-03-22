"""OCR Agent - テキスト認識を担当するエージェント"""

from __future__ import annotations

import logging

from src.agents.base_agent import BaseAgent
from src.models.data_models import ProcessedImage, OcrResult
from src.ocr_engines.base_engine import BaseOcrEngine
from src.ocr_engines.engine_factory import createOcrEngine
from src.config.settings import Settings

logger = logging.getLogger(__name__)

# 信頼度がこの値以下の場合に警告を出す
LOW_CONFIDENCE_THRESHOLD = 0.5


class OcrAgent(BaseAgent):
  """OCR認識を担当するエージェント"""

  def __init__(self, engine: BaseOcrEngine | None = None, settings: Settings | None = None):
    super().__init__('OcrAgent')
    self.engine = engine or createOcrEngine(settings)

  def process(self, inputData: ProcessedImage | dict) -> OcrResult:
    """
    前処理済み画像からテキストを認識する

    Args:
      inputData: ProcessedImage または {'processedImage': ProcessedImage, 'promptMode': str}
    """
    if isinstance(inputData, dict):
      image = inputData['processedImage'].image
      promptMode = inputData.get('promptMode', 'テキスト読み取り')
    else:
      image = inputData.image
      promptMode = 'テキスト読み取り'

    result = self.engine.recognize(image, lang='jpn', promptMode=promptMode)

    # 信頼度が低い場合に警告
    if result.confidence < LOW_CONFIDENCE_THRESHOLD:
      self.logger.warning(
        f'OCR信頼度が低い: {result.confidence:.2f} '
        f'(しきい値: {LOW_CONFIDENCE_THRESHOLD})'
      )

    return result
