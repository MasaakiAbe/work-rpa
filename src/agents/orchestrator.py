"""Orchestrator - エージェントチームを統括するコーディネーター"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from src.agents.base_agent import BaseAgent
from src.agents.image_processor import ImageProcessorAgent
from src.agents.ocr_agent import OcrAgent
from src.models.data_models import RegionSelection, ProcessedImage, PipelineResult
from src.config.settings import Settings, getSettings

logger = logging.getLogger(__name__)


class Orchestrator(BaseAgent):
  """
  Agent Teamのオーケストレーター。
  エンジン種別に応じてパイプラインを切り替える。
  - Vision LLM (gemini): 元画像をクロップしてそのまま渡す（前処理不要）
  - 従来OCR (tesseract/azure): 前処理パイプラインを通す
  """

  def __init__(self, settings: Settings | None = None):
    super().__init__('Orchestrator')
    self.settings = settings or getSettings()
    self.imageProcessor = ImageProcessorAgent(self.settings)
    self.ocrAgent = OcrAgent(settings=self.settings)

  def process(self, inputData: dict) -> PipelineResult:
    """OCRパイプラインを実行する"""
    try:
      self._validate(inputData)

      image = inputData['image']
      region: RegionSelection = inputData['region']
      promptMode = inputData.get('promptMode', 'テキスト読み取り')
      engineName = self.ocrAgent.engine.engineName

      if engineName == 'gemini':
        return self._runVisionLlm(image, region, promptMode)
      else:
        return self._runTraditionalOcr(inputData)

    except Exception as e:
      self.logger.error(f'パイプラインエラー: {e}')
      return PipelineResult(
        success=False,
        errorMessage=str(e),
      )

  def _runVisionLlm(self, image, region: RegionSelection, promptMode: str = 'テキスト読み取り') -> PipelineResult:
    """Vision LLM用: クロップのみでOCR実行"""
    self.logger.info('Step 1: 画像クロップ（Vision LLMモード）')

    if isinstance(image, Image.Image):
      image = np.array(image)

    originalSize = (image.shape[1], image.shape[0])

    # クロップだけ実行
    x1, y1, x2, y2 = region.toTuple()
    h, w = image.shape[:2]
    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    cropped = image[y1:y2, x1:x2]

    processedImage = ProcessedImage(
      image=cropped,
      originalSize=originalSize,
      croppedRegion=region,
      preprocessSteps=['crop_only'],
    )

    self.logger.info(f'Step 2: Gemini Vision OCR実行 (mode={promptMode})')
    ocrResult = self.ocrAgent.execute({
      'processedImage': processedImage,
      'promptMode': promptMode,
    })

    return PipelineResult(
      ocrResult=ocrResult,
      processedImage=processedImage,
      success=True,
    )

  def _runTraditionalOcr(self, inputData: dict) -> PipelineResult:
    """従来OCR用: 前処理 → OCR"""
    self.logger.info('Step 1: 画像前処理を実行')
    processedImage = self.imageProcessor.execute(inputData)

    self.logger.info('Step 2: OCR認識を実行')
    ocrResult = self.ocrAgent.execute(processedImage)

    return PipelineResult(
      ocrResult=ocrResult,
      processedImage=processedImage,
      success=True,
    )

  def _validate(self, inputData: dict) -> None:
    """入力データのバリデーション"""
    if 'image' not in inputData:
      raise ValueError('image が指定されていません')
    if 'region' not in inputData:
      raise ValueError('region が指定されていません')

    image = inputData['image']
    if not isinstance(image, (np.ndarray, Image.Image)):
      raise ValueError('image は PIL.Image または numpy.ndarray である必要があります')

    region = inputData['region']
    if not isinstance(region, RegionSelection):
      raise ValueError('region は RegionSelection である必要があります')
    if region.width <= 0 or region.height <= 0:
      raise ValueError('region のサイズが不正です')
