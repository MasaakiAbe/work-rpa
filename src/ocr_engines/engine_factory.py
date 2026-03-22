"""OCRエンジンファクトリー - 設定に基づいてエンジンを生成"""

from __future__ import annotations

import logging

from src.config.settings import Settings, getSettings
from src.ocr_engines.base_engine import BaseOcrEngine
from src.ocr_engines.tesseract_engine import TesseractEngine
from src.ocr_engines.azure_engine import AzureEngine
from src.ocr_engines.gemini_engine import GeminiEngine

logger = logging.getLogger(__name__)


def createOcrEngine(settings: Settings | None = None) -> BaseOcrEngine:
  """
  設定に基づいてOCRエンジンを生成する。
  プライマリエンジンが利用不可の場合、フォールバックに切り替える。
  """
  settings = settings or getSettings()

  engines = {
    'gemini': lambda: GeminiEngine(apiKey=settings.geminiApiKey),
    'azure': lambda: AzureEngine(
      endpoint=settings.azureEndpoint,
      apiKey=settings.azureApiKey,
    ),
    'tesseract': lambda: TesseractEngine(
      tesseractCmd=settings.tesseractCmd,
      tessdataPrefix=settings.tessdataPrefix,
    ),
  }

  # プライマリエンジンを試行
  primaryName = settings.ocrPrimaryEngine.lower()
  if primaryName in engines:
    engine = engines[primaryName]()
    if engine.isAvailable():
      logger.info(f'OCRエンジン: {engine.engineName} (primary)')
      return engine
    logger.warning(f'{primaryName} は利用不可。フォールバックに切り替えます。')

  # フォールバックエンジン
  fallbackName = settings.ocrFallbackEngine.lower()
  if fallbackName in engines:
    engine = engines[fallbackName]()
    if engine.isAvailable():
      logger.info(f'OCRエンジン: {engine.engineName} (fallback)')
      return engine

  raise RuntimeError('利用可能なOCRエンジンがありません。')
