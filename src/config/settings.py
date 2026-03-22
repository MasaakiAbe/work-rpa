"""アプリケーション設定 - 環境変数 / Streamlit secrets から読み込み"""

import os

from pydantic_settings import BaseSettings
from pydantic import Field


def _loadStreamlitSecrets():
  """Streamlit Cloudのsecretsを環境変数に注入する"""
  try:
    import streamlit as st
    for key, value in st.secrets.items():
      if isinstance(value, str):
        os.environ.setdefault(key, value)
  except Exception:
    pass

# Streamlit secrets → 環境変数にマッピング
_loadStreamlitSecrets()


class Settings(BaseSettings):
  # OCRエンジン設定
  ocrPrimaryEngine: str = Field(default='gemini', alias='OCR_PRIMARY_ENGINE')
  ocrFallbackEngine: str = Field(default='tesseract', alias='OCR_FALLBACK_ENGINE')

  # Azure AI Document Intelligence
  azureEndpoint: str = Field(default='', alias='AZURE_ENDPOINT')
  azureApiKey: str = Field(default='', alias='AZURE_API_KEY')

  # 画像前処理設定
  imageMaxSize: int = Field(default=4096, alias='IMAGE_MAX_SIZE')
  denoiseStrength: int = Field(default=10, alias='DENOISE_STRENGTH')
  binarizeBlockSize: int = Field(default=11, alias='BINARIZE_BLOCK_SIZE')

  # Tesseractパス
  tesseractCmd: str = Field(default='tesseract', alias='TESSERACT_CMD')
  tessdataPrefix: str = Field(default='', alias='TESSDATA_PREFIX')

  # Gemini API
  geminiApiKey: str = Field(default='', alias='GEMINI_API_KEY')

  # アプリ設定
  logLevel: str = Field(default='INFO', alias='LOG_LEVEL')

  model_config = {
    'env_file': '.env',
    'env_file_encoding': 'utf-8',
    'populate_by_name': True,
  }


def getSettings() -> Settings:
  return Settings()
