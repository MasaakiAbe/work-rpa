"""OCRエンジン基底クラス - Strategy パターン"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.models.data_models import OcrResult


class BaseOcrEngine(ABC):
  """OCRエンジンのインターフェース"""

  @property
  @abstractmethod
  def engineName(self) -> str:
    """エンジン名"""
    ...

  @abstractmethod
  def recognize(self, image: np.ndarray, lang: str = 'jpn', promptMode: str = 'テキスト読み取り') -> OcrResult:
    """
    画像からテキストを認識する

    Args:
      image: 前処理済み画像 (numpy配列)
      lang: 認識言語 (デフォルト: 日本語)

    Returns:
      OcrResult: 認識結果
    """
    ...

  def isAvailable(self) -> bool:
    """このエンジンが利用可能か確認"""
    return True
