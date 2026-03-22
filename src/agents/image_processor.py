"""Image Processor Agent - 画像前処理パイプライン"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.agents.base_agent import BaseAgent
from src.config.settings import Settings, getSettings
from src.models.data_models import RegionSelection, ProcessedImage


class ImageProcessorAgent(BaseAgent):
  """FAX画像の前処理を担当するエージェント"""

  def __init__(self, settings: Settings | None = None, mode: str = 'auto'):
    super().__init__('ImageProcessor')
    self.settings = settings or getSettings()
    self.mode = mode  # 'auto', 'handwritten', 'printed'

  def process(self, inputData: dict) -> ProcessedImage:
    """
    画像前処理パイプラインを実行する

    Args:
      inputData: {
        'image': PIL.Image または numpy.ndarray,
        'region': RegionSelection,
        'mode': 'auto' | 'handwritten' | 'printed' (optional)
      }
    """
    image = inputData['image']
    region: RegionSelection = inputData['region']
    mode = inputData.get('mode', self.mode)

    # PIL → numpy変換
    if isinstance(image, Image.Image):
      image = np.array(image)

    originalSize = (image.shape[1], image.shape[0])
    steps: list[str] = []

    # 1. 指定領域をクロップ
    cropped = self._crop(image, region)
    steps.append('crop')

    # 2. 拡大（小さい画像はOCR精度が落ちるため）
    upscaled = self._upscale(cropped)
    if upscaled is not cropped:
      steps.append('upscale')

    # 3. グレースケール変換
    gray = self._toGrayscale(upscaled)
    steps.append('grayscale')

    if mode == 'handwritten':
      # 手書きモード: 軽めの前処理（文字を潰さない）
      result = self._preprocessHandwritten(gray, steps)
    elif mode == 'printed':
      # 活字モード: 強めの前処理
      result = self._preprocessPrinted(gray, steps)
    else:
      # 自動: 手書き寄りのソフトな処理
      result = self._preprocessHandwritten(gray, steps)

    return ProcessedImage(
      image=result,
      originalSize=originalSize,
      croppedRegion=region,
      preprocessSteps=steps,
    )

  def _preprocessHandwritten(self, gray: np.ndarray, steps: list[str]) -> np.ndarray:
    """手書き向け: 文字の線を保護する軽い前処理"""
    # 軽いノイズ除去（ガウシアンブラー）
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    steps.append('gaussian_blur')

    # コントラスト強調（穏やか）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    steps.append('clahe')

    # Otsu二値化（手書きに適している）
    _, binarized = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps.append('otsu_binarize')

    # モルフォロジー処理で文字を太くする（細い線の認識率向上）
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binarized, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    steps.append('morphology_close')

    return eroded

  def _preprocessPrinted(self, gray: np.ndarray, steps: list[str]) -> np.ndarray:
    """活字向け: 強めのノイズ除去と二値化"""
    strength = self.settings.denoiseStrength
    denoised = cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)
    steps.append('denoise')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    steps.append('clahe')

    blockSize = self.settings.binarizeBlockSize
    if blockSize % 2 == 0:
      blockSize += 1
    binarized = cv2.adaptiveThreshold(
      enhanced, 255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY,
      blockSize, 2,
    )
    steps.append('adaptive_binarize')

    return binarized

  def _crop(self, image: np.ndarray, region: RegionSelection) -> np.ndarray:
    """指定領域をクロップ"""
    x1, y1, x2, y2 = region.toTuple()
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return image[y1:y2, x1:x2]

  def _upscale(self, image: np.ndarray, minHeight: int = 150) -> np.ndarray:
    """小さすぎる画像を拡大してOCR精度を上げる"""
    h, w = image.shape[:2]
    if h < minHeight:
      scale = minHeight / h
      newW = int(w * scale)
      newH = int(h * scale)
      return cv2.resize(image, (newW, newH), interpolation=cv2.INTER_CUBIC)
    return image

  def _toGrayscale(self, image: np.ndarray) -> np.ndarray:
    """グレースケール変換"""
    if len(image.shape) == 3:
      return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
