"""データモデル定義 - Agent間で共有するデータ構造"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RegionSelection:
  """ユーザが選択した矩形領域"""
  x: int       # 左上X座標
  y: int       # 左上Y座標
  width: int   # 幅
  height: int  # 高さ

  def toTuple(self) -> tuple[int, int, int, int]:
    """(x, y, x+w, y+h) 形式で返す"""
    return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class ProcessedImage:
  """前処理済み画像データ"""
  image: np.ndarray                       # 前処理済み画像
  originalSize: tuple[int, int]           # 元画像サイズ (w, h)
  croppedRegion: RegionSelection          # クロップ領域
  preprocessSteps: list[str] = field(default_factory=list)  # 適用した前処理


@dataclass
class OcrResult:
  """OCR認識結果"""
  text: str                     # 認識テキスト
  confidence: float             # 信頼度 (0.0 ~ 1.0)
  engineUsed: str               # 使用エンジン名
  rawResponse: dict | None = None  # デバッグ用生レスポンス
  processingTimeMs: int = 0     # 処理時間(ms)


@dataclass
class PipelineResult:
  """パイプライン全体の結果"""
  ocrResult: OcrResult | None = None
  processedImage: ProcessedImage | None = None
  success: bool = False
  errorMessage: str | None = None
