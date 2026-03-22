"""基底エージェントクラス - 全エージェントの共通インターフェース"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
  """エージェントの基底クラス"""

  def __init__(self, name: str):
    self.name = name
    self.logger = logging.getLogger(f'agent.{name}')

  @abstractmethod
  def process(self, inputData: Any) -> Any:
    """メイン処理 - サブクラスで実装"""
    ...

  def execute(self, inputData: Any) -> Any:
    """実行ラッパー - ログと計測を付加"""
    self.logger.info(f'[{self.name}] 処理開始')
    startTime = time.time()
    try:
      result = self.process(inputData)
      elapsedMs = int((time.time() - startTime) * 1000)
      self.logger.info(f'[{self.name}] 処理完了 ({elapsedMs}ms)')
      return result
    except Exception as e:
      elapsedMs = int((time.time() - startTime) * 1000)
      self.logger.error(f'[{self.name}] エラー ({elapsedMs}ms): {e}')
      raise
