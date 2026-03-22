"""OCR結果表示パネル"""

from __future__ import annotations

import streamlit as st
import cv2
from PIL import Image

from src.models.data_models import PipelineResult


def renderResultPanel(result: PipelineResult) -> None:
  """OCR結果を表示する"""
  if not result.success:
    st.error(f'エラーが発生しました: {result.errorMessage}')
    return

  ocrResult = result.ocrResult
  processedImage = result.processedImage

  # 結果テキスト表示
  st.subheader('認識結果')

  if ocrResult.text:
    # テキスト量に応じて高さを自動調整（1行あたり約30px）
    lineCount = ocrResult.text.count('\n') + 1
    textHeight = max(200, min(lineCount * 30, 800))

    # コピー可能なテキストエリア
    st.text_area(
      'OCR結果（選択してコピーできます）',
      value=ocrResult.text,
      height=textHeight,
      key='ocr_result_text',
    )

    # メタ情報
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric('信頼度', f'{ocrResult.confidence:.1%}')
    with col2:
      st.metric('処理時間', f'{ocrResult.processingTimeMs}ms')
    with col3:
      st.metric('エンジン', ocrResult.engineUsed)

    # Tesseractフォールバック時の警告
    if 'tesseract' in ocrResult.engineUsed:
      st.warning(
        'Tesseractで処理されました（手書き文字の精度が低い場合があります）。\n'
        'Gemini APIキーが設定されているか確認してください。'
      )
  else:
    st.warning('テキストが認識されませんでした。領域を変えて再度お試しください。')

  # 前処理済み画像のプレビュー
  if processedImage is not None:
    with st.expander('前処理済み画像を表示'):
      previewImg = Image.fromarray(processedImage.image)
      st.image(previewImg, caption='前処理後', use_container_width=True)
      st.caption(f'適用ステップ: {" → ".join(processedImage.preprocessSteps)}')
