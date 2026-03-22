"""OCR結果表示パネル"""

from __future__ import annotations

import re

import streamlit as st
from PIL import Image

from src.models.data_models import PipelineResult


def _highlightUncertain(text: str) -> str:
  """[?文字?] を赤色ハイライトに変換する"""
  return re.sub(
    r'\[\?(.+?)\?\]',
    r':red-background[**\1❓**]',
    text,
  )


def _parseExtractedItems(text: str) -> list[tuple[str, str]]:
  """
  「項目名: 値」形式のテキストをパースする。
  例: "会社名: 株式会社テスト" → [("会社名", "株式会社テスト")]
  """
  items = []
  for line in text.strip().splitlines():
    line = line.strip()
    if not line:
      continue
    # 「項目名: 値」または「- 項目名: 値」形式にマッチ
    match = re.match(r'^[-・*]*\s*(.+?)\s*[:：]\s*(.+)$', line)
    if match:
      items.append((match.group(1).strip(), match.group(2).strip()))
  return items


def renderResultPanel(result: PipelineResult, promptMode: str = 'テキスト読み取り') -> None:
  """OCR結果を表示する"""
  if not result.success:
    st.error(f'エラーが発生しました: {result.errorMessage}')
    return

  ocrResult = result.ocrResult
  processedImage = result.processedImage

  st.subheader('認識結果')

  if not ocrResult.text:
    st.warning('テキストが認識されませんでした。領域を変えて再度お試しください。')
    return

  if promptMode == '項目抽出':
    _renderExtractedItems(ocrResult.text)
  else:
    _renderRawText(ocrResult.text)

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

  # 前処理済み画像のプレビュー
  if processedImage is not None:
    with st.expander('前処理済み画像を表示'):
      previewImg = Image.fromarray(processedImage.image)
      st.image(previewImg, caption='前処理後', use_container_width=True)
      st.caption(f'適用ステップ: {" → ".join(processedImage.preprocessSteps)}')


def _renderRawText(text: str) -> None:
  """テキスト読み取りモードの表示"""
  # 低確度部分をハイライト表示
  if '[?' in text:
    st.markdown(_highlightUncertain(text))
    st.divider()

  # コピー用テキストエリア
  lineCount = text.count('\n') + 1
  textHeight = max(200, min(lineCount * 30, 800))
  st.text_area(
    'OCR結果（選択してコピーできます）',
    value=text,
    height=textHeight,
    key='ocr_result_text',
  )


def _renderExtractedItems(text: str) -> None:
  """項目抽出モードの表示（テーブル + コピー用テキスト）"""
  items = _parseExtractedItems(text)

  if items:
    # テーブル表示
    for itemName, itemValue in items:
      col1, col2 = st.columns([1, 3])
      with col1:
        st.markdown(f'**{itemName}**')
      with col2:
        if itemValue == '-':
          st.markdown(f'*{itemValue}*')
        else:
          st.markdown(_highlightUncertain(itemValue))
    st.divider()

  # 生テキストも折りたたみで表示（コピー用）
  with st.expander('生テキストを表示（コピー用）'):
    st.text_area(
      'OCR結果',
      value=text,
      height=300,
      key='ocr_result_text',
    )
