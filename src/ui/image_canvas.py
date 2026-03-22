"""画像キャンバスコンポーネント - 全画面デフォルト + 領域/ページ調整UI"""

from __future__ import annotations

import streamlit as st
from PIL import Image

from src.models.data_models import RegionSelection

# 画像プレビューの最大表示幅
PREVIEW_MAX_WIDTH = 300


def renderImageCanvas(
  image: Image.Image,
  pages: list[Image.Image] | None = None,
  totalPages: int = 1,
) -> tuple[RegionSelection | None, Image.Image]:
  """
  デフォルトは1ページ目・全画面選択。
  PDFの場合はページ選択も可能。

  Returns:
    (RegionSelection, 選択ページの画像)
  """
  # PDFのページ選択
  if pages and totalPages > 1:
    pageNum = st.selectbox(
      f'ページ選択（全{totalPages}ページ）',
      range(1, totalPages + 1),
      index=0,
      format_func=lambda x: f'{x} ページ目',
    )
    image = pages[pageNum - 1]

  imgWidth, imgHeight = image.size

  # デフォルト: 全画面
  x, y, w, h = 0, 0, imgWidth, imgHeight

  # 画像プレビュー（常時表示・小さめ）
  st.image(image, width=PREVIEW_MAX_WIDTH)

  # スライダーのみexpanderに格納
  with st.expander('領域を調整する（デフォルト: 全体）'):
    col1, col2 = st.columns(2)
    with col1:
      x = st.slider('X（左端）', 0, imgWidth, 0, key='region_x')
      w = st.slider('幅', 1, imgWidth - x, imgWidth - x, key='region_w')
    with col2:
      y = st.slider('Y（上端）', 0, imgHeight, 0, key='region_y')
      h = st.slider('高さ', 1, imgHeight - y, imgHeight - y, key='region_h')

  return RegionSelection(x=x, y=y, width=w, height=h), image
