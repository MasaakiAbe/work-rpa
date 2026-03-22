"""画像キャンバスコンポーネント - 全画面デフォルト + 領域調整UI"""

from __future__ import annotations

import streamlit as st
from PIL import Image

from src.models.data_models import RegionSelection

# 画像プレビューの最大表示幅
PREVIEW_MAX_WIDTH = 300


def renderImageCanvas(image: Image.Image) -> RegionSelection | None:
  """
  デフォルトは全画面選択。
  画像プレビューは常時表示、スライダーはexpanderに格納。
  """
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

  return RegionSelection(x=x, y=y, width=w, height=h)
