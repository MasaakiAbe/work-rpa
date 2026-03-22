"""
FAX OCR Agent Team - メインアプリケーション
Streamlit エントリーポイント

起動: streamlit run src/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# プロジェクトルートをパスに追加
projectRoot = Path(__file__).parent.parent
if str(projectRoot) not in sys.path:
  sys.path.insert(0, str(projectRoot))

from src.agents.orchestrator import Orchestrator
from src.ui.image_canvas import renderImageCanvas
from src.ui.result_panel import renderResultPanel

# ログ設定
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)


def main():
  st.set_page_config(
    page_title='FAX OCR Agent Team',
    page_icon='📠',
    layout='wide',
  )

  st.title('📠 FAX OCR Agent Team')
  st.caption('画像をアップロード → 領域を選択 → テキストを認識')

  # サイドバー: 設定
  with st.sidebar:
    st.header('設定')
    from src.config.settings import getSettings
    settings = getSettings()
    engineName = settings.ocrPrimaryEngine.upper()
    st.success(f'OCRエンジン: **{engineName}**')
    if engineName == 'GEMINI':
      st.caption('Gemini Vision - 手書き日本語に最適')
    st.caption('エンジン切替は `.env` の `OCR_PRIMARY_ENGINE` を変更')

  # ファイルアップロード
  uploadedFile = st.file_uploader(
    'FAX画像をアップロード',
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pdf'],
    help='対応形式: PNG, JPG, TIFF, BMP, PDF',
  )

  if uploadedFile is None:
    st.info('👆 FAX画像ファイルをアップロードしてください')
    return

  # 画像読み込み
  image = Image.open(uploadedFile).convert('RGB')

  # 領域選択（デフォルト全画面）
  region = renderImageCanvas(image)

  # モード選択 + OCR実行ボタン
  if region is not None:
    promptMode = st.radio(
      '認識モード',
      ['テキスト読み取り', '項目抽出'],
      horizontal=True,
      help='テキスト読み取り: そのまま文字を読み取り / 項目抽出: 会社名・担当者・電話番号・製品などを構造化抽出',
    )

    if st.button('🔍 テキスト認識を実行', type='primary', use_container_width=True):
      with st.spinner('Agent Team が処理中...'):
        orchestrator = Orchestrator()
        result = orchestrator.execute({
          'image': image,
          'region': region,
          'promptMode': promptMode,
        })
        st.session_state['lastResult'] = result

  if 'lastResult' in st.session_state:
    renderResultPanel(st.session_state['lastResult'])


if __name__ == '__main__':
  main()
