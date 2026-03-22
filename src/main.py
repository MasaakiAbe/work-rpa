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


def loadPdfPages(uploadedFile) -> list[Image.Image]:
  """PDFを画像のリストに変換する"""
  import fitz
  pdfBytes = uploadedFile.read()
  uploadedFile.seek(0)
  doc = fitz.open(stream=pdfBytes, filetype='pdf')
  pages = []
  for page in doc:
    # 高解像度でレンダリング（DPI 200相当）
    mat = fitz.Matrix(200 / 72, 200 / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    pages.append(img)
  doc.close()
  return pages


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
    if engineName == 'GEMINI' and settings.geminiApiKey:
      st.success(f'OCRエンジン: **{engineName}**')
      st.caption('Gemini Vision - 手書き日本語に最適')
    elif engineName == 'GEMINI' and not settings.geminiApiKey:
      st.error('GEMINI_API_KEY が未設定です')
      st.caption('Streamlit Cloud: Settings → Secrets に設定してください')
    else:
      st.info(f'OCRエンジン: **{engineName}**')

  # ファイルアップロード
  st.subheader('ファイルを選択')
  uploadedFile = st.file_uploader(
    'ここにファイルをドラッグ＆ドロップ、またはクリックして選択',
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pdf'],
    label_visibility='visible',
  )

  if uploadedFile is None:
    st.info('画像（PNG, JPG）またはPDFファイルをアップロードしてください')
    return

  # PDF / 画像の読み込み
  isPdf = uploadedFile.name.lower().endswith('.pdf')

  if isPdf:
    pages = loadPdfPages(uploadedFile)
    totalPages = len(pages)
    image = pages[0]  # デフォルトは1ページ目
  else:
    pages = None
    totalPages = 1
    image = Image.open(uploadedFile).convert('RGB')

  # 領域選択（デフォルト全画面）+ PDF時はページ選択
  region, image = renderImageCanvas(image, pages=pages, totalPages=totalPages)

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
        st.session_state['lastPromptMode'] = promptMode

  if 'lastResult' in st.session_state:
    renderResultPanel(
      st.session_state['lastResult'],
      promptMode=st.session_state.get('lastPromptMode', 'テキスト読み取り'),
    )


if __name__ == '__main__':
  main()
