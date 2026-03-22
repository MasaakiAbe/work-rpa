"""セットアップスクリプト - 環境チェックと初期設定"""

import shutil
import subprocess
import sys
from pathlib import Path


def checkPython():
  print(f'✓ Python {sys.version}')


def checkTesseract():
  # よくあるWindowsパス
  commonPaths = [
    'C:/Program Files/Tesseract-OCR/tesseract.exe',
    'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe',
  ]

  # PATHから検索
  tesseractPath = shutil.which('tesseract')
  if tesseractPath:
    print(f'✓ Tesseract found: {tesseractPath}')
    return

  # 標準パスを確認
  for p in commonPaths:
    if Path(p).exists():
      print(f'✓ Tesseract found: {p}')
      print(f'  → .env に TESSERACT_CMD={p} を設定してください')
      return

  print('✗ Tesseract がインストールされていません')
  print()
  print('  インストール方法:')
  print('  1. https://github.com/UB-Mannheim/tesseract/wiki からダウンロード')
  print('  2. インストール時に「Additional language data」で「Japanese」を選択')
  print('  3. インストール後、.env に TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe を設定')
  print()


def checkPackages():
  required = [
    'streamlit', 'cv2', 'PIL', 'pytesseract',
    'pydantic', 'pydantic_settings',
  ]
  missing = []
  for pkg in required:
    try:
      __import__(pkg)
    except ImportError:
      missing.append(pkg)

  if missing:
    print(f'✗ 不足パッケージ: {", ".join(missing)}')
    print('  → pip install -e . で依存パッケージをインストールしてください')
  else:
    print('✓ 必要なパッケージはすべてインストール済み')


def createEnvFile():
  envPath = Path('.env')
  examplePath = Path('.env.example')
  if not envPath.exists() and examplePath.exists():
    envPath.write_text(examplePath.read_text(encoding='utf-8'), encoding='utf-8')
    print('✓ .env ファイルを作成しました')
  elif envPath.exists():
    print('✓ .env ファイルは存在します')


def main():
  print('=' * 50)
  print('FAX OCR Agent Team - セットアップチェック')
  print('=' * 50)
  print()

  checkPython()
  checkPackages()
  checkTesseract()
  createEnvFile()

  print()
  print('セットアップが完了したら以下で起動:')
  print('  streamlit run src/main.py')


if __name__ == '__main__':
  main()
