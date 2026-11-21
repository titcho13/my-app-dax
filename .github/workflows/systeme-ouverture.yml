name: Systeme ouverture DAX

on:
  schedule:
    - cron: "50 6 * * 1-5"     # 06:50 UTC = 07:50 Paris (hiver) / 08:50 (été) ; le script attend 09:00 Paris
  workflow_dispatch:

concurrency:
  group: systeme-ouverture
  cancel-in-progress: true

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 300

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Installer les dépendances
        run: pip install yfinance scikit-learn pandas numpy requests

      - name: Lancer le système
        env:
          TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python systeme_ouverture_ml.py
