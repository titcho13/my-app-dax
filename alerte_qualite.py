# -*- coding: utf-8 -*-
import os
import numpy as np, pandas as pd, requests, yfinance as yf

TZ = "Europe/Paris"
ST_PERIOD, ST_MULT = 14, 3.0
MA_LEN, ATR_LEN, ADX_LEN = 200, 14, 14
SLOPE_N, ATR_SLOPE_N, SLOPE_THR = 20, 20, 0.20
MAX_AGE_MIN = 14      # n'alerte que si la bougie vient de clôturer (évite les doublons)
MVT_MIN = 3.0         # mouvement minimal depuis l'ouverture du jour, en ATR

TOKEN   = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
FORCE   = os.environ.get("FORCE", "false").lower() == "true"


def fetch_data():
    d = yf.download("^GDAXI", interval="15m", period="30d",
                    auto_adjust=False, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True).tz_convert(TZ)
    return d[~d.index.duplicated()].sort_index()


def wilder(s, n): return s.ewm(alpha=1/n, adjust=False).mean()

def true_range(d):
    pc = d["Close"].shift()
    return pd.concat([d["High"]-d["Low"], (d["High"]-pc).abs(),
                      (d["Low"]-pc).abs()], axis=1).max(axis=1)

def supertrend_dir(d, period, mult):
    atr = wilder(true_range(d), period)
    hl2 = (d["High"] + d["Low"]) / 2
    bub, blb = (hl2 + mult*atr).values, (hl2 - mult*atr).values
    c = d["Close"].values; n = len(d)
    fub, flb = bub.copy(), blb.copy()
    dirn = np.ones(n, dtype=int)
    for i in range(1, n):
        fub[i] = bub[i] if (bub[i] < fub[i-1] or c[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = blb[i] if (blb[i] > flb[i-1] or c[i-1] < flb[i-1]) else flb[i-1]
        dirn[i] = (-1 if c[i] < flb[i] else 1) if dirn[i-1] == 1 else (1 if c[i] > fub[i] else -1)
    return pd.Series(dirn, d.index)

def adx_f(d, n):
    up, dn = d["High"].diff(), -d["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), d.index)
    atr = wilder(true_range(d), n)
    pdi, mdi = 100*wilder(pdm, n)/atr, 100*wilder(mdm, n)/atr
    return wilder(100*(pdi-mdi).abs()/(pdi+mdi), n)


def telegram(text):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=20)
    r.raise_for_status()


def main():
    df = fetch_data()
    now = pd.Timestamp.now(tz=TZ)

    # on ne garde que les bougies clôturées
    df = df[df.index + pd.Timedelta(minutes=15) <= now]
    if len(df) < MA_LEN + SLOPE_N + 5:
        print("Pas assez de données"); return

    MA  = df["Close"].rolling(MA_LEN).mean()
    ATR = wilder(true_range(df), ATR_LEN)
    ADX = adx_f(df, ADX_LEN)
    DIR = supertrend_dir(df, ST_PERIOD, ST_MULT)
    SLOPE = (MA - MA.shift(SLOPE_N)) / ATR
    ATR_SLOPE = (ATR - ATR.shift(ATR_SLOPE_N)) / ATR.shift(ATR_SLOPE_N)
    px = df["Close"]

    # achats uniquement
    long_q = (px > MA) & (DIR == 1) & (SLOPE > SLOPE_THR) \
             & (ATR_SLOPE >= ATR_SLOPE.shift()) & (ADX > 20)
    dec = pd.Series(np.where(long_q, 1, 0), df.index)

    # mouvement depuis l'ouverture du jour, en ATR
    day_open = df["Open"].groupby(df.index.date).transform("first")
    MVT = (px - day_open) / ATR

    i = len(df) - 1
    ts = df.index[i]
    age_min = (now - (ts + pd.Timedelta(minutes=15))).total_seconds() / 60
    cur, prev = dec.iloc[i], dec.iloc[i-1]
    mvt = float(MVT.iloc[i])
    print(f"{ts:%d/%m %H:%M} | décision={cur} (préc.={prev}) | mvt ouverture={mvt:+.1f} ATR | âge={age_min:.0f} min")

    new_signal = cur == 1 and prev != 1 and age_min <= MAX_AGE_MIN
    if new_signal and mvt < MVT_MIN and not FORCE:
        print(f"Signal ignoré : mouvement depuis l'ouverture {mvt:.1f} ATR < {MVT_MIN}")
        return
    if not (new_signal or FORCE):
        return

    adx_v, atr_pts, prix = ADX.iloc[i], ATR.iloc[i], px.iloc[i]
    if adx_v < 20:   sl_m, tp_m, profil = 1.0, 1.5, "PRUDENT"
    elif adx_v < 25: sl_m, tp_m, profil = 1.0, 2.0, "STANDARD"
    elif adx_v < 35: sl_m, tp_m, profil = 1.0, 3.0, "TENDANCE FORTE"
    else:            sl_m, tp_m, profil = 1.2, 4.0, "ACCÉLÉRATION"
    sl_pts, tp_pts = round(atr_pts*sl_m), round(atr_pts*tp_m)

    titre = "🚀 ACHAT QUALITÉ" if cur == 1 else "TEST (aucun signal actif)"

    msg = (
        f"{titre} - DAX 15 min\n"
        f"Bougie {ts:%d/%m %H:%M}\n\n"
        f"Entrée : {prix:.0f}\n"
        f"Stop Loss : {prix - sl_pts:.0f} (-{sl_pts} pts)\n"
        f"Take Profit : {prix + tp_pts:.0f} (+{tp_pts} pts)\n"
        f"Profil : {profil} | Ratio 1:{tp_m/sl_m:.1f}\n\n"
        f"Mouvement depuis ouverture : {mvt:+.1f} ATR (seuil {MVT_MIN})\n"
        f"Pente MM200 : {SLOPE.iloc[i]:+.2f} ATR\n"
        f"Pente ATR : {ATR_SLOPE.iloc[i]*100:+.1f} %\n"
        f"ADX : {adx_v:.1f} | ATR : {atr_pts:.1f} pts"
    )
    telegram(msg)
    print("Message Telegram envoyé")


if __name__ == "__main__":
    main()
