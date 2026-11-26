import os, json, requests
import pandas as pd, yfinance as yf, numpy as np

TICKER, INTERVALLE_MIN = "^GDAXI", 15
FENETRE_MIN = 45                   # on n'alerte que sur les signaux récents
SENT_FILE = "sent_my_system.json"
TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

def charger():
    data = yf.download(TICKER, period="60d", interval="15m", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data.reset_index().rename(columns={
        "Datetime": "timestamp", "Date": "timestamp", "Open": "open",
        "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df["timestamp"].iloc[-1] + pd.Timedelta(minutes=INTERVALLE_MIN) > now:
        df = df.iloc[:-1]
    return df

def calcul_signaux(df):
    df["prev_close"] = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["prev_close"]).abs(),
                    (df["low"] - df["prev_close"]).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ret"] = df["close"].pct_change()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_200"] = df["close"].rolling(200).mean()
    df["slope_200"] = (df["ma_200"] - df["ma_200"].shift(20)) / df["atr"]

    a = 1 / 14
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), df.index)
    atr_w = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr_w
    mdi = 100 * mdm.ewm(alpha=a, adjust=False).mean() / atr_w
    df["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=a, adjust=False).mean()

    df["regime"] = "range"
    df.loc[df["close"] > df["ma_50"] * 1.002, "regime"] = "tendance_haussiere"
    df.loc[df["close"] < df["ma_50"] * 0.998, "regime"] = "tendance_baissiere"

    df["direction"] = None
    adx_ok = df["adx"] > 20
    long_f = (df["close"] > df["ma_200"]) & (df["slope_200"] > 0.15) & adx_ok
    short_f = (df["close"] < df["ma_200"]) & (df["slope_200"] < -0.15) & adx_ok
    df.loc[(df["regime"] == "tendance_haussiere") & (df["ret"] < -0.001) & long_f, "direction"] = "LONG"
    df.loc[(df["regime"] == "tendance_baissiere") & (df["ret"] > 0.001) & short_f, "direction"] = "SHORT"

    df["jour"] = df["timestamp"].dt.strftime("%A")
    df["heure"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    return df

def valide(row):
    d, jour, h, m = row["direction"], row["jour"], row["heure"], row["minute"]
    if jour == "Monday": return False
    if jour == "Tuesday" and d != "LONG": return False
    if jour in ("Thursday", "Friday") and d != "SHORT": return False
    if d == "LONG" and h == 8: return False
    if d == "SHORT" and h in (11, 13): return False
    if d == "SHORT" and 8*60 <= h*60 + m <= 9*60 + 15: return False
    return True

def telegram(txt):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": txt}, timeout=20)
    r.raise_for_status()

def main():
    df = calcul_signaux(charger())
    df["valide"] = [valide(r) if r["direction"] else False for _, r in df.iterrows()]
    debut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=FENETRE_MIN)
    sig = df[(df["timestamp"] >= debut) & df["valide"]]

    sent = json.load(open(SENT_FILE)) if os.path.exists(SENT_FILE) else []
    for _, r in sig.iterrows():
        cle = f"{r['timestamp'].isoformat()}_{r['direction']}"
        if cle in sent:
            continue
        emoji = "🟢" if r["direction"] == "LONG" else "🔴"
        telegram(
            f"🧠 MY SYSTEM\n"
            f"{emoji} Signal MA50 DAX {r['direction']}\n"
            f"Heure UTC : {r['timestamp'].strftime('%d/%m %H:%M')}\n"
            f"Entrée : {r['close']:.1f}\n"
            f"ATR : {r['atr']:.1f} | ADX : {r['adx']:.1f}\n"
            f"Pente MM200 : {r['slope_200']:.2f} ATR\n"
            f"Régime : {r['regime']}")
        sent.append(cle)
    json.dump(sent[-200:], open(SENT_FILE, "w"))

if __name__ == "__main__":
    main()
