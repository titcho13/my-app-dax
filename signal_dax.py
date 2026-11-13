import os, time, datetime
import requests
import pandas as pd
import yfinance as yf

TICKER = "^GDAXI"
INTERVALLE_MIN = 15
DUREE_MAX_MIN = int(os.getenv("DUREE_MAX_MIN", "330"))
TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


def telegram(msg):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": msg},
        timeout=20,
    )
    if not r.ok:
        print("Erreur Telegram :", r.text)


def charger():
    df = yf.Ticker(TICKER).history(period="1mo", interval="15m").reset_index()
    df = df.rename(columns={"Datetime": "timestamp", "Date": "timestamp", "Open": "open",
                            "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df["timestamp"].iloc[-1] + pd.Timedelta(minutes=INTERVALLE_MIN) > now:
        df = df.iloc[:-1]
    return df.reset_index(drop=True)


def calcul_signaux(df):
    df["prev_close"] = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["prev_close"]).abs(),
                    (df["low"] - df["prev_close"]).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ret"] = df["close"].pct_change()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["regime"] = "range"
    df.loc[df["close"] > df["ma_50"] * 1.002, "regime"] = "tendance_haussiere"
    df.loc[df["close"] < df["ma_50"] * 0.998, "regime"] = "tendance_baissiere"

    df["direction"] = None
    df.loc[(df["regime"] == "tendance_haussiere") & (df["ret"] < -0.001), "direction"] = "LONG"
    df.loc[(df["regime"] == "tendance_baissiere") & (df["ret"] > 0.001), "direction"] = "SHORT"

    df["jour"] = df["timestamp"].dt.strftime("%A")
    df["heure"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    return df


def valide(row):
    d, jour, h, m = row["direction"], row["jour"], row["heure"], row["minute"]
    if jour == "Monday":
        return False
    if jour == "Tuesday" and d != "LONG":
        return False
    if jour in ("Thursday", "Friday") and d != "SHORT":
        return False
    if d == "LONG" and h == 8:
        return False
    if d == "SHORT" and h in (11, 13):
        return False
    if d == "SHORT" and 8 * 60 <= h * 60 + m <= 9 * 60 + 15:
        return False
    return True


def cycle(deja_vus):
    df = calcul_signaux(charger())
    for _, row in df.dropna(subset=["direction"]).iterrows():
        ts = row["timestamp"]
        if ts in deja_vus:
            continue
        deja_vus.add(ts)
        if not valide(row):
            continue
        emoji = "🟢" if row["direction"] == "LONG" else "🔴"
        telegram(
            f"⚠️ ALERTE DAX {emoji} {row['direction']}\n"
            f"Heure (UTC) : {ts:%Y-%m-%d %H:%M}\n"
            f"Régime : {row['regime']}\n"
            f"Entrée : {row['close']:.2f} pts\n"
            f"ATR : {row['atr']:.2f} pts"
        )
        print("Alerte envoyée :", ts)


def main():
    telegram("✅ Test signal_dax depuis GitHub Actions")
    debut = time.time()
    df0 = calcul_signaux(charger())
    deja_vus = set(df0.dropna(subset=["direction"])["timestamp"])
    print(f"Démarrage {datetime.datetime.utcnow():%H:%M} UTC, {len(deja_vus)} signaux passés ignorés")

    while (time.time() - debut) < DUREE_MAX_MIN * 60:
        now = datetime.datetime.utcnow()
        attente = (INTERVALLE_MIN - now.minute % INTERVALLE_MIN) * 60 - now.second + 45
        time.sleep(attente)
        try:
            cycle(deja_vus)
        except Exception as e:
            print("Erreur cycle :", e)


if __name__ == "__main__":
    main()
