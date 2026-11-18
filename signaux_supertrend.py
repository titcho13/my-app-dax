#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
#  DAX 15 min — Supertrend 14,3 / MM200 + filtre anti-épuisement + proba
#  Version HEADLESS pour GitHub Actions : alertes Telegram (pas de graphique, pas de son).
#  Secrets GitHub requis : TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
#  Longs : lun-mar-mer | Shorts : mer-jeu-ven | Entrées : 08h00-12h00 (Paris)
# ============================================================
import os, time, warnings
import numpy as np, pandas as pd, requests, yfinance as yf
warnings.filterwarnings("ignore")

# ================= PARAMÈTRES =================
TZ = "Europe/Paris"
ST_PERIOD, ST_MULT = 14, 3.0
MA_LEN, ATR_LEN, ADX_LEN = 200, 14, 14
ADX_MIN = 25
SLOPE_N, SLOPE_THR = 20, 0.20
ATR_SLOPE_N, ATR_THR = 20, -0.10
POLL_SEC = 60
LONG_DAYS, SHORT_DAYS = {0, 1, 2}, {2, 3, 4}
JOURS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
NO_ENTRY_AFTER = (12, 0)
END_LOOP = (17, 35)
LOOP_MAX_S = 5.75 * 3600          # limite GitHub : 6 h par job
AGE_MAX_MIN = 20                  # au démarrage : alerte quand même si la bougie date de moins de 20 min
MIN_N = 8
CTX_REFRESH_SEC = 3600
EXT_MAX = 4.0
RSI_LEN, RSI_OB, RSI_OS = 14, 70, 30
ADX_DROP = 5.0
LAB_FETCH_RETRY = 3

# ============================================================
# TELEGRAM
# ============================================================
def telegram(texte):
    tok, chat = os.environ.get("TELEGRAM_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
    if not (tok and chat):
        print("⚠ TELEGRAM_TOKEN / TELEGRAM_CHAT_ID absents :", texte); return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": texte}, timeout=10)
        if not r.ok: print(f"⚠ Telegram {r.status_code} : {r.text[:200]}")
        return r.ok
    except Exception as e:
        print("⚠ Telegram :", e); return False

# ============================================================
# CONTEXTE JOURNALIER (biais + événements forts du jour)
# ============================================================
CFG = {"symbole": "^GDAXI", "periode": "3y", "ema_court": 20, "ema_rapide": 50, "ema_lente": 200,
       "er_fenetre": 20, "er_tendance": 0.30, "er_range": 0.15, "atr_fenetre": 14, "swing": 5,
       "devises": ["EUR", "USD"], "session": (9.0, 17.5)}
UA = {"User-Agent": "Mozilla/5.0"}

def charger_journalier(c):
    df = yf.download(c["symbole"], period=c["periode"], interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty: raise RuntimeError("Aucune donnée journalière.")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def retirer_barre_du_jour(df, c):
    now = pd.Timestamp.now(tz=TZ)
    if df.index[-1].date() == now.date() and (now.hour + now.minute / 60) < c["session"][1] + 0.3: df = df.iloc[:-1]
    return df

def pivots(d, n):
    hi, lo = d["high"].values, d["low"].values; ph, pl = [], []
    for i in range(n, len(d) - n):
        if hi[i] == hi[i - n:i + n + 1].max(): ph.append((d.index[i], hi[i]))
        if lo[i] == lo[i - n:i + n + 1].min(): pl.append((d.index[i], lo[i]))
    return ph, pl

def analyser_regime(df, c):
    d = df.copy(); cl = d["close"]
    for k in (c["ema_court"], c["ema_rapide"], c["ema_lente"]): d[f"ema{k}"] = cl.ewm(span=k, adjust=False).mean()
    n = c["er_fenetre"]; er = ((cl - cl.shift(n)).abs() / cl.diff().abs().rolling(n).sum()).iloc[-1]
    pc = cl.shift(1)
    d["vol20"] = np.log(cl / pc).rolling(20).std() * np.sqrt(252)
    last = d.iloc[-1]; rang_vol = (d["vol20"].iloc[-252:] <= last["vol20"]).mean() * 100
    ph, pl = pivots(d, c["swing"])
    if len(ph) >= 2 and len(pl) >= 2:
        hh, hl = ph[-1][1] > ph[-2][1], pl[-1][1] > pl[-2][1]
        s_struct = 1 if hh and hl else -1 if (not hh and not hl) else 0
    else: s_struct = 0
    pente = d["ema50"].iloc[-1] / d["ema50"].iloc[-11] - 1
    score = sum([1 if last["close"] > last["ema200"] else -1, 1 if last["ema50"] > last["ema200"] else -1,
                 1 if last["close"] > last["ema50"] else -1, 1 if pente > 0 else -1,
                 1 if cl.iloc[-1] > cl.iloc[-21] else -1, s_struct])
    biais = "HAUSSIER" if score >= 3 else "BAISSIER" if score <= -3 else "NEUTRE"
    mode = "TENDANCE" if er >= c["er_tendance"] else "RANGE" if er <= c["er_range"] else "TRANSITION"
    vol = "élevée" if rang_vol >= 80 else "faible" if rang_vol <= 20 else "normale"
    return dict(biais=biais, score=score, mode=mode, vol=vol)

def evenements_forts(now):
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", headers=UA, timeout=10); r.raise_for_status()
        out = []
        for e in r.json():
            dt_ = pd.Timestamp(e["date"]); dt_ = dt_.tz_localize("America/New_York") if dt_.tzinfo is None else dt_
            dt_ = dt_.tz_convert(TZ)
            if dt_.date() == now.date() and e.get("country") in CFG["devises"] and e.get("impact") == "High":
                out.append((dt_, f"{dt_:%H:%M} {e['country']} {e['title']}"))
        return [t for _, t in sorted(out)][:4]
    except Exception:
        return None

CTX, CTX_T = {}, 0.0
def maj_contexte(force=False):
    global CTX, CTX_T
    if not force and time.time() - CTX_T < CTX_REFRESH_SEC: return
    try:
        df = retirer_barre_du_jour(charger_journalier(CFG), CFG)
        CTX = analyser_regime(df, CFG)
    except Exception as e:
        print("⚠ Contexte indisponible :", e); CTX = {}
    CTX["events"] = evenements_forts(pd.Timestamp.now(tz=TZ)); CTX_T = time.time()

def bias_txt(sens):
    b = CTX.get("biais")
    if b in ("HAUSSIER", "BAISSIER"):
        return "AVEC le biais journalier" if (b == "HAUSSIER") == (sens == 1) else "CONTRE le biais journalier"
    return "biais journalier neutre"

def ctx_ligne():
    if not CTX.get("biais"): return "Contexte : indisponible"
    ev = CTX.get("events")
    evt = "calendrier indisponible" if ev is None else ("aucun événement fort" if not ev else " | ".join(ev))
    return f"Contexte : biais {CTX['biais']} ({CTX['score']:+d}/6), {CTX['mode']}, volatilité {CTX['vol']}\nÉvénements forts : {evt}"

# ============================================================
# DONNÉES + INDICATEURS 15 min
# ============================================================
def fetch_data(period="60d"):
    err = None
    for _ in range(LAB_FETCH_RETRY):
        try:
            d = yf.download("^GDAXI", interval="15m", period=period, auto_adjust=False, progress=False)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            d = d[["Open", "High", "Low", "Close"]].dropna()
            if d.empty: raise RuntimeError("Yahoo n'a renvoyé aucune donnée 15 min.")
            d.index = pd.to_datetime(d.index, utc=True).tz_convert(TZ)
            return d[~d.index.duplicated()].sort_index()
        except Exception as e:
            err = e; time.sleep(5)
    raise err

def closed_only(df):
    if pd.Timestamp.now(tz=TZ) < df.index[-1] + pd.Timedelta(minutes=15): return df.iloc[:-1]
    return df

def wilder(s, n): return s.ewm(alpha=1/n, adjust=False).mean()

def true_range(d):
    pc = d["Close"].shift()
    return pd.concat([d["High"]-d["Low"], (d["High"]-pc).abs(), (d["Low"]-pc).abs()], axis=1).max(axis=1)

def supertrend(d, period, mult):
    atr = wilder(true_range(d), period); hl2 = (d["High"] + d["Low"]) / 2
    bub, blb = (hl2 + mult*atr).values, (hl2 - mult*atr).values
    c = d["Close"].values; n = len(d); fub, flb = bub.copy(), blb.copy()
    dirn = np.ones(n, dtype=int); st = np.full(n, np.nan)
    for i in range(1, n):
        fub[i] = bub[i] if (bub[i] < fub[i-1] or c[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = blb[i] if (blb[i] > flb[i-1] or c[i-1] < flb[i-1]) else flb[i-1]
        dirn[i] = (-1 if c[i] < flb[i] else 1) if dirn[i-1] == 1 else (1 if c[i] > fub[i] else -1)
        st[i] = flb[i] if dirn[i] == 1 else fub[i]
    return pd.Series(st, d.index), pd.Series(dirn, d.index)

def adx_calc(d, n):
    up, dn = d["High"].diff(), -d["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), d.index)
    atr = wilder(true_range(d), n)
    pdi, mdi = 100*wilder(pdm, n)/atr, 100*wilder(mdm, n)/atr
    return wilder(100*(pdi-mdi).abs()/(pdi+mdi), n)

def compute(df):
    px = df["Close"]
    MA = px.rolling(MA_LEN).mean(); ATR = wilder(true_range(df), ATR_LEN); ADX = adx_calc(df, ADX_LEN)
    ST, DIR = supertrend(df, ST_PERIOD, ST_MULT)
    SLOPE = (MA - MA.shift(SLOPE_N)) / ATR
    ATR_SLOPE = (ATR - ATR.shift(ATR_SLOPE_N)) / ATR.shift(ATR_SLOPE_N)
    REGIME = pd.Series(np.where(ATR_SLOPE <= ATR_THR, -1, np.where(ATR_SLOPE > 0, 1, 0)), df.index)
    MOM = pd.Series(np.where(SLOPE >= SLOPE_THR, 1, np.where(SLOPE <= -SLOPE_THR, -1, 0)), df.index)
    cu = (ST.shift() <= MA.shift()) & (ST > MA); cd = (ST.shift() >= MA.shift()) & (ST < MA)
    ext = (px - MA) / ATR; dpx = px.diff()
    rsi = 100 - 100 / (1 + wilder(dpx.clip(lower=0), RSI_LEN) / wilder((-dpx).clip(lower=0), RSI_LEN))
    adx_fade = (ADX.rolling(20).max() - ADX) > ADX_DROP
    ok_long = (ext < EXT_MAX) & (rsi < RSI_OB) & ~adx_fade
    ok_short = (ext > -EXT_MAX) & (rsi > RSI_OS) & ~adx_fade
    dow = pd.Series(df.index.dayofweek, df.index); hm = pd.Series(df.index.hour * 60 + df.index.minute, df.index)
    ok_t = (hm >= 8 * 60) & (hm <= NO_ENTRY_AFTER[0] * 60 + NO_ENTRY_AFTER[1])
    mon = ATR_SLOPE >= ATR_SLOPE.shift()
    lc = (px > MA) & (DIR == 1) & (SLOPE > SLOPE_THR) & mon & (ADX > ADX_MIN) & dow.isin(LONG_DAYS) & ok_t & ok_long
    sc = (px < MA) & (DIR == -1) & (SLOPE < -SLOPE_THR) & mon & (ADX > ADX_MIN) & dow.isin(SHORT_DAYS) & ok_t & ok_short
    return dict(MA=MA, ATR=ATR, ADX=ADX, MOM=MOM, cu=cu, cd=cd, REGIME=REGIME,
                newL=lc & ~lc.shift(fill_value=False), newS=sc & ~sc.shift(fill_value=False))

# ============================================================
# TP/SL, SIMULATION, PROBA
# ============================================================
def tpsl(atr, adx_v):
    if adx_v < 20:   s, t, p = 1.0, 1.5, "PRUDENT"
    elif adx_v < 25: s, t, p = 1.0, 2.0, "STANDARD"
    elif adx_v < 35: s, t, p = 1.0, 3.0, "TENDANCE FORTE"
    else:            s, t, p = 1.2, 4.0, "ACCÉLÉRATION"
    return atr * s, atr * t, p

def simulate(d, i, sens, sl, tp):
    e, day, last = d["Close"].iloc[i], d.index[i].date(), i
    for j in range(i + 1, len(d)):
        if d.index[j].date() != day: break
        h, l = d["High"].iloc[j], d["Low"].iloc[j]
        if sens == 1:
            if l <= e - sl: return "SL", -sl
            if h >= e + tp: return "TP", tp
        else:
            if h >= e + sl: return "SL", -sl
            if l <= e - tp: return "TP", tp
        last = j
    pnl = (d["Close"].iloc[last] - e) * sens
    en_cours = last == len(d) - 1 and (d.index[last].hour * 60 + d.index[last].minute) < 17 * 60 + 15
    return ("EN COURS" if en_cours else "SORTIE"), pnl

def proba_row(tab, k):
    r = tab.iloc[k]
    for keys in (["Type", "bucket"], ["Type"]):
        m = (tab[keys] == r[keys]).all(axis=1) & tab["win"].notna() & (tab.index != k)
        n = int(m.sum())
        if n >= MIN_N: return (tab.loc[m, "win"].sum() + 1) / (n + 2), n
    return np.nan, 0

def build_table(d, ind):
    rows = []
    for sens, mask in ((1, ind["newL"]), (-1, ind["newS"])):
        for i in np.where(mask.values)[0]:
            atr_v, adx_v = ind["ATR"].iloc[i], ind["ADX"].iloc[i]
            sl, tp, prof = tpsl(atr_v, adx_v); res, pnl = simulate(d, i, sens, sl, tp)
            e, t = d["Close"].iloc[i], d.index[i]
            rows.append(dict(_ts=t, Type="ACHAT" if sens == 1 else "VENTE", Entrée=e, TP=e + sens * tp, SL=e - sens * sl,
                             Profil=prof, ADX=adx_v, bucket="ADX<35" if adx_v < 35 else "ADX>=35", Résultat=res, pnl=pnl,
                             win=np.nan if res == "EN COURS" else float(res == "TP" or (res == "SORTIE" and pnl > 0))))
    if not rows: return pd.DataFrame()
    tab = pd.DataFrame(rows).sort_values("_ts").reset_index(drop=True)
    pn = [proba_row(tab, k) for k in range(len(tab))]
    tab["p"], tab["n"] = [x[0] for x in pn], [x[1] for x in pn]
    return tab

def ptxt(p, n): return "n/a (échantillon faible)" if (p is None or np.isnan(p)) else f"{p:.0%} (n={n})"

def msg_signal(r, ts):
    sens = 1 if r["Type"] == "ACHAT" else -1
    ico = "🚀 ACHAT" if sens == 1 else "🔻 VENTE"
    return (f"{ico} QUALITÉ — DAX 15 min\n{JOURS[ts.dayofweek]} {ts:%d/%m %H:%M} (Paris, ouverture de la bougie)\n"
            f"Entrée {r['Entrée']:.0f} | TP {r['TP']:.0f} | SL {r['SL']:.0f}\n"
            f"Profil {r['Profil']} (ADX {r['ADX']:.0f})\n"
            f"Proba de réussite : {ptxt(r['p'], r['n'])} | {bias_txt(sens)}\n{ctx_ligne()}")

# ============================================================
# BOUCLE
# ============================================================
def main():
    maj_contexte(force=True)
    telegram(f"✅ Signaux Supertrend/MM200 démarrés {pd.Timestamp.now(tz=TZ):%d/%m %H:%M} (Paris)")
    t0, last_candle, pos = time.time(), None, 0
    while True:
        try:
            now = pd.Timestamp.now(tz=TZ)
            if now.weekday() >= 5 or (now.hour, now.minute) >= END_LOOP:
                print("Marché fermé : arrêt."); break
            if time.time() - t0 > LOOP_MAX_S:
                telegram("⏹ Fin de session GitHub (limite 6 h). Relancer le workflow si besoin."); break
            d = closed_only(fetch_data())
            if len(d) < MA_LEN + SLOPE_N + 2:
                time.sleep(POLL_SEC); continue
            ts = d.index[-1]; i = len(d) - 1
            if last_candle is None or ts > last_candle:
                first = last_candle is None; last_candle = ts
                maj_contexte(); ind = compute(d); tab = build_table(d, ind)
                print(f"{now:%H:%M:%S} bougie {ts:%d/%m %H:%M} | DAX {d['Close'].iloc[-1]:.0f}")
                mom, reg = ind["MOM"].iloc[i], ind["REGIME"].iloc[i]
                cu_, cd_ = bool(ind["cu"].iloc[i]), bool(ind["cd"].iloc[i])
                if pos == 1 and (mom != 1 or reg == -1 or cd_):
                    telegram(f"⚠️ SORTIE LONG ({ts:%H:%M}) : momentum/régime ATR dégradé ou croisement baissier. Sortir ou resserrer le stop."); pos = 0
                elif pos == -1 and (mom != -1 or reg == -1 or cu_):
                    telegram(f"⚠️ SORTIE SHORT ({ts:%H:%M}) : momentum/régime ATR dégradé ou croisement haussier. Sortir ou resserrer le stop."); pos = 0
                r = tab[tab["_ts"] == ts] if not tab.empty else tab
                if len(r):
                    age_min = (now - (ts + pd.Timedelta(minutes=15))).total_seconds() / 60
                    if not first or age_min <= AGE_MAX_MIN:
                        r = r.iloc[-1]; pos = 1 if r["Type"] == "ACHAT" else -1
                        telegram(msg_signal(r, ts))
            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Erreur (nouvel essai dans 60 s) :", e); time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
