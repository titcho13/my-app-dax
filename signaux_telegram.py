# -*- coding: utf-8 -*-
# ============================================================
#  signaux_telegram.py — DAX 15 min : Supertrend/MM200 + momentum + pente ATR
#  + filtre jours (longs lun-mer / shorts mer-ven) + filtre anti-épuisement
#  -> alertes Telegram (signal avec proba, sorties, contexte du matin, récap de fin de séance)
#  Variables d'environnement : TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (obligatoires)
#                              MIN_PROBA (0 = tout envoyer, 0.6 = proba >= 60 %), MAX_RUN_MIN, SEND_CONTEXT
# ============================================================
import os, io, time, warnings, contextlib
import numpy as np, pandas as pd, requests, yfinance as yf
import xml.etree.ElementTree as ET
from urllib.parse import quote
warnings.filterwarnings("ignore")

# ================= CONFIG =================
TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
MIN_PROBA = float(os.getenv("MIN_PROBA", "0"))
MAX_RUN_MIN = int(os.getenv("MAX_RUN_MIN", "345"))       # durée max d'un job (GitHub : 360 min)
SEND_CONTEXT = os.getenv("SEND_CONTEXT", "1") == "1"     # envoie le contexte journalier si démarrage avant 10h30

TZ = "Europe/Paris"
ST_PERIOD, ST_MULT = 14, 3.0
MA_LEN, ATR_LEN, ADX_LEN = 200, 14, 14
ADX_MIN = 25
SLOPE_N, SLOPE_THR = 20, 0.20
ATR_SLOPE_N, ATR_THR = 20, -0.10
POLL_SEC = 60
LONG_DAYS = {0, 1, 2}             # lundi=0 ... vendredi=4
SHORT_DAYS = {2, 3, 4}
JOURS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
NO_ENTRY_AFTER = (16, 45)
END_LOOP = (17, 35)
MIN_N = 8
CTX_REFRESH_SEC = 3600
# filtre anti-épuisement (désactiver : EXT_MAX=99 ; RSI_OB, RSI_OS=101, -1 ; ADX_DROP=99)
EXT_MAX = 4.0
RSI_LEN, RSI_OB, RSI_OS = 14, 70, 30
ADX_DROP = 5.0

# ================= TELEGRAM =================
def tg(text):
    print(text, flush=True)
    if not (TOKEN and CHAT):
        print("(TELEGRAM_TOKEN / TELEGRAM_CHAT_ID absents : affichage console uniquement)", flush=True); return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          data={"chat_id": CHAT, "text": text[:4000], "disable_web_page_preview": True}, timeout=20)
        if not r.ok: print("Telegram erreur :", r.status_code, r.text[:200], flush=True)
    except Exception as e:
        print("Telegram erreur :", e, flush=True)

def tg_long(text):
    buf = ""
    for line in text.splitlines():
        if len(buf) + len(line) + 1 > 3800: tg(buf); buf = ""
        buf += line + "\n"
    if buf.strip(): tg(buf)

# ============================================================
# 1) CONTEXTE JOURNALIER
# ============================================================
CFG = {
    "symbole": "^GDAXI", "nom": "DAX", "fuseau": TZ, "periode": "3y",
    "ema_court": 20, "ema_rapide": 50, "ema_lente": 200,
    "er_fenetre": 20, "er_tendance": 0.30, "er_range": 0.15,
    "atr_fenetre": 14, "swing": 5,
    "devises": ["EUR", "USD"], "impacts": ["High", "Medium"],
    "session": (9.0, 17.5), "nb_news": 8,
    "requete_news": "DAX Bourse de Francfort",
    "globaux": {"S&P 500 futures": "ES=F", "VIX": "^VIX", "Nikkei 225": "^N225", "EUR/USD": "EURUSD=X"},
    "mots_cles": ["bce", "ecb", "lagarde", "fed", "powell", "fomc", "inflation", "cpi", "pib", "gdp", "nfp", "emploi",
                  "chômage", "taux", "zew", "ifo", "pmi", "tarif", "droits de douane", "tariff", "guerre", "ukraine",
                  "iran", "israël", "élection", "shutdown", "récession", "recession", "merz", "budget", "faillite",
                  "émeute", "tension", "manifestation", "grève", "attentat", "crise", "gouvernement", "censure",
                  "dissolution", "notation", "moody", "fitch", "s&p global", "oat", "spread", "krach", "chute", "plonge"],
}
UA = {"User-Agent": "Mozilla/5.0"}

def charger_journalier(c):
    df = yf.download(c["symbole"], period=c["periode"], interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty: raise RuntimeError("Aucune donnée journalière reçue de Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def retirer_barre_du_jour(df, c):
    now = pd.Timestamp.now(tz=c["fuseau"]); partielle = False
    if df.index[-1].date() == now.date() and (now.hour + now.minute / 60) < c["session"][1] + 0.3:
        df = df.iloc[:-1]; partielle = True
    return df, partielle

def pivots(d, n):
    hi, lo = d["high"].values, d["low"].values; ph, pl = [], []
    for i in range(n, len(d) - n):
        if hi[i] == hi[i - n:i + n + 1].max(): ph.append((d.index[i], hi[i]))
        if lo[i] == lo[i - n:i + n + 1].min(): pl.append((d.index[i], lo[i]))
    return ph, pl

def analyser_regime(df, c):
    d = df.copy(); cl = d["close"]
    for k in (c["ema_court"], c["ema_rapide"], c["ema_lente"]): d[f"ema{k}"] = cl.ewm(span=k, adjust=False).mean()
    n = c["er_fenetre"]; d["er"] = (cl - cl.shift(n)).abs() / cl.diff().abs().rolling(n).sum()
    pc = cl.shift(1); tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(c["atr_fenetre"]).mean(); d["atr_pct"] = d["atr"] / cl
    d["vol20"] = np.log(cl / pc).rolling(20).std() * np.sqrt(252)
    last = d.iloc[-1]
    rang_vol = (d["vol20"].iloc[-252:] <= last["vol20"]).mean() * 100
    ph, pl = pivots(d, c["swing"])
    if len(ph) >= 2 and len(pl) >= 2:
        hh, hl = ph[-1][1] > ph[-2][1], pl[-1][1] > pl[-2][1]
        structure = "HH + HL (haussière)" if hh and hl else "LH + LL (baissière)" if (not hh and not hl) else "mixte (sommets/creux contradictoires)"
        s_struct = 1 if hh and hl else -1 if (not hh and not hl) else 0
    else: structure, s_struct = "indéterminée", 0
    pente = d["ema50"].iloc[-1] / d["ema50"].iloc[-11] - 1
    criteres = [("Clôture > EMA200", 1 if last["close"] > last["ema200"] else -1),
                ("EMA50 > EMA200", 1 if last["ema50"] > last["ema200"] else -1),
                ("Clôture > EMA50", 1 if last["close"] > last["ema50"] else -1),
                ("Pente EMA50 sur 10 j", 1 if pente > 0 else -1),
                ("Rendement 20 j", 1 if cl.iloc[-1] > cl.iloc[-21] else -1),
                (f"Structure : {structure}", s_struct)]
    score = sum(v for _, v in criteres)
    biais = "HAUSSIER" if score >= 3 else "BAISSIER" if score <= -3 else "NEUTRE / MIXTE"
    er = last["er"]
    mode = "TENDANCE" if er >= c["er_tendance"] else "RANGE" if er <= c["er_range"] else "TRANSITION"
    vol_lbl = "élevée" if rang_vol >= 80 else "faible" if rang_vol <= 20 else "normale"
    h252 = d["high"].iloc[-252:].max()
    return dict(d=d, last=last, score=score, criteres=criteres, biais=biais, mode=mode, er=er, structure=structure,
                vol_lbl=vol_lbl, rang_vol=rang_vol, dd=cl.iloc[-1] / h252 - 1, h252=h252,
                perf={k: cl.iloc[-1] / cl.iloc[-k - 1] - 1 for k in (5, 20, 60)},
                niveaux=dict(veille_h=d["high"].iloc[-1], veille_b=d["low"].iloc[-1], veille_c=cl.iloc[-1],
                             h20=d["high"].iloc[-20:].max(), b20=d["low"].iloc[-20:].min()))

def contexte_global(c):
    out = {}
    for nom, sym in c["globaux"].items():
        try:
            x = yf.download(sym, period="7d", interval="1d", auto_adjust=False, progress=False, threads=False)
            if isinstance(x.columns, pd.MultiIndex): x.columns = x.columns.get_level_values(0)
            cl = x["Close"].dropna(); out[nom] = (float(cl.iloc[-1]), float(cl.iloc[-1] / cl.iloc[-2] - 1))
        except Exception: out[nom] = None
    return out

def calendrier_jour(c, now):
    r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", headers=UA, timeout=10)
    r.raise_for_status(); rows = []
    for e in r.json():
        try:
            dt_ = pd.Timestamp(e["date"])
            dt_ = dt_.tz_localize("America/New_York") if dt_.tzinfo is None else dt_
            dt_ = dt_.tz_convert(c["fuseau"])
        except Exception: continue
        if dt_.date() != now.date() or e.get("country") not in c["devises"] or e.get("impact") not in c["impacts"]: continue
        rows.append(dict(heure=dt_, devise=e["country"], evenement=e["title"], impact=e["impact"],
                         prevision=e.get("forecast", ""), precedent=e.get("previous", "")))
    return pd.DataFrame(rows).sort_values("heure").reset_index(drop=True) if rows else pd.DataFrame()

def _important(titre, c): return any(m in titre.lower() for m in c["mots_cles"])

def news_google(c, n):
    url = f"https://news.google.com/rss/search?q={quote(c['requete_news'])}+when:1d&hl=fr&gl=FR&ceid=FR:fr"
    r = requests.get(url, headers=UA, timeout=10); r.raise_for_status(); out = []
    for it in ET.fromstring(r.content).iter("item"):
        out.append(dict(titre=(it.findtext("title") or "").strip(), source=(it.findtext("source") or "").strip(),
                        date=pd.to_datetime(it.findtext("pubDate"), utc=True, errors="coerce")))
        if len(out) >= n * 2: break
    return out

def news_yahoo(c, n):
    out = []
    for it in (yf.Ticker(c["symbole"]).news or [])[: n * 2]:
        k = it.get("content", it)
        titre = k.get("title") or it.get("title") or ""
        pub = k.get("pubDate") or it.get("providerPublishTime")
        date = pd.to_datetime(pub, unit="s", utc=True, errors="coerce") if isinstance(pub, (int, float)) else pd.to_datetime(pub, utc=True, errors="coerce")
        out.append(dict(titre=titre.strip(), source=((k.get("provider") or {}).get("displayName") or it.get("publisher") or ""), date=date))
    return out

def collecter_news(c):
    items = []
    for f in (news_google, news_yahoo):
        try: items += f(c, c["nb_news"])
        except Exception as e: print(f"  (source news indisponible : {f.__name__} — {e})")
    vus, uniques = set(), []
    for it in sorted(items, key=lambda x: x["date"] if pd.notna(x["date"]) else pd.Timestamp(0, tz="UTC"), reverse=True):
        cle = it["titre"].lower()[:60]
        if it["titre"] and cle not in vus: vus.add(cle); uniques.append(it)
    return uniques

def rapport(c=CFG):
    now = pd.Timestamp.now(tz=c["fuseau"])
    df = charger_journalier(c); df, partielle = retirer_barre_du_jour(df, c); R = analyser_regime(df, c); L = R["last"]
    print("=" * 40); print(f"CONTEXTE {c['nom']} — {now:%A %d/%m/%Y %H:%M}"); print("=" * 40)
    print(f"BIAIS JOURNALIER : {R['biais']} (score {R['score']:+d}/6)\nRÉGIME : {R['mode']} (ER {R['er']:.2f}) | volatilité {R['vol_lbl']}")
    print(f"Dernière clôture : {df.index[-1]:%d/%m/%Y} à {L['close']:,.0f}" + (" (barre du jour exclue)" if partielle else ""))
    print("-" * 40); print("RÉGIME DE MARCHÉ")
    for nom, v in R["criteres"]: print(f"  {'▲' if v > 0 else '▼' if v < 0 else '•'} {nom}")
    print(f"  EMA20 {L['ema20']:,.0f} | EMA50 {L['ema50']:,.0f} | EMA200 {L['ema200']:,.0f}")
    print(f"  Perf 5 j {R['perf'][5]:+.1%} | 20 j {R['perf'][20]:+.1%} | 60 j {R['perf'][60]:+.1%}")
    print(f"  ATR14 {L['atr']:.0f} pts | vol 20 j {L['vol20']:.1%} (percentile 1 an : {R['rang_vol']:.0f})")
    N = R["niveaux"]; print(f"  Veille : haut {N['veille_h']:,.0f} / bas {N['veille_b']:,.0f} | 20 j : {N['h20']:,.0f} / {N['b20']:,.0f}")
    print("-" * 40); print("CONTEXTE GLOBAL")
    for nom, v in contexte_global(c).items(): print(f"  {nom:<16} " + (f"{v[0]:>10,.2f}  {v[1]:+.2%}" if v else "indisponible"))
    print("-" * 40); print("CALENDRIER ÉCONOMIQUE (EUR / USD)")
    verdict = "CALME (aucun événement majeur)"
    try:
        cal = calendrier_jour(c, now)
        if cal.empty: print("  Aucun événement à impact fort ou moyen aujourd'hui.")
        else:
            for _, e in cal.iterrows():
                tag = "⚠ " if e["impact"] == "High" else "  "
                print(f"  {tag}{e['heure']:%H:%M} {e['devise']} {e['evenement']} [{e['impact']}] prév {e['prevision'] or '-'} / préc {e['precedent'] or '-'}")
            fortes = cal[(cal["impact"] == "High") & cal["heure"].apply(lambda h: c["session"][0] <= h.hour + h.minute / 60 <= c["session"][1])]
            verdict = f"JOUR À NEWS — {len(fortes)} événement(s) à impact fort" if len(fortes) else "MODÉRÉ (événements moyens seulement)"
    except Exception as e:
        print(f"  Calendrier indisponible ({e})."); verdict = "INCONNU (calendrier indisponible)"
    print("-" * 40); print("NEWS DU JOUR")
    news = collecter_news(c); nb = 0
    for it in news:
        if nb >= c["nb_news"]: break
        h = it["date"].tz_convert(c["fuseau"]).strftime("%d/%m %H:%M") if pd.notna(it["date"]) else "?"
        print(f"  {'⚠ ' if _important(it['titre'], c) else '  '}[{h}] {it['titre'][:100]} ({it['source']})"); nb += 1
    if not news: print("  Aucune news récupérée.")
    imp = sum(_important(i["titre"], c) for i in news[: c["nb_news"]])
    if imp and verdict.startswith("CALME"): verdict = "CALME côté calendrier, mais NEWS SENSIBLES"
    print("=" * 40); print(f"SYNTHÈSE : {R['biais']} / {R['mode']} / vol {R['vol_lbl']} | {verdict} | news sensibles : {imp}"); print("=" * 40)
    return R

CTX_TXT, CTX_R, CTX_T = "", None, 0.0
def maj_contexte(force=False):
    global CTX_TXT, CTX_R, CTX_T
    if not force and time.time() - CTX_T < CTX_REFRESH_SEC: return
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): CTX_R = rapport()
    except Exception as e:
        buf.write(f"Contexte indisponible : {e}")
    CTX_TXT = buf.getvalue(); CTX_T = time.time()

def bias_txt(sens):
    b = CTX_R["biais"] if CTX_R else None
    if b in ("HAUSSIER", "BAISSIER"):
        return "✅ AVEC le biais journalier" if (b == "HAUSSIER") == (sens == 1) else "⚠️ CONTRE le biais journalier"
    return "Biais journalier neutre"

# ============================================================
# 2) DONNÉES + INDICATEURS 15 min
# ============================================================
def fetch_data(period="60d"):
    d = yf.download("^GDAXI", interval="15m", period=period, auto_adjust=False, progress=False)
    if d is None or d.empty: raise RuntimeError("Aucune donnée 15 min reçue de Yahoo Finance.")
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True).tz_convert(TZ)
    return d[~d.index.duplicated()].sort_index()

def closed_only(df):
    if pd.Timestamp.now(tz=TZ) < df.index[-1] + pd.Timedelta(minutes=15): return df.iloc[:-1]
    return df

def wilder(s, n): return s.ewm(alpha=1/n, adjust=False).mean()

def true_range(d):
    pc = d["Close"].shift()
    return pd.concat([d["High"]-d["Low"], (d["High"]-pc).abs(), (d["Low"]-pc).abs()], axis=1).max(axis=1)

def supertrend(d, period, mult):
    atr = wilder(true_range(d), period)
    hl2 = (d["High"] + d["Low"]) / 2
    bub, blb = (hl2 + mult*atr).values, (hl2 - mult*atr).values
    c = d["Close"].values; n = len(d)
    fub, flb = bub.copy(), blb.copy()
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
    MA = px.rolling(MA_LEN).mean()
    ATR = wilder(true_range(df), ATR_LEN)
    ADX = adx_calc(df, ADX_LEN)
    ST, DIR = supertrend(df, ST_PERIOD, ST_MULT)
    SLOPE = (MA - MA.shift(SLOPE_N)) / ATR
    ATR_SLOPE = (ATR - ATR.shift(ATR_SLOPE_N)) / ATR.shift(ATR_SLOPE_N)
    REGIME = pd.Series(np.where(ATR_SLOPE <= ATR_THR, -1, np.where(ATR_SLOPE > 0, 1, 0)), df.index)
    MOM = pd.Series(np.where(SLOPE >= SLOPE_THR, 1, np.where(SLOPE <= -SLOPE_THR, -1, 0)), df.index)
    cu = (ST.shift() <= MA.shift()) & (ST > MA)
    cd = (ST.shift() >= MA.shift()) & (ST < MA)
    # filtre anti-épuisement
    ext = (px - MA) / ATR
    dpx = px.diff()
    rsi = 100 - 100 / (1 + wilder(dpx.clip(lower=0), RSI_LEN) / wilder((-dpx).clip(lower=0), RSI_LEN))
    adx_fade = (ADX.rolling(20).max() - ADX) > ADX_DROP
    ok_long = (ext < EXT_MAX) & (rsi < RSI_OB) & ~adx_fade
    ok_short = (ext > -EXT_MAX) & (rsi > RSI_OS) & ~adx_fade
    # signaux qualité (jours + heure + anti-épuisement)
    dow = pd.Series(df.index.dayofweek, df.index)
    hm = pd.Series(df.index.hour * 60 + df.index.minute, df.index)
    ok_t = hm <= NO_ENTRY_AFTER[0] * 60 + NO_ENTRY_AFTER[1]
    mon = ATR_SLOPE >= ATR_SLOPE.shift()
    lc = (px > MA) & (DIR == 1) & (SLOPE > SLOPE_THR) & mon & (ADX > ADX_MIN) & dow.isin(LONG_DAYS) & ok_t & ok_long
    sc = (px < MA) & (DIR == -1) & (SLOPE < -SLOPE_THR) & mon & (ADX > ADX_MIN) & dow.isin(SHORT_DAYS) & ok_t & ok_short
    newL = lc & ~lc.shift(fill_value=False)
    newS = sc & ~sc.shift(fill_value=False)
    return dict(MA=MA, ATR=ATR, ADX=ADX, ST=ST, DIR=DIR, SLOPE=SLOPE, MOM=MOM, cu=cu, cd=cd,
                ATR_SLOPE=ATR_SLOPE, REGIME=REGIME, newL=newL, newS=newS, EXT=ext, RSI=rsi, ADX_FADE=adx_fade)

# ============================================================
# 3) TP/SL, SIMULATION, PROBABILITÉS
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
            sl, tp, prof = tpsl(atr_v, adx_v)
            res, pnl = simulate(d, i, sens, sl, tp)
            e, t = d["Close"].iloc[i], d.index[i]
            rows.append(dict(_ts=t, Date=t.strftime("%d/%m/%Y"), Type="ACHAT" if sens == 1 else "VENTE",
                             Entrée=e, TP=e + sens * tp, SL=e - sens * sl, Profil=prof,
                             bucket="ADX<35" if adx_v < 35 else "ADX>=35", Résultat=res, pnl=pnl,
                             win=np.nan if res == "EN COURS" else float(res == "TP" or (res == "SORTIE" and pnl > 0))))
    if not rows: return pd.DataFrame()
    tab = pd.DataFrame(rows).sort_values("_ts").reset_index(drop=True)
    pn = [proba_row(tab, k) for k in range(len(tab))]
    tab["p"], tab["n"] = [x[0] for x in pn], [x[1] for x in pn]
    return tab

def ptxt(p, n): return "n/a (échantillon faible)" if (p is None or np.isnan(p)) else f"{p:.0%} (n={n})"

# ============================================================
# 4) MESSAGES TELEGRAM
# ============================================================
def msg_signal(r, ind, ts, sens):
    sig_t = ts + pd.Timedelta(minutes=15)      # heure de clôture de la bougie = heure du signal
    return (f"{'🚀 ACHAT' if sens == 1 else '🔻 VENTE'} DAX — {JOURS[sig_t.dayofweek]} {sig_t:%d/%m %H:%M}\n"
            f"Entrée : {r['Entrée']:.0f}\nTake profit : {r['TP']:.0f}\nStop loss : {r['SL']:.0f}\n"
            f"Proba de réussite : {ptxt(r['p'], r['n'])}\nProfil : {r['Profil']}\n"
            f"{bias_txt(sens)}\n"
            f"ADX {ind['ADX'].iloc[-1]:.0f} | RSI {ind['RSI'].iloc[-1]:.0f} | écart MM200 {ind['EXT'].iloc[-1]:+.1f} ATR")

def recap(tab, now):
    msg = f"📊 Récap {JOURS[now.weekday()]} {now:%d/%m} (DAX)\n"
    today = tab[tab["Date"] == now.strftime("%d/%m/%Y")] if not tab.empty else tab
    if today.empty: msg += "Aucun signal aujourd'hui."
    else:
        for _, r in today.iterrows():
            h = (r["_ts"] + pd.Timedelta(minutes=15)).strftime("%H:%M")
            msg += f"{h} {r['Type']} {r['Entrée']:.0f} → TP {r['TP']:.0f} / SL {r['SL']:.0f} : {r['Résultat']} ({r['pnl']:+.0f} pts)\n"
        msg += f"Total : {today['pnl'].sum():+.0f} pts"
    tg(msg)

# ============================================================
# 5) BOUCLE PRINCIPALE
# ============================================================
def main():
    t0 = time.time()
    now = pd.Timestamp.now(tz=TZ)
    if now.weekday() >= 5:
        print("Week-end : rien à faire."); return
    maj_contexte(force=True)
    sens_ok = " + ".join(x for x, s in (("LONGS", LONG_DAYS), ("SHORTS", SHORT_DAYS)) if now.weekday() in s)
    tg(f"✅ Surveillance DAX démarrée — {JOURS[now.weekday()]} {now:%d/%m %H:%M}\n"
       f"Sens autorisés aujourd'hui : {sens_ok}\n"
       f"Filtres : ADX > {ADX_MIN} | anti-épuisement | proba mini {MIN_PROBA:.0%}")
    if SEND_CONTEXT and now.hour * 60 + now.minute < 10 * 60 + 30:
        tg_long(CTX_TXT)

    last_candle, pos, tab = None, 0, pd.DataFrame()
    while True:
        try:
            now = pd.Timestamp.now(tz=TZ)
            if (now.hour, now.minute) >= END_LOOP:
                if last_candle is not None: recap(tab, now)
                tg("🔕 Fin de séance DAX : surveillance arrêtée.")
                return
            if time.time() - t0 > MAX_RUN_MIN * 60:
                print("Durée max du job atteinte : relais par le job suivant."); return

            d = closed_only(fetch_data())
            if len(d) < MA_LEN + SLOPE_N + 2:
                time.sleep(POLL_SEC); continue
            ind = compute(d); ts = d.index[-1]; i = len(d) - 1

            if last_candle is None or ts > last_candle:
                first = last_candle is None
                last_candle = ts
                maj_contexte()
                tab = build_table(d, ind)
                print(f"[{now:%H:%M:%S}] bougie {ts:%d/%m %H:%M} | DAX {d['Close'].iloc[-1]:.0f} | {len(tab)} signaux sur l'historique", flush=True)

                # sorties de position virtuelle
                mom, reg = ind["MOM"].iloc[i], ind["REGIME"].iloc[i]
                cu_, cd_ = bool(ind["cu"].iloc[i]), bool(ind["cd"].iloc[i])
                if pos == 1 and (mom != 1 or reg == -1 or cd_):
                    tg(f"⚠️ SORTIE LONG DAX ({ts + pd.Timedelta(minutes=15):%H:%M}) : momentum/régime ATR dégradé ou croisement baissier. Sortir ou resserrer le stop."); pos = 0
                elif pos == -1 and (mom != -1 or reg == -1 or cu_):
                    tg(f"⚠️ SORTIE SHORT DAX ({ts + pd.Timedelta(minutes=15):%H:%M}) : momentum/régime ATR dégradé ou croisement haussier. Sortir ou resserrer le stop."); pos = 0

                # nouveau signal sur la bougie qui vient de clôturer
                r = tab[tab["_ts"] == ts] if not tab.empty else tab
                if len(r) and not first:
                    r = r.iloc[-1]; sens = 1 if r["Type"] == "ACHAT" else -1
                    if MIN_PROBA > 0 and not np.isnan(r["p"]) and r["p"] < MIN_PROBA:
                        print(f"Signal {r['Type']} ignoré : proba {r['p']:.0%} < seuil {MIN_PROBA:.0%}", flush=True)
                    else:
                        pos = sens
                        tg(msg_signal(r, ind, ts, sens))
            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("Arrêt manuel."); return
        except Exception as e:
            print("Erreur (nouvelle tentative dans 60 s) :", e, flush=True); time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
