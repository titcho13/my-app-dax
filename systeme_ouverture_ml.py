#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
#  DAX — SESSION D'OUVERTURE : filtre V3.1 + ML jour de semaine + régime de marché + signal 15 min
#  Version HEADLESS pour GitHub Actions (alertes Telegram).
#  Secrets GitHub : TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
#  Régime de marché « app.py » : si un dossier moteurs/regime.py est à la racine du dépôt, il est utilisé ; sinon régime approché MM200.
# ============================================================
import os, sys, time, warnings
import numpy as np, pandas as pd, requests, yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# ================= PARAMÈTRES =================
TICKER, TZ = "^GDAXI", "Europe/Paris"
SESSION_DEBUT, SESSION_FIN = (9, 0), (12, 0)
POLL_SEC = 60
REFRESH_FILTRES_S = 1800
STOP, TP = 40, 60
COST_PTS = 1.5
PROBA_MIN, AUC_MIN = 0.55, 0.54
SENS_ACTIFS = ("short",)                         # ("short", "long") pour envoyer aussi les longs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOTEURS_DIR = BASE_DIR                           # dossier qui contient moteurs/regime.py
REGIME_DEBUT = "2015-01-01"
JN = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

# ============================================================
# OUTILS
# ============================================================
def telegram(msg):
    tok, chat = os.environ.get("TELEGRAM_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
    print("[TELEGRAM]", msg)
    if not (tok and chat): print("⚠ TELEGRAM_TOKEN / TELEGRAM_CHAT_ID absents."); return
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": msg}, timeout=20)
        if not r.ok: print(f"⚠ Telegram {r.status_code} : {r.text[:200]}")
    except Exception as e: print("⚠ Telegram :", e)

def maintenant(): return pd.Timestamp.now(tz=TZ)

def dl(interval, period):
    err = None
    for _ in range(3):
        try:
            d = yf.download(TICKER, period=period, interval=interval, auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            d = d[["Open", "High", "Low", "Close"]].dropna()
            if d.empty: raise RuntimeError(f"Yahoo : aucune donnée {interval}.")
            idx = pd.to_datetime(d.index)
            if interval == "1d": d.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
            else: d.index = idx.tz_localize("UTC").tz_convert(TZ) if idx.tz is None else idx.tz_convert(TZ)
            return d[~d.index.duplicated()].sort_index()
        except Exception as e:
            err = e; time.sleep(5)
    raise err

def wilder(s, n): return s.ewm(alpha=1 / n, adjust=False).mean()

def true_range(d):
    pc = d["Close"].shift()
    return pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)

def adx_wilder(d, n=14):
    up, dn = d["High"].diff(), -d["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), d.index)
    atr = wilder(true_range(d), n)
    pdi, mdi = 100 * wilder(pdm, n) / atr, 100 * wilder(mdm, n) / atr
    return wilder(100 * (pdi - mdi).abs() / (pdi + mdi), n)

# ============================================================
# COUCHE 1 — FILTRE V3.1 + RÉGIME DE MARCHÉ
# ============================================================
def regime_marche(ratio):
    def f(x):
        try: return float(x)
        except Exception: return np.nan
    try:
        if MOTEURS_DIR and MOTEURS_DIR not in sys.path: sys.path.insert(0, MOTEURS_DIR)
        from moteurs import regime
        L = regime.executer(TICKER, REGIME_DEBUT)["lecture"]
        return dict(source="app (moteurs.regime)", regime=str(L["regime"]), confiance=f(L.get("confiance")))
    except Exception as e:
        reg = "Bull" if ratio > 1.02 else "Bear" if ratio < 0.98 else "Neutre"
        return dict(source=f"approx. MM200 ({type(e).__name__})", regime=reg, confiance=np.nan)

def calculer_filtres():
    daily, h1 = dl("1d", "2y"), dl("1h", "730d")
    h4 = h1.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    adx_d, adx_h4, adx_h1 = (float(adx_wilder(x).iloc[-1]) for x in (daily, h4, h1))
    cours = float(daily["Close"].iloc[-1]); mm200 = float(daily["Close"].rolling(200).mean().iloc[-1]); ratio = cours / mm200
    trend = "BULL" if ratio > 1.02 else "BEAR" if ratio < 0.98 else "NEUTRE"
    feu = "🟢 V3.1 ACTIF" if (adx_d > 25 and adx_h4 > 25) else "🟠 V3.1 PRUDENT" if adx_d > 20 else "🔴 PAS DE TENDANCE"
    return dict(cours=cours, ratio=ratio, trend=trend, adx_d=adx_d, adx_h4=adx_h4, adx_h1=adx_h1, feu=feu,
                **{"reg_" + k: v for k, v in regime_marche(ratio).items()})

# ============================================================
# COUCHE 2 — ML jour de semaine
# ============================================================
BASE = ["rsi", "adx_d", "dist_ma200", "dist_ma50", "atr_pct", "ret1", "ret5", "vol20"]
FEATS = BASE + ["gap_atr"] + [f"dow_{k}" for k in range(5)]

def preparer_ml():
    d = dl("1d", "10y"); pc = d["Close"].shift()
    d["atr"] = true_range(d).rolling(14).mean(); d["atr_pct"] = d["atr"] / d["Close"] * 100
    d["adx_d"] = adx_wilder(d)
    dd = d["Close"].diff(); g, l = wilder(dd.clip(lower=0), 14), wilder((-dd).clip(lower=0), 14)
    d["rsi"] = 100 - 100 / (1 + g / l)
    d["dist_ma200"] = d["Close"] / d["Close"].rolling(200).mean() - 1
    d["dist_ma50"] = d["Close"] / d["Close"].rolling(50).mean() - 1
    d["ret1"], d["ret5"] = d["Close"].pct_change() * 100, d["Close"].pct_change(5) * 100
    d["vol20"] = np.log(d["Close"] / pc).rolling(20).std() * 100
    F = d[BASE].shift(1)
    F["gap_atr"] = (d["Open"] - d["Close"].shift()) / d["atr"].shift()
    for k in range(5): F[f"dow_{k}"] = (d.index.dayofweek == k).astype(int)
    F["dow"] = d.index.dayofweek
    h1 = dl("1h", "730d")
    s = h1[(h1.index.hour >= SESSION_DEBUT[0]) & (h1.index.hour < SESSION_FIN[0])]
    gb = s.groupby(s.index.date)
    sess = pd.DataFrame({"open": gb["Open"].first(), "close": gb["Close"].last(), "n": gb["Close"].size()})
    sess = sess[sess["n"] == SESSION_FIN[0] - SESSION_DEBUT[0]]; sess.index = pd.to_datetime(sess.index)
    sess["short"] = sess["open"] - sess["close"]; sess["long"] = -sess["short"]
    return F.join(sess[["short", "long"]], how="inner").dropna(), F, d

def stats_jours(data, titre):
    rows = []
    for k in range(5):
        x = data[data["dow"] == k]
        if len(x) < 5: continue
        r = {"Jour": JN[k], "n": len(x)}
        for sens in ("short", "long"):
            m, sd = x[sens].mean(), x[sens].std(); t = m / (sd / np.sqrt(len(x))) if sd > 0 else np.nan
            r[f"{sens} moy pts"] = round(m, 1); r[f"{sens} %gagn"] = f"{(x[sens] > 0).mean():.0%}"; r[f"{sens} t"] = f"{t:+.1f}{'*' if abs(t) > 2 else ''}"
        rows.append(r)
    print(f"\n{titre}\n(* = |t| > 2 ; avec 10 tests, un « * » isolé peut être du hasard)")
    print(pd.DataFrame(rows).to_string(index=False))

def entrainer_ml(data):
    X, res = data[FEATS], {}
    mk = lambda: RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=15, random_state=0, n_jobs=-1)
    for sens in ("short", "long"):
        y = (data[sens] > COST_PTS).astype(int); oos = pd.Series(np.nan, index=data.index)
        for tr, te in TimeSeriesSplit(n_splits=5).split(X):
            m = mk().fit(X.iloc[tr], y.iloc[tr]); p = m.predict_proba(X.iloc[te])
            oos.iloc[te] = p[:, list(m.classes_).index(1)] if 1 in m.classes_ else 0.0
        ok = oos.notna(); sel = ok & (oos >= PROBA_MIN)
        auc = roc_auc_score(y[ok], oos[ok]) if y[ok].nunique() > 1 else np.nan
        gain_sel = data.loc[sel, sens].mean() - COST_PTS if sel.sum() else np.nan
        fiable = bool(auc >= AUC_MIN and sel.sum() >= 20 and gain_sel > 0)
        res[sens] = dict(model=mk().fit(X, y), auc=auc, n_oos=int(ok.sum()), n_sel=int(sel.sum()), gain_sel=gain_sel,
                         base=data.loc[ok, sens].mean() - COST_PTS, fiable=fiable)
    return res

def ligne_cible(F, d):
    cible = maintenant().tz_localize(None).normalize()
    while cible.dayofweek >= 5: cible += pd.Timedelta(days=1)
    if cible in F.index: x = F.loc[[cible]].copy()
    else:
        x = pd.DataFrame([d[BASE].iloc[-1].to_dict()], index=[cible]); x["gap_atr"] = 0.0
        for k in range(5): x[f"dow_{k}"] = int(cible.dayofweek == k)
    return cible, x[FEATS].fillna(0.0)

def predire(res, x):
    out = {}
    for sens, r in res.items():
        m = r["model"]; p = m.predict_proba(x)[0]; out[sens] = float(p[list(m.classes_).index(1)]) if 1 in m.classes_ else 0.0
    return out

def rapport_ml(data, d, res):
    stats_jours(data, f"SESSION D'OUVERTURE {SESSION_DEBUT[0]}h-{SESSION_FIN[0]}h — {len(data)} séances — gain brut (pts)")
    dj = pd.DataFrame({"dow": d.index.dayofweek, "short": d["Open"] - d["Close"]}); dj["long"] = -dj["short"]
    stats_jours(dj.dropna(), f"JOURNÉE ENTIÈRE — {len(dj)} séances (~10 ans)")
    rows = []
    for sens, r in res.items():
        rows.append({"Sens": sens.upper(), "AUC": round(r["auc"], 3), "n oos": r["n_oos"], f"n P>={PROBA_MIN:.0%}": r["n_sel"],
                     "Gain net filtré": round(r["gain_sel"], 1) if r["n_sel"] else np.nan, "Gain net base": round(r["base"], 1),
                     "Fiable": "OUI" if r["fiable"] else "NON -> ignoré"})
    print("\nML (validation glissante hors échantillon)"); print(pd.DataFrame(rows).to_string(index=False))

# ============================================================
# COUCHE 3 — SIGNAL 15 min
# ============================================================
def indicateurs_15m(df):
    df = df.copy()
    df["TR"] = true_range(df); df["ATR"] = df["TR"].rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / df["Close"] * 100; df["ATR_slope"] = df["ATR_pct"] - df["ATR_pct"].shift(4)
    df["MM200"] = df["Close"].rolling(200).mean(); df["MM200_slope"] = df["MM200"] - df["MM200"].shift(8)
    up, dn = df["High"].diff(), -df["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0), df.index); mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), df.index)
    pdi, mdi = 100 * pdm.rolling(14).mean() / df["ATR"], 100 * mdm.rolling(14).mean() / df["ATR"]
    df["ADX"] = ((pdi - mdi).abs() / (pdi + mdi) * 100).rolling(14).mean()
    df["LOW_5"] = df["Low"].rolling(5).min().shift(1); df["HIGH_5"] = df["High"].rolling(5).max().shift(1)
    return df

def closed_only(df):
    return df.iloc[:-1] if maintenant() < df.index[-1] + pd.Timedelta(minutes=15) else df

def evaluer_15m(last):
    s = {"Cours<MM200": last["Close"] < last["MM200"], "PenteMM200<0": last["MM200_slope"] < 0, "ADX>25": last["ADX"] > 25,
         "PenteATR>0": last["ATR_slope"] > 0, "Cassure bas 5b": last["Close"] < last["LOW_5"]}
    l = {"Cours>MM200": last["Close"] > last["MM200"], "PenteMM200>0": last["MM200_slope"] > 0, "ADX>25": last["ADX"] > 25,
         "PenteATR>0": last["ATR_slope"] > 0, "Cassure haut 5b": last["Close"] > last["HIGH_5"]}
    score = lambda c: int(last["ADX"] > 25) + int(last["ADX"] > 30) + int(last["ATR_slope"] > 0) + int(last["ATR_slope"] > 0.03) + int(list(c.values())[1]) + int(list(c.values())[4])
    return {"short": dict(cond=s, ok=all(s.values()), score=score(s)), "long": dict(cond=l, ok=all(l.values()), score=score(l))}

# ============================================================
# DÉCISION
# ============================================================
def autorisations(E, ML):
    out = {}
    for sens in ("short", "long"):
        ok, raisons = True, []
        if E["adx_d"] <= 20: ok = False; raisons.append("ADX jour <= 20")
        contre = "BULL" if sens == "short" else "BEAR"
        if E["trend"] == contre: ok = False; raisons.append(f"tendance MM200 {contre}")
        if E["reg_regime"].lower().startswith(contre.lower()): ok = False; raisons.append(f"régime {E['reg_regime']}")
        if ML["fiable"][sens] and ML["p"][sens] < PROBA_MIN: ok = False; raisons.append(f"ML P={ML['p'][sens]:.0%}<{PROBA_MIN:.0%}")
        out[sens] = (ok, raisons)
    return out

def log_etat(E, ML, ev, aut, ts, close):
    print(f"\n{maintenant():%a %d/%m %H:%M:%S} | bougie 15 min {ts:%H:%M} | DAX {close:.0f} | {E['feu']} | ratio MM200 {E['ratio']:.3f} {E['trend']} | régime {E['reg_regime']}")
    for sens in ("short", "long"):
        ok, raisons = aut[sens]
        print(f"  {sens.upper():5} {'autorisé' if ok else 'bloqué (' + ', '.join(raisons) + ')'} | " + " ".join(f"{'+' if v else '-'}{k}" for k, v in ev[sens]["cond"].items()) + f" -> {'SIGNAL' if ev[sens]['ok'] else 'rien'}")

def message(sens, last, ts, E, ML, score):
    e = last["Close"]; s = 1 if sens == "long" else -1
    return (f"{'🟢' if s == 1 else '🔴'} SIGNAL V3.1 + ML — {sens.upper()} DAX\n{ts:%d/%m/%Y %H:%M} (Paris)\n"
            f"Entrée {e:.1f} | STOP {e - s * STOP:.1f} | TP {e + s * TP:.1f}\n"
            f"ADX {last['ADX']:.1f} | Pente ATR {last['ATR_slope']:.4f} | Pente MM200 {last['MM200_slope']:.2f} | Score {score}/6\n"
            f"{E['feu']} | Régime {E['reg_regime']} | ML {JN[ML['cible'].dayofweek]} P({sens})={ML['p'][sens]:.0%}")

def resume_matin(E, ML):
    ml = " | ".join(f"P({s})={ML['p'][s]:.0%} (AUC {ML['res'][s]['auc']:.2f}, {'utilisé' if ML['fiable'][s] else 'ignoré'})" for s in ("short", "long"))
    conf = "" if np.isnan(E["reg_confiance"]) else f" confiance {E['reg_confiance']:.0f}/100"
    return (f"✅ Système ouverture DAX — {JN[ML['cible'].dayofweek]} {ML['cible']:%d/%m}\n{E['feu']} | ADX jour {E['adx_d']:.1f} | 4H {E['adx_h4']:.1f} | "
            f"Cours/MM200 {E['ratio']:.3f} ({E['trend']})\nRégime {E['reg_regime']}{conf} [{E['reg_source']}]\nML : {ml}\nSens actifs : {', '.join(s.upper() for s in SENS_ACTIFS)}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("Chargement des données et entraînement du ML…")
    data, F, dj = preparer_ml(); res = entrainer_ml(data); cible, xj = ligne_cible(F, dj)
    rapport_ml(data, dj, res)
    ML = dict(res=res, cible=cible, p=predire(res, xj), fiable={k: v["fiable"] for k, v in res.items()})
    print(f"\nJour cible {JN[cible.dayofweek]} {cible:%d/%m/%Y} | P(short) {ML['p']['short']:.0%} | P(long) {ML['p']['long']:.0%}")

    n0 = maintenant(); debut = n0.normalize() + pd.Timedelta(hours=SESSION_DEBUT[0], minutes=SESSION_DEBUT[1])
    fin = n0.normalize() + pd.Timedelta(hours=SESSION_FIN[0], minutes=SESSION_FIN[1])
    semaine = n0.weekday() < 5
    if semaine and n0 < fin:
        while maintenant() < debut: time.sleep(20)
    en_session = semaine and maintenant() < fin
    E, t_f, derniere, deja = calculer_filtres(), time.time(), None, set()
    if en_session: telegram(resume_matin(E, ML))
    else: print("Hors session d'ouverture : analyse seule.")

    while True:
        try:
            if time.time() - t_f > REFRESH_FILTRES_S: E, t_f = calculer_filtres(), time.time()
            df = closed_only(indicateurs_15m(dl("15m", "60d"))); ts = df.index[-1]; last = df.iloc[-1]
            if derniere is None or ts > derniere:
                derniere = ts; ev, aut = evaluer_15m(last), autorisations(E, ML)
                log_etat(E, ML, ev, aut, ts, last["Close"])
                if en_session and ts.date() == maintenant().date():
                    for sens in SENS_ACTIFS:
                        if ev[sens]["ok"] and aut[sens][0] and (sens, ts) not in deja:
                            deja.add((sens, ts)); telegram(message(sens, last, ts, E, ML, ev[sens]["score"]))
            if not en_session: break
            if maintenant() >= fin: print("\nFin de la session d'ouverture : arrêt."); break
            time.sleep(POLL_SEC)
        except KeyboardInterrupt: break
        except Exception as e:
            print("Erreur (nouvel essai dans 60 s) :", e); time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
