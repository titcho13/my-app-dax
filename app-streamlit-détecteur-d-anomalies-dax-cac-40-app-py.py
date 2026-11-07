# -*- coding: utf-8 -*-
# ============================================================================
# DÉTECTEUR D'ANOMALIES DAX / CAC 40 — APPLICATION STREAMLIT
# ----------------------------------------------------------------------------
# Déploiement : GitHub + Streamlit Cloud (gratuit).
# Secrets Streamlit Cloud (app → Settings → Secrets) :
#   CODE_ACCES = "votre-code"
#   TELEGRAM_TOKEN = "..."
#   TELEGRAM_CHAT_ID = "..."
#   EMAIL_SMTP / EMAIL_PORT / EMAIL_USER / EMAIL_PASS / EMAIL_DEST (optionnel)
# Le journal SQLite est écrit dans /tmp (éphémère sur Streamlit Cloud).
# ============================================================================
import os, io, json, time, sqlite3, smtplib, contextlib, warnings
from email.message import EmailMessage
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
import streamlit as st
import requests

warnings.filterwarnings("ignore")

try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k: return a[0]
        return lambda f: f

st.set_page_config(page_title="Anomalies DAX / CAC 40", page_icon="📈", layout="wide")

# ── Secrets ────────────────────────────────────────────────────────────────
def _secret(nom, defaut=None):
    try:
        if nom in st.secrets: return st.secrets[nom]
    except Exception:
        pass
    return os.environ.get(nom, defaut)

def _config(k, defaut):
    v = _secret(k, None)
    return defaut if v in (None, "") else (str(v).lower() == "true" if isinstance(defaut, bool) else v)

TELEGRAM_ACTIF = _config("TELEGRAM_ACTIF", True)
EMAIL_ACTIF    = _config("EMAIL_ACTIF", False)
POPUP_ACTIF    = _config("POPUP_ACTIF", True)        # = toasts Streamlit
SON_ACTIF      = _config("SON_ACTIF", False)
DOSSIER_EXPORT = _config("DOSSIER_EXPORT", "/tmp/exports_dax")
ETAT_DOSSIER   = _config("ETAT_DOSSIER", "/tmp/etat_live")
DB_JOURNAL     = os.path.join(ETAT_DOSSIER, "journal_signaux.db")
ALERTE_TENDANCE_PRIORITAIRE = _config("ALERTE_TENDANCE_PRIORITAIRE", False)
INTERVALLE_REFRESH_S = int(_config("INTERVALLE_REFRESH_S", 60))

# ── Outils vectorisés (ligne i = fenêtre du PASSÉ strict x[i-w:i]) ─────────
def _fenetres_passees(x, w):
    x = np.asarray(x, float); n = len(x)
    return sliding_window_view(np.r_[np.full(w, np.nan), x], w)[:n]

def _stat_passee(x, w, minp, f):
    win = _fenetres_passees(x, w); cnt = (~np.isnan(win)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); res = np.asarray(f(win), float)
    res[cnt < minp] = np.nan
    return res

def _med_mad_passee(x, w, minp):
    win = _fenetres_passees(x, w); cnt = (~np.isnan(win)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(win, axis=1); mad = np.nanmedian(np.abs(win - med[:, None]), axis=1)
    med[cnt < minp] = np.nan; mad[cnt < minp] = np.nan
    return med, mad

def _rang_passe(x, w, minp):
    x = np.asarray(x, float); win = _fenetres_passees(x, w)
    valid = ~np.isnan(win); cnt = valid.sum(axis=1); less = ((win < x[:, None]) & valid).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"): out = less / cnt
    out[(cnt < minp) | np.isnan(x)] = np.nan
    return out

def _par_creneau(x, slot, fn):
    x = np.asarray(x, float); slot = np.asarray(slot); out = np.full(len(x), np.nan)
    for s in np.unique(slot):
        m = slot == s; out[m] = fn(x[m])
    return out

# ── Paramètres centralisés ─────────────────────────────────────────────────
PARAMS_DEFAUT = {
    "symbole": "^GDAXI", "intervalle": "5m", "debut": None, "fin": None, "fuseau": "Europe/Paris",
    "ignorer_derniere_barre_incomplete": True,
    "z_fenetre": 100, "z_seuil": 2.0, "z_extreme": 3.0, "z_base": "classique",
    "vol_fenetre": 20, "vol_ref": 100, "vol_seuil": 2.0,
    "cusum_k": 0.5, "cusum_h": 5.0, "compression_ratio": 0.7, "expansion_ratio": 1.3,
    "volume_fenetre": 50, "volume_z_seuil": 2.0, "volume_slot_fenetre": 20, "volume_rel_slot_min": 1.5,
    "q_fenetre": 250, "q_bas": 0.01, "q_haut": 0.99,
    "div_fenetre": 10, "div_prix_seuil": 1.0, "div_activite_baisse": 0.8,
    "div_vol_score": 1.0, "div_ralentissement": 0.25,
    "ema_rapide": 50, "ema_lente": 200, "er_fenetre": 20, "er_tendance": 0.30, "er_range": 0.15,
    "regime_vol_haut": 1.5, "regime_vol_bas": -1.0,
    "strategie": "A", "anomalies_strategie": ["a1", "a4"], "anomalie_memoire": 3,
    "breakout_n": 3, "atr_fenetre": 14, "stop_atr": 1.5, "target_atr": 2.0,
    "cible_sma": 20, "tenue_max": 12,
    "cout_bp": 1.0, "slippage_bp": 0.5,
    "horizons": None, "n_min_evenements": 20, "alpha": 0.10, "bootstrap": 1000,
    "split": (0.5, 0.25, 0.25), "seed": 42, "rf_annuel": 0.0,
    "exiger_validation_test": True, "capital": 10000, "risque_pct": 0.5,
    "normaliser_creneau": True, "creneau_fenetre": 40, "creneau_min": 15,
    "filtres_contexte": True, "heure_min_signal": "09:15", "heure_max_signal": "17:00",
    "annonces": [], "annonce_avant_min": 15, "annonce_apres_min": 30,
    "filtre_alignement": "declasse", "ema_tendance_h": 20,
    "calib_horizon": 3, "calib_n_min": 30, "graphiques": True,
}
FREQ = {
    "5m":  dict(max_jours=59, barres_an=102 * 252, horizons=[1, 3, 6, 12]),
    "15m": dict(max_jours=59, barres_an=34 * 252, horizons=[1, 3, 6, 12]),
    "1d":  dict(max_jours=None, barres_an=252, horizons=[1, 3, 5]),
}
NOMS = ["zscore", "volatilite", "volume", "quantile", "divergence"]
def est_intraday(p): return p["intervalle"] != "1d"
def horizons(p): return p["horizons"] or FREQ[p["intervalle"]]["horizons"]
def cout_aller_retour(p): return (p["cout_bp"] + 2 * p["slippage_bp"]) / 1e4
def creneau_actif(p): return bool(p.get("normaliser_creneau", False)) and est_intraday(p)

# ── Données Yahoo Finance ──────────────────────────────────────────────────
@st.cache_data(ttl=45, show_spinner=False)
def charger_donnees(symbole, itv, debut, fin, fuseau, ignorer_incomplete, _p):
    p = dict(_p); p.update(symbole=symbole, intervalle=itv, debut=debut, fin=fin, fuseau=fuseau,
             ignorer_derniere_barre_incomplete=ignorer_incomplete)
    if itv not in FREQ: raise ValueError(f"Intervalle non géré : {itv}")
    today = pd.Timestamp.today().normalize()
    _fin = pd.Timestamp(fin) if fin else today + pd.Timedelta(days=1)
    mj = FREQ[itv]["max_jours"]
    _debut = pd.Timestamp(debut) if debut else _fin - pd.Timedelta(days=mj or 3650)
    if mj and _debut < today - pd.Timedelta(days=mj):
        _debut = today - pd.Timedelta(days=mj)
    raw = yf.download(symbole, start=_debut, end=_fin, interval=itv,
                      auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("Aucune donnée reçue (symbole, période ou limite Yahoo ?).")
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated()].sort_index()
    if df.index.tz is None:
        if est_intraday(p): df.index = df.index.tz_localize("UTC").tz_convert(fuseau)
    else:
        df.index = df.index.tz_convert(fuseau)
    df.index.name = "timestamp"
    if ignorer_incomplete and len(df) > 1:
        now = pd.Timestamp.now(tz=fuseau) if df.index.tz is not None else pd.Timestamp.now()
        if est_intraday(p):
            if df.index[-1] + pd.Timedelta(itv.replace("m", "min")) > now: df = df.iloc[:-1]
        elif df.index[-1].normalize() == now.normalize() and now.hour < 18:
            df = df.iloc[:-1]
    vol = df["volume"].fillna(0)
    volume_ok = bool((vol > 0).mean() > 0.5 and vol.std() > 0)
    meta = dict(symbole=symbole, intervalle=itv, debut=df.index[0], fin=df.index[-1], n=len(df), volume_ok=volume_ok)
    return df, meta

def preparer(df, p):
    d = df.copy()
    d["ret"] = np.log(d["close"] / d["close"].shift(1))
    if est_intraday(p):
        jour = pd.Series(d.index.date, index=d.index)
        prem = (jour != jour.shift(1)).values
        d.loc[prem, "ret"] = np.log(d["close"] / d["open"])[prem]
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(p["atr_fenetre"]).mean()
    return d

# ── Anomalies 1 et 2 ───────────────────────────────────────────────────────
def anomalie_1_zscore(d, p):
    r, w = d["ret"], p["z_fenetre"]; rs = r.shift(1); rv = r.values
    mu = rs.rolling(w, min_periods=w).mean(); sd = rs.rolling(w, min_periods=w).std()
    z = (r - mu) / sd.replace(0, np.nan)
    med, mad = _med_mad_passee(rv, w, w)
    zr = pd.Series(0.6745 * (rv - med) / np.where(mad == 0, np.nan, mad), index=d.index)
    if creneau_actif(p):
        slot = np.asarray(d.index.strftime("%H:%M")); wc, mc = p["creneau_fenetre"], p["creneau_min"]
        mu_c = _par_creneau(rv, slot, lambda a: _stat_passee(a, wc, mc, lambda W: np.nanmean(W, axis=1)))
        sd_c = _par_creneau(rv, slot, lambda a: _stat_passee(a, wc, mc, lambda W: np.nanstd(W, axis=1, ddof=1)))
        med_c = _par_creneau(rv, slot, lambda a: _med_mad_passee(a, wc, mc)[0])
        mad_c = _par_creneau(rv, slot, lambda a: _med_mad_passee(a, wc, mc)[1])
        z_c = (rv - mu_c) / np.where(sd_c == 0, np.nan, sd_c)
        zr_c = 0.6745 * (rv - med_c) / np.where(mad_c == 0, np.nan, mad_c)
        z = pd.Series(np.where(np.isnan(z_c), z.values, z_c), index=d.index)
        zr = pd.Series(np.where(np.isnan(zr_c), zr.values, zr_c), index=d.index)
    zb = z if p["z_base"] == "classique" else zr
    out = pd.DataFrame(index=d.index)
    out["z"], out["z_robuste"] = z, zr
    out["a1_on"] = zb.abs() >= p["z_seuil"]
    out["a1_extreme"] = zb.abs() >= p["z_extreme"]
    out["a1_dir"] = np.sign(zb).where(out["a1_on"], 0).fillna(0).astype(int)
    return out

@njit(cache=True)
def _cusum(x, k, h):
    n = len(x); sp = np.zeros(n); sm = np.zeros(n); st = np.zeros(n, dtype=np.int64); a = 0.0; b = 0.0
    for i in range(n):
        xi = x[i]
        if np.isnan(xi):
            a = 0.0; b = 0.0
            continue
        a = max(0.0, a + xi - k); b = max(0.0, b - xi - k)
        sp[i] = a; sm[i] = b
        if a > h:
            st[i] = 1; a = 0.0
        elif b > h:
            st[i] = -1; b = 0.0
    return sp, sm, st

def anomalie_2_volatilite(d, p):
    r = d["ret"]; fv, fr = p["vol_fenetre"], p["vol_ref"]
    vol = r.rolling(fv).std()
    ref_m = vol.shift(1).rolling(fr).mean(); ref_s = vol.shift(1).rolling(fr).std()
    score = (vol - ref_m) / ref_s.replace(0, np.nan); ratio = vol / ref_m
    ax = r.abs(); m = ax.shift(1).rolling(fr).mean(); s = ax.shift(1).rolling(fr).std()
    x = ((ax - m) / s.replace(0, np.nan)).values.astype(float)
    sp, sm, stt = _cusum(x, float(p["cusum_k"]), float(p["cusum_h"]))
    typ = np.select([stt == 1, (score >= p["vol_seuil"]) & (stt != 1), (stt == -1) | (score <= -p["vol_seuil"])],
                    ["augmentation_persistante", "expansion_soudaine", "baisse_inhabituelle"], "aucune")
    phase = np.select([ratio <= p["compression_ratio"], ratio >= p["expansion_ratio"]],
                      ["compression", "expansion"], "neutre")
    out = pd.DataFrame(index=d.index)
    out["vol_realisee"], out["vol_score"], out["vol_ratio"] = vol, score, ratio
    out["cusum_pos"], out["cusum_neg"], out["etat_cusum"] = sp, sm, stt
    out["phase_vol"], out["anomalie_vol"] = phase, typ
    out["a2_on"] = out["anomalie_vol"] != "aucune"
    out["a2_dir"] = np.sign(r).where(out["a2_on"], 0).fillna(0).astype(int)
    return out

# ── Anomalies 3, 4, 5 ──────────────────────────────────────────────────────
def anomalie_3_volume(d, p, volume_ok):
    out = pd.DataFrame(index=d.index)
    out["vol_rel"] = np.nan; out["vol_z"] = np.nan
    out["a3_on"] = False; out["a3_dir"] = 0
    out["anomalie_volume"] = "indisponible" if not volume_ok else "aucune"
    if not volume_ok: return out
    v = d["volume"].astype(float).replace(0, np.nan); w = p["volume_fenetre"]
    m = v.shift(1).rolling(w).mean(); s = v.shift(1).rolling(w).std()
    zv = (v - m) / s.replace(0, np.nan); rel = v / m
    spike = zv >= p["volume_z_seuil"]
    if est_intraday(p):
        slot = np.asarray(d.index.strftime("%H:%M"))
        sm = pd.Series(_par_creneau(v.values, slot, lambda a: _stat_passee(a, p["volume_slot_fenetre"], 5,
                                                                           lambda W: np.nanmean(W, axis=1))), index=d.index)
        spike &= (v / sm) >= p["volume_rel_slot_min"]
    zret = d["ret"] / d["ret"].shift(1).rolling(w).std()
    typ = np.select([spike & (zret >= 1), spike & (zret <= -1), spike & (zret.abs() < 1)],
                    ["volume_avec_hausse", "volume_avec_chute", "volume_sans_progression"], "aucune")
    out["vol_rel"], out["vol_z"] = rel, zv
    out["anomalie_volume"] = typ; out["a3_on"] = spike.fillna(False).values
    out["a3_dir"] = np.sign(d["ret"]).where(out["a3_on"], 0).fillna(0).astype(int)
    return out

def anomalie_4_quantiles(d, p):
    r, w = d["ret"], p["q_fenetre"]; rs = r.shift(1); out = pd.DataFrame(index=d.index)
    for q in sorted({0.01, 0.05, 0.5, 0.95, 0.99, p["q_bas"], p["q_haut"]}):
        out[f"q{int(round(q * 100)):02d}" if abs(q * 100 - round(q * 100)) < 1e-9 else f"q{q}"] = \
            rs.rolling(w, min_periods=w).quantile(q)
    lo = rs.rolling(w, min_periods=w).quantile(p["q_bas"]); hi = rs.rolling(w, min_periods=w).quantile(p["q_haut"])
    out["q_bas_val"], out["q_haut_val"] = lo, hi
    rang = _rang_passe(r.values, w, w)
    if creneau_actif(p):
        slot = np.asarray(d.index.strftime("%H:%M"))
        rc = _par_creneau(r.values, slot, lambda a: _rang_passe(a, p["creneau_fenetre"], p["creneau_min"]))
        rang = np.where(np.isnan(rc), rang, rc)
    out["quantile_rendement"] = rang
    out["a4_on"] = (r < lo) | (r > hi)
    out["a4_dir"] = np.where(r < lo, -1, np.where(r > hi, 1, 0))
    return out

def anomalie_5_divergence(d, p, volume_ok):
    n = p["div_fenetre"]; c = d["close"]
    cr = np.log(c / c.shift(n))
    sig = d["ret"].shift(1).rolling(p["z_fenetre"]).std() * np.sqrt(n)
    zc = cr / sig.replace(0, np.nan)
    if volume_ok:
        v = d["volume"].astype(float).replace(0, np.nan)
        act = v.rolling(n).mean() / v.shift(n).rolling(p["volume_fenetre"]).mean()
        zvmax = ((v - v.shift(1).rolling(p["volume_fenetre"]).mean()) /
                 v.shift(1).rolling(p["volume_fenetre"]).std()).rolling(n).max()
    else:
        act = d["vol_ratio"]; zvmax = pd.Series(np.nan, index=d.index)
    d1 = (zc >= p["div_prix_seuil"]) & (act <= p["div_activite_baisse"])
    d2 = (zc <= -p["div_prix_seuil"]) & (d["vol_score"] >= p["div_vol_score"])
    d3 = (zvmax >= p["volume_z_seuil"]) & (zc.abs() < 0.5) if volume_ok else pd.Series(False, index=d.index)
    prior = np.log(c.shift(n) / c.shift(2 * n)); zp = prior / sig.replace(0, np.nan)
    d4 = (zp.abs() >= 1.5 * p["div_prix_seuil"]) & (cr.abs() <= p["div_ralentissement"] * prior.abs())
    typ = np.select([d1, d2, d3, d4], ["prix_hausse_activite_baisse", "prix_baisse_volatilite_hausse",
                                       "volume_eleve_sans_progression", "ralentissement_apres_forte_variation"], "aucune")
    dire = np.select([d1, d2, d3, d4], [1, -1, np.sign(cr), np.sign(prior)], 0)
    out = pd.DataFrame(index=d.index)
    out["anomalie_divergence"] = typ
    out["a5_on"] = (typ != "aucune")
    out["a5_dir"] = np.nan_to_num(dire).astype(int)
    out["a5_variante"] = "complete" if volume_ok else "prix_volatilite"
    return out

# ── Configurations de tendance + calibration + contexte ────────────────────
TEND = dict(
    q_achat=0.90, q_achat_fort=0.95, q_vente=0.10, q_vente_fort=0.05,
    q_surv_long=(0.80, 0.90), q_surv_short=(0.10, 0.20),
    bonus_vol="expansion_soudaine", bonus_conf=10, conf_max=95,
    z_conf_forte=1.0, z_exceptionnel=2.0, z_clip=3.0,
    poids_quantile=0.8, poids_zscore=0.2, calib=None,
)
COULEURS_TEND = {
    "ACHAT FORT": ("#1e8449", "white"), "ACHAT": ("#d5f5e3", "#145a32"),
    "SURVEILLANCE LONG": ("#f9e79f", "#7d6608"), "SURVEILLANCE SHORT": ("#f9e79f", "#7d6608"),
    "VENTE": ("#fadbd8", "#922b21"), "VENTE FORTE": ("#922b21", "white"),
}
SENS_CFG = {"ACHAT FORT": 1, "ACHAT": 1, "SURVEILLANCE LONG": 1, "VENTE FORTE": -1, "VENTE": -1, "SURVEILLANCE SHORT": -1}
DECLASSE = {"ACHAT FORT": "ACHAT", "ACHAT": "SURVEILLANCE LONG", "SURVEILLANCE LONG": "ATTENTE",
            "VENTE FORTE": "VENTE", "VENTE": "SURVEILLANCE SHORT", "SURVEILLANCE SHORT": "ATTENTE"}

def ajouter_tendance(d, t=TEND):
    d = d.copy()
    q = d["quantile_rendement"]; r = d["ret"]; z = d["z_robuste"]
    exp = (d["anomalie_vol"] == t["bonus_vol"]).fillna(False)
    positif = (r > 0).fillna(False); negatif = (r < 0).fillna(False)
    d["classe_q"] = np.select([q >= t["q_achat_fort"], q >= t["q_achat"], q <= t["q_vente_fort"], q <= t["q_vente"]],
                              ["ACHAT FORT", "ACHAT", "VENTE FORTE", "VENTE"], default="")
    d["signal_tendance"] = np.select(
        [positif & (q >= t["q_achat_fort"]), positif & (q >= t["q_achat"]),
         positif & (q >= t["q_surv_long"][0]) & (q < t["q_surv_long"][1]),
         negatif & (q <= t["q_vente_fort"]), negatif & (q <= t["q_vente"]),
         negatif & (q > t["q_surv_short"][0]) & (q <= t["q_surv_short"][1])],
        ["ACHAT FORT", "ACHAT", "SURVEILLANCE LONG", "VENTE FORTE", "VENTE", "SURVEILLANCE SHORT"],
        default="ATTENTE")
    d["signal_prioritaire"] = np.select(
        [positif & (q >= t["q_achat"]) & exp, positif & (q >= t["q_achat"]), negatif & (q <= t["q_vente"]) & exp],
        ["🟢🟢 LONG PRIORITAIRE RENFORCÉ", "🟢 LONG PRIORITAIRE", "🔴🔴 SHORT PRIORITAIRE RENFORCÉ"], default="")
    d["signal_prioritaire"] = np.where(negatif & (q <= t["q_vente"]) & (d["signal_prioritaire"] == ""),
                                       "🔴 SHORT PRIORITAIRE", d["signal_prioritaire"])
    pct0 = np.select([q >= t["q_achat_fort"], q >= t["q_achat"], q <= t["q_vente_fort"], q <= t["q_vente"]],
                     [80, 70, 80, 70], default=np.nan).astype(float)
    pct = np.where(exp.values & ~np.isnan(pct0), np.minimum(pct0 + t["bonus_conf"], t["conf_max"]), pct0)
    if t.get("calib"):
        cal = np.array([t["calib"].get((c, bool(e)), np.nan) for c, e in zip(d["classe_q"], exp)], float)
        pct = np.where(~np.isnan(cal) & ~np.isnan(pct0), np.minimum(cal, t["conf_max"]), pct)
    sens = np.where(q >= t["q_achat"], "LONG", np.where(q <= t["q_vente"], "SHORT", ""))
    d["probabilite_directionnelle"] = [f"{s} {int(x)}%" if (s and not np.isnan(x)) else "" for s, x in zip(sens, pct)]
    comp_q = (q * 100).clip(0, 100)
    comp_z = ((z.clip(-t["z_clip"], t["z_clip"]) + t["z_clip"]) / (2 * t["z_clip"]) * 100)
    d["score_momentum"] = (t["poids_quantile"] * comp_q + t["poids_zscore"] * comp_z).round(1)
    score_dir = np.where(r >= 0, d["score_momentum"], 100 - d["score_momentum"])
    d["momentum"] = pd.cut(score_dir, [-0.1, 40, 60, 75, 100.1],
                           labels=["Faible", "Neutre", "Momentum", "Momentum fort"]).astype(str)
    d["confirmation_zscore"] = np.select(
        [z > t["z_exceptionnel"], z < -t["z_exceptionnel"], z > t["z_conf_forte"], z < -t["z_conf_forte"]],
        ["⚠ Exceptionnel : Z > 2", "⚠ Exceptionnel : Z < -2", "Confirmation forte (Z > 1)", "Confirmation forte (Z < -1)"],
        default="—")
    return d

def _minutes(hhmm): h, m = hhmm.split(":"); return int(h) * 60 + int(m)

def fenetre_macro(index, p):
    m = np.zeros(len(index), bool)
    for a in p.get("annonces", []):
        ts = pd.Timestamp(a)
        if index.tz is not None: ts = ts.tz_localize(p["fuseau"]) if ts.tzinfo is None else ts.tz_convert(p["fuseau"])
        m |= np.asarray((index >= ts - pd.Timedelta(minutes=p["annonce_avant_min"])) &
                        (index <= ts + pd.Timedelta(minutes=p["annonce_apres_min"])))
    return m

def tendance_horaire(d, p):
    if not est_intraday(p): return np.sign(d["ema_rapide"] - d["ema_lente"]).fillna(0).values
    h = d["close"].resample("1h").last().dropna()
    sg = np.sign(h - h.ewm(span=p["ema_tendance_h"], adjust=False).mean()).shift(1)
    return sg.reindex(d.index.floor("1h")).fillna(0).values

def ajouter_contexte(d, p):
    d = d.copy(); n = len(d); actif = p.get("filtres_contexte", True)
    if est_intraday(p) and actif:
        hm = np.asarray(d.index.hour * 60 + d.index.minute)
        ok_h = (hm >= _minutes(p["heure_min_signal"])) & (hm <= _minutes(p["heure_max_signal"]))
        ok_m = ~fenetre_macro(d.index, p)
    else:
        ok_h = np.ones(n, bool); ok_m = np.ones(n, bool)
    d["ok_horaire"], d["ok_macro"], d["contexte_ok"] = ok_h, ok_m, ok_h & ok_m
    ema_s = np.sign((d["ema_rapide"] - d["ema_lente"]).values.astype(float)); ema_s[:p["ema_lente"]] = 0
    d["tend_1h"] = tendance_horaire(d, p)
    d["align_score"] = (ema_s + d["tend_1h"].values).astype(int)
    brut = d["signal_tendance"]; sens = brut.map(SENS_CFG).fillna(0).astype(int).values
    mode = p["filtre_alignement"] if actif else "off"
    d["contre_tendance"] = (d["align_score"].values * sens) < 0
    fin = brut.copy()
    if mode == "declasse": fin = fin.where(~d["contre_tendance"], fin.map(DECLASSE))
    elif mode == "bloque": fin = fin.where(~d["contre_tendance"], "ATTENTE")
    fin = fin.where(d["contexte_ok"], "ATTENTE")
    motif = np.select([~ok_h, ~ok_m, d["contre_tendance"].values & (mode != "off")],
                      ["hors plage horaire", "annonce macro", "contre-tendance"], default="")
    d["filtre_motif"] = np.where(brut.values != "ATTENTE", motif, "")
    d["signal_tendance_brut"], d["signal_tendance"] = brut, fin
    action = fin.isin(["ACHAT FORT", "ACHAT", "VENTE", "VENTE FORTE"])
    d["signal_prioritaire"] = d["signal_prioritaire"].where(action, "")
    d["probabilite_directionnelle"] = d["probabilite_directionnelle"].where(action, "")
    return d

def finaliser_tendance(d, p): return ajouter_contexte(ajouter_tendance(d), p)

def calibrer_probas(d, p, masque=None):
    h, nmin = p["calib_horizon"], p["calib_n_min"]
    fwd = (d["close"].shift(-h) / d["open"].shift(-1) - 1).values
    cl = d["classe_q"].values; exp = (d["anomalie_vol"] == TEND["bonus_vol"]).values
    ok = (cl != "") & ~np.isnan(fwd)
    if masque is not None: ok &= np.asarray(masque)
    rows, calib = [], {}
    for c in ["ACHAT FORT", "ACHAT", "VENTE FORTE", "VENTE"]:
        sens = 1 if c.startswith("ACHAT") else -1
        for e in (False, True):
            pos = np.flatnonzero(ok & (cl == c) & (exp == e)); keep, last = [], -10**9
            for i in pos:
                if i - last >= h: keep.append(i); last = i
            n = len(keep)
            if n < 5: continue
            x = sens * fwd[keep]; k = int((x > 0).sum()); ci = binomtest(k, n).proportion_ci()
            rows.append(dict(classe=c, expansion=e, n=n, p_reel=k / n, ic_bas=ci.low, ic_haut=ci.high,
                             moy_bp=x.mean() * 1e4, significatif=bool(ci.low > 0.5 or ci.high < 0.5)))
            if n >= nmin: calib[(c, e)] = int(round(100 * k / n))
    return pd.DataFrame(rows).round(3), calib

def calibrer_et_appliquer(d, p, verbose=True):
    masques, _ = decouper(d, p)
    tab, calib = calibrer_probas(d, p, masques["apprentissage"] | masques["validation"])
    if verbose:
        st.markdown(f"**Calibration des probabilités (horizon {p['calib_horizon']} barres)**")
        st.dataframe(tab if len(tab) else pd.DataFrame([{"info": "Pas assez d'événements."}]))
        tt, _ = calibrer_probas(d, p, masques["test"])
        st.dataframe(tt if len(tt) else pd.DataFrame([{"info": "Test trop court."}]))
    TEND["calib"] = calib or None
    return finaliser_tendance(d, p)

COLONNES_TEND = ["close", "rendement", "quantile_rendement", "z_robuste", "anomalie_vol",
                 "signal", "signal_tendance", "filtre_motif", "signal_prioritaire",
                 "probabilite_directionnelle", "score_momentum", "momentum",
                 "confirmation_zscore", "type_anomalie", "motif"]

def tableau_tendance(d, jour=None, seulement_signaux=False):
    if "signal_tendance" not in d.columns: d = finaliser_tendance(d, PARAMS_DEFAUT)
    if jour is None: jour = d.index[-1].date()
    s = d[d.index.date == jour].copy()
    s["rendement"] = s["ret"]
    s = s[[c for c in COLONNES_TEND if c in s.columns]]
    if seulement_signaux: s = s[s["signal_tendance"] != "ATTENTE"]
    def _bg(row):
        cfg = COULEURS_TEND.get(row["signal_tendance"])
        if cfg is None: return ["" for _ in row.index]
        bg, fg = cfg; gras = "font-weight:bold;" if row["signal_prioritaire"] else ""
        return [f"background-color:{bg};color:{fg};{gras}" for _ in row.index]
    return s.style.apply(_bg, axis=1).format(
        {"close": "{:.2f}", "rendement": "{:+.4%}", "quantile_rendement": "{:.3f}",
         "z_robuste": "{:+.2f}", "score_momentum": "{:.1f}"}, na_rep="—")

# ── Régime, confluence, signaux ─────────────────────────────────────────────
def classer_regime(d, p):
    d = d.copy()
    d["ema_rapide"] = d["close"].ewm(span=p["ema_rapide"], adjust=False).mean()
    d["ema_lente"] = d["close"].ewm(span=p["ema_lente"], adjust=False).mean()
    n = p["er_fenetre"]
    d["er"] = (d["close"] - d["close"].shift(n)).abs() / d["close"].diff().abs().rolling(n).sum()
    d["regime"] = np.select(
        [(d["er"] >= p["er_tendance"]) & (d["ema_rapide"] > d["ema_lente"]),
         (d["er"] >= p["er_tendance"]) & (d["ema_rapide"] < d["ema_lente"]), d["er"] <= p["er_range"]],
        ["tendance_haussiere", "tendance_baissiere", "range"], "indetermine")
    d.loc[d["er"].isna() | d["ema_lente"].isna(), "regime"] = "indetermine"
    d["regime_vol"] = np.select([d["vol_score"] >= p["regime_vol_haut"], d["vol_score"] <= p["regime_vol_bas"]],
                                ["volatilite_elevee", "volatilite_faible"], "normale")
    return d

def calculer_confluence(d):
    dir_prix = np.where(d["a1_on"], d["a1_dir"], np.where(d["a4_on"], d["a4_dir"], 0))
    fams = [d["a1_on"] | d["a4_on"], d["a2_on"], d["a3_on"], d["a5_on"]]
    dirs = [dir_prix, d["a2_dir"].values, d["a3_dir"].values, d["a5_dir"].values]
    d["score_confluence"] = sum(f.astype(int) for f in fams)
    d["direction_observee"] = np.sign(sum(np.where(f, x, 0) for f, x in zip(fams, dirs))).astype(int)
    d["n_anomalies"] = d[[f"a{k}_on" for k in range(1, 6)]].sum(axis=1)
    on = np.column_stack([d[f"a{k}_on"].values for k in range(1, 6)])
    d["type_anomalie"] = ["+".join(n for n, x in zip(NOMS, row) if x) or "aucune" for row in on]
    return d

def generer_signaux(d, p, strat=None):
    strat = strat or p["strategie"]; n = len(d); ex = np.zeros(n, dtype=int)
    for k in p["anomalies_strategie"]:
        ex = np.where(ex != 0, ex, np.where(d[f"{k}_on"], d[f"{k}_dir"], 0))
    exs = pd.Series(ex, index=d.index); rec = np.zeros(n, dtype=int)
    for lag in range(1, p["anomalie_memoire"] + 1):
        rec = np.where(rec != 0, rec, exs.shift(lag).fillna(0).astype(int).values)
    c, atr = d["close"], d["atr"]
    sig = np.zeros(n, dtype=int)
    if strat == "A":
        ctx = d["regime"].isin(["range", "indetermine"]).values
        cible = c.rolling(p["cible_sma"]).mean()
        long_ = (rec == -1) & (c > d["high"].shift(1)).values & ctx & (cible > c).values
        short = (rec == 1) & (c < d["low"].shift(1)).values & ctx & (cible < c).values
        tgt = cible.values
    else:
        hh = d["high"].shift(1).rolling(p["breakout_n"]).max(); ll = d["low"].shift(1).rolling(p["breakout_n"]).min()
        long_ = (rec == 1) & (d["regime"] == "tendance_haussiere").values & (c > hh).values
        short = (rec == -1) & (d["regime"] == "tendance_baissiere").values & (c < ll).values
        tgt = (c + sig * p["target_atr"] * atr).values
    sig = np.where(long_, 1, np.where(short, -1, 0)); sig[atr.isna().values] = 0
    if strat != "A": tgt = (c + sig * p["target_atr"] * atr).values
    stop = (c - sig * p["stop_atr"] * atr).values
    out = pd.DataFrame({"sig": sig, "stop": stop, "target": tgt}, index=d.index)
    out["motif_signal"] = ""
    for i in np.flatnonzero(sig):
        out.iloc[i, 3] = (("Retour à la moyenne" if strat == "A" else "Continuation") +
                          f" {'ACHAT' if sig[i] > 0 else 'VENTE'} : anomalie récente ({d['type_anomalie'].iloc[max(i-3,0):i+1][lambda s: s!='aucune'].iloc[-1] if (d['type_anomalie'].iloc[max(i-3,0):i+1]!='aucune').any() else 'prix'}), "
                          f"régime {d['regime'].iloc[i]}, confirmation par le prix, invalidation {stop[i]:.1f}")
    return out

def detecter_anomalies(df, params, volume_ok=True, tendance=True):
    p = params; d = preparer(df, p)
    d = pd.concat([d, anomalie_1_zscore(d, p), anomalie_2_volatilite(d, p)], axis=1)
    d = pd.concat([d, anomalie_3_volume(d, p, volume_ok), anomalie_4_quantiles(d, p)], axis=1)
    d = pd.concat([d, anomalie_5_divergence(d, p, volume_ok)], axis=1)
    d = calculer_confluence(classer_regime(d, p))
    d = d.join(generer_signaux(d, p))
    d["signal"] = d["sig"].map({1: "ACHAT", -1: "VENTE", 0: "ATTENTE"})
    def motif(r):
        if r.n_anomalies == 0: return "Aucune anomalie."
        m = []
        if r.a1_on: m.append(f"Z={r.z:.2f}")
        if r.a2_on: m.append(f"vol:{r.anomalie_vol}")
        if r.a3_on: m.append(f"volume:{r.anomalie_volume}")
        if r.a4_on: m.append(f"quantile={r.quantile_rendement:.3f}")
        if r.a5_on: m.append(f"div:{r.anomalie_divergence}")
        return " | ".join(m) + f" | régime {r.regime}/{r.regime_vol} | confluence {r.score_confluence}"
    d["motif"] = [motif(r) if r.n_anomalies else "Aucune anomalie." for r in d.itertuples()]
    if tendance: d = finaliser_tendance(d, p)
    return d

COLONNES_MIN = ["close", "ret", "z", "z_robuste", "anomalie_vol", "etat_cusum", "anomalie_volume",
                "quantile_rendement", "anomalie_divergence", "regime", "score_confluence",
                "type_anomalie", "direction_observee", "signal", "motif",
                "signal_tendance_brut", "signal_tendance", "filtre_motif", "contexte_ok", "align_score",
                "signal_prioritaire", "probabilite_directionnelle", "score_momentum", "momentum", "confirmation_zscore"]

def controle_fuites(df, p, volume_ok, frac=0.8):
    d_full = detecter_anomalies(df, p, volume_ok, tendance=False)
    k = int(len(df) * frac); d_cut = detecter_anomalies(df.iloc[:k], p, volume_ok, tendance=False)
    cols = ["z", "z_robuste", "vol_score", "etat_cusum", "vol_z", "q_bas_val", "q_haut_val", "quantile_rendement",
            "er", "score_confluence", "sig"]
    ok = True; problemes = []
    for c in cols:
        a, b = d_full[c].iloc[:k].values.astype(float), d_cut[c].values.astype(float)
        if not np.allclose(a, b, equal_nan=True, atol=1e-9): problemes.append(c); ok = False
    return ok, problemes

# ── Étude d'événements ────────────────────────────────────────────────────
def _ic_boot(x, nb, rng):
    if len(x) < 5: return (np.nan, np.nan)
    m = rng.choice(x, size=(nb, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))

def _maxdd(r):
    if len(r) == 0: return np.nan
    eq = np.cumprod(1 + np.asarray(r)); return float((eq / np.maximum.accumulate(eq) - 1).min())

def etude_evenements(d, flags, p):
    rng = np.random.default_rng(p["seed"]); cout = cout_aller_retour(p); entry = d["open"].shift(-1); rows = []
    for h in horizons(p):
        fwd = d["close"].shift(-h) / entry - 1; fv = fwd.values; base = np.nanmean(fv)
        for nom, (on, dr_) in flags.items():
            onv, drv = np.asarray(on, bool), np.asarray(dr_)
            for sens in (-1, 1):
                pos = np.flatnonzero(onv & (drv == sens) & ~np.isnan(fv)); keep, last = [], -10**9
                for q in pos:
                    if q - last >= h: keep.append(q); last = q
                n = len(keep); row = dict(anomalie=nom, direction="hausse" if sens > 0 else "baisse",
                                          horizon=h, n_brut=len(pos), n=n)
                if n >= p["n_min_evenements"]:
                    ev = fv[keep]; cont = sens * (ev - base)
                    lo, hi = _ic_boot(cont, p["bootstrap"], rng)
                    row.update(ret_moy_bp=ev.mean() * 1e4, ret_med_bp=np.median(ev) * 1e4,
                               pct_hausse=(ev > 0).mean(), pct_baisse=(ev < 0).mean(), base_bp=base * 1e4,
                               cont_exc_bp=cont.mean() * 1e4, ic_bas_bp=lo * 1e4, ic_haut_bp=hi * 1e4,
                               p=stats.ttest_1samp(cont, 0).pvalue if cont.std() > 0 else np.nan,
                               dd_continuation=_maxdd(sens * ev - cout), dd_retour_moyenne=_maxdd(-sens * ev - cout))
                rows.append(row)
    t = pd.DataFrame(rows)
    if "p" in t.columns:
        ok = t["p"].notna(); t["p_adj"] = np.nan
        if ok.sum() > 0: t.loc[ok, "p_adj"] = multipletests(t.loc[ok, "p"], method="fdr_bh")[1]
        def verdict(r):
            if pd.isna(r.get("p_adj")): return "n insuffisant"
            if r.p_adj <= p["alpha"]: return "continuation" if r.cont_exc_bp > 0 else "retour_moyenne"
            if (r.ic_bas_bp > 0) or (r.ic_haut_bp < 0): return "indice faible (non corrigé)"
            return "aucun avantage démontré"
        t["verdict"] = t.apply(verdict, axis=1)
    return t

def flags_anomalies(d):
    return {f"a{k}_{NOMS[k-1]}": (d[f"a{k}_on"], d[f"a{k}_dir"]) for k in range(1, 6)}

def flags_combinaisons(d, tailles=(2, 3)):
    from itertools import combinations
    f = {}
    for t in tailles:
        for combo in combinations(range(1, 6), t):
            on = np.ones(len(d), bool); dirs = [d[f"a{k}_dir"].values for k in combo]
            for k in combo: on &= d[f"a{k}_on"].values
            for x in dirs[1:]: on &= (x == dirs[0])
            f["+".join(NOMS[k-1] for k in combo)] = (on, np.where(on, dirs[0], 0))
    f["confluence>=3 familles"] = (d["score_confluence"].values >= 3, d["direction_observee"].values)
    return f

def issue_evenement(d, i, s, h, p):
    c0, a = d["close"].iloc[i], d["atr"].iloc[i]
    if np.isnan(a): return "n/a"
    fav, adv = c0 + s * p["target_atr"] * a, c0 - s * p["stop_atr"] * a
    for k in range(i + 1, min(i + h + 1, len(d))):
        hi, lo = d["high"].iloc[k], d["low"].iloc[k]
        hit_f = hi >= fav if s > 0 else lo <= fav; hit_a = lo <= adv if s > 0 else hi >= adv
        if hit_a: return "defavorable"
        if hit_f: return "exploitable"
    return "sans_suite"

def synthese_evenements(d, p):
    h = horizons(p)[len(horizons(p)) // 2]; pos = np.flatnonzero((d["n_anomalies"] > 0) & (d["direction_observee"] != 0))
    keep, last = [], -10**9
    for q in pos:
        if q - last >= h: keep.append(q); last = q
    rows = []
    for i in keep:
        dr = int(d["direction_observee"].iloc[i]); g = "simultanees" if d["score_confluence"].iloc[i] >= 2 else "isolees"
        rows.append(dict(groupe=g, continuation=issue_evenement(d, i, dr, h, p), retour_moyenne=issue_evenement(d, i, -dr, h, p)))
    if not rows: return pd.DataFrame()
    r = pd.DataFrame(rows)
    out = pd.concat({hyp: pd.crosstab(r["groupe"], r[hyp]) for hyp in ["continuation", "retour_moyenne"]}, axis=1).fillna(0).astype(int)
    return out, len(r)

# ── Stratégies A/B : simulation, métriques, validation ──────────────────────
def simuler(d, sig, p, masque=None):
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    sg, st, tg = sig["sig"].values, sig["stop"].values, sig["target"].values
    n = len(d); cout = cout_aller_retour(p); trades = []; i = 0
    while i < n - 1:
        if sg[i] != 0 and (masque is None or masque[i]) and not np.isnan(st[i]) and not np.isnan(tg[i]):
            dr, e = sg[i], i + 1; entry = o[e]; px = None; jx = e; motif = "temps"
            for j in range(e, min(n, e + p["tenue_max"])):
                jx = j
                if dr == 1:
                    if o[j] <= st[i]: px, motif = o[j], "stop"
                    elif o[j] >= tg[i]: px, motif = o[j], "cible"
                    elif l[j] <= st[i]: px, motif = st[i], "stop"
                    elif h[j] >= tg[i]: px, motif = tg[i], "cible"
                else:
                    if o[j] >= st[i]: px, motif = o[j], "stop"
                    elif o[j] <= tg[i]: px, motif = o[j], "cible"
                    elif h[j] >= st[i]: px, motif = st[i], "stop"
                    elif l[j] <= tg[i]: px, motif = tg[i], "cible"
                if px is not None: break
            if px is None: px = c[jx]
            trades.append(dict(signal_time=d.index[i], entry_time=d.index[e], exit_time=d.index[jx], dir=int(dr),
                               ret=dr * (px / entry - 1) - cout, sortie=motif, regime=d["regime"].iloc[i],
                               type_anomalie=d["type_anomalie"].iloc[max(i-p["anomalie_memoire"], 0):i+1].iloc[-1]))
            i = jx
        else:
            i += 1
    return pd.DataFrame(trades)

def metriques(tr, p, debut, fin):
    cols = ["n_trades", "taux_reussite", "esperance_nette_bp", "profit_factor", "rendement_cumule", "drawdown_max", "sharpe", "sortino"]
    if tr is None or tr.empty: return dict.fromkeys(cols, np.nan) | {"n_trades": 0}
    r = tr["ret"].values; g, pe = r[r > 0].sum(), -r[r < 0].sum(); eq = np.cumprod(1 + r)
    ans = max((fin - debut).days / 365.25, 1 / 12); tpa = len(r) / ans
    sd = r.std(ddof=1) if len(r) > 1 else np.nan; dsd = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    ex = r.mean() - p["rf_annuel"] / tpa
    return dict(n_trades=len(r), taux_reussite=(r > 0).mean(), esperance_nette_bp=r.mean() * 1e4,
                profit_factor=g / pe if pe > 0 else np.inf, rendement_cumule=eq[-1] - 1, drawdown_max=_maxdd(r),
                sharpe=ex / sd * np.sqrt(tpa) if sd and sd > 0 else np.nan,
                sortino=ex / dsd * np.sqrt(tpa) if dsd > 0 else np.nan)

def decouper(d, p):
    n = len(d); a, b, _ = p["split"]; t1, t2 = d.index[int(n * a)], d.index[int(n * (a + b))]
    return {"apprentissage": d.index < t1, "validation": (d.index >= t1) & (d.index < t2), "test": d.index >= t2}, (t1, t2)

def baseline_aleatoire(d, p, n_signaux, masque, nsim=100):
    rng = np.random.default_rng(p["seed"]); idx = np.flatnonzero(masque & d["atr"].notna().values)
    if n_signaux < 5 or len(idx) < 50: return np.nan, np.nan
    res = []
    for _ in range(nsim):
        s = np.zeros(len(d), int); ch = rng.choice(idx, size=min(n_signaux, len(idx)), replace=False)
        s[ch] = rng.choice([-1, 1], size=len(ch)); c, a = d["close"], d["atr"]
        sg = pd.DataFrame({"sig": s, "stop": (c - s * p["stop_atr"] * a).values,
                           "target": (c + s * p["target_atr"] * a).values}, index=d.index)
        t = simuler(d, sg, p, masque)
        if len(t): res.append(t["ret"].mean() * 1e4)
    return (np.mean(res), np.std(res)) if res else (np.nan, np.nan)

def evaluer_strategie(d, p, strat, verbose=True):
    masques, (t1, t2) = decouper(d, p); sig = generer_signaux(d, p, strat)
    rows, trades = [], {}
    for seg, m in masques.items():
        tr = simuler(d, sig, p, m); trades[seg] = tr
        deb, fin = d.index[m][0], d.index[m][-1]
        rows.append({"segment": seg, "n_signaux": int((sig["sig"].values[m] != 0).sum()), **metriques(tr, p, deb, fin)})
    tous = simuler(d, sig, p); trades["tous"] = tous
    rows.append({"segment": "tous", "n_signaux": int((sig["sig"] != 0).sum()), **metriques(tous, p, d.index[0], d.index[-1])})
    tab = pd.DataFrame(rows).set_index("segment")
    mt = masques["test"]; bm, bs = baseline_aleatoire(d, p, tab.loc["test", "n_signaux"], mt)
    extras = dict(t1=t1, t2=t2, baseline=(bm, bs),
                  bh=d["close"][mt].iloc[-1] / d["close"][mt].iloc[0] - 1)
    return tab, trades, sig, extras

def walk_forward(d, p, strat, k=5):
    sig = generer_signaux(d, p, strat); edges = np.linspace(0, len(d), k + 1, dtype=int); rows = []
    for f in range(k):
        m = np.zeros(len(d), bool); m[edges[f]:edges[f + 1]] = True
        tr = simuler(d, sig, p, m); rows.append({"bloc": f + 1, "debut": d.index[edges[f]].date(),
                                                 **metriques(tr, p, d.index[edges[f]], d.index[edges[f+1]-1])})
    return pd.DataFrame(rows).set_index("bloc")

# ── Tableau de bord / décision ─────────────────────────────────────────────
def decision_finale(d, p, tab_strat):
    last = d.iloc[-1]; s = int(last["sig"])
    if s == 0: return "ATTENTE", "Règles de la stratégie non satisfaites (anomalie récente + confirmation + contexte requis)."
    if p["exiger_validation_test"]:
        t = tab_strat.loc["test"]
        if t["n_trades"] < 5 or not (t["esperance_nette_bp"] > 0):
            return "ATTENTE", (f"Signal {'ACHAT' if s>0 else 'VENTE'} présent mais avantage non démontré hors échantillon "
                               f"(test : {int(t['n_trades'])} trades, espérance {t['esperance_nette_bp']:.2f} bp).")
    return ("ACHAT" if s > 0 else "VENTE"), last["motif_signal"]

def resume_last(d):
    last = d.iloc[-1]
    lignes = {
        "Clôture": f"{last['close']:.2f}",
        "Régime": f"{last['regime']} / {last['regime_vol']}",
        "Confluence": f"{int(last['score_confluence'])}/4 familles",
        "Direction observée": f"{int(last['direction_observee']):+d}",
        "Tendance": f"{last['signal_tendance']} (brut {last['signal_tendance_brut']}, {last['filtre_motif'] or 'aucun filtre'})",
        "Prioritaire": last["signal_prioritaire"] or "—",
        "Probabilité": last["probabilite_directionnelle"] or "—",
        "Momentum": f"{last['score_momentum'] if pd.notna(last['score_momentum']) else '—'} ({last['momentum']})",
        "Z-score": f"{last['z']:.2f} (robuste {last['z_robuste']:.2f})",
        "Anomalies actives": last["type_anomalie"],
        "Rendement barre": f"{last['ret']:+.4%}" if pd.notna(last["ret"]) else "—",
    }
    return lignes

# ── Système DQZ ────────────────────────────────────────────────────────────
SYS = dict(z_min=1.0, q_long=0.6, q_short=0.4, q_sortie=0.5, z_stop=1.0,
           fin_session_min=17 * 60 + 30,
           taille_base=0.5, taille_conf=1.0, conf_min=2,
           sizing="atr", taille_conf_mult=2.0, taille_min=0.1, taille_max=3.0,
           stop_atr=2.0, suiveur=True, filtre_contexte=True, bloquer_contre_tendance=False)

def taille_position(atr, conf, p, s=SYS):
    forte = conf >= s["conf_min"]
    if s["sizing"] == "atr" and s["stop_atr"] and atr and not np.isnan(atr) and atr > 0:
        risque = p["capital"] * p["risque_pct"] / 100
        if not forte: risque /= s["taille_conf_mult"]
        return float(np.clip(round(risque / (s["stop_atr"] * atr), 2), s["taille_min"], s["taille_max"]))
    return s["taille_conf"] if forte else s["taille_base"]

def systeme_dqz(d, p, s=SYS):
    n = len(d); dur = int(p["intervalle"].replace("m", ""))
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close")); atr = d["atr"].values
    dr = d["direction_observee"].values.astype(int)
    z, q = d["z"].values, d["quantile_rendement"].values
    conf = d["score_confluence"].values
    jour = np.asarray(d.index.date); hm = d.index.hour * 60 + d.index.minute
    force = (hm + dur >= s["fin_session_min"]) | np.r_[jour[1:] != jour[:-1], True]
    prev = np.r_[0, dr[:-1]]
    entree = ((dr != 0) & (prev == 0) & (np.abs(z) >= s["z_min"]) &
              (((dr == 1) & (q >= s["q_long"])) | ((dr == -1) & (q <= s["q_short"]))))
    if s.get("filtre_contexte", True) and "contexte_ok" in d: entree &= d["contexte_ok"].values
    if s.get("bloquer_contre_tendance", False) and "align_score" in d: entree &= ~((d["align_score"].values * dr) < 0)
    cout = cout_aller_retour(p); trades = []; i = 0
    while i < n - 1:
        if entree[i] and not force[i] and jour[i + 1] == jour[i]:
            side, e = dr[i], i + 1; entry = o[e]; j = e; a = atr[i]
            use_atr = bool(s["stop_atr"]) and not np.isnan(a) and a > 0; ext = entry
            while True:
                if use_atr:
                    stop = ext - side * s["stop_atr"] * a
                    if (l[j] <= stop) if side == 1 else (h[j] >= stop):
                        gap = (o[j] <= stop) if side == 1 else (o[j] >= stop)
                        px, xi = (o[j] if gap else stop), j
                        motif = "stop ATR suiveur" if (s["suiveur"] and ext != entry) else "stop ATR initial"; break
                    if s["suiveur"]: ext = max(ext, h[j]) if side == 1 else min(ext, l[j])
                if side == 1:  tp, sm, sd = q[j] < s["q_sortie"], z[j] <= -s["z_stop"], dr[j] == -1
                else:          tp, sm, sd = q[j] > s["q_sortie"], z[j] >=  s["z_stop"], dr[j] == 1
                motif = "TP quantile" if tp else "stop momentum" if sm else "stop direction" if sd else None
                if force[j]:
                    px, xi, motif = c[j], j, (motif or "fin de session"); break
                if motif:
                    px, xi = o[j + 1], j + 1; break
                j += 1
            taille = taille_position(a, int(conf[i]), p, s)
            pts = side * (px - entry) - cout * entry
            trades.append(dict(signal_time=d.index[i], entry_time=d.index[e], exit_time=d.index[xi],
                               sens="ACHAT" if side > 0 else "VENTE", entree=entry, sortie=px,
                               pts_nets=pts, taille=taille, pnl_eur=pts * taille, sortie_motif=motif,
                               confluence=int(conf[i]), regime=d["regime"].iloc[i],
                               type_anomalie=d["type_anomalie"].iloc[i]))
            i = xi
        else:
            i += 1
    return pd.DataFrame(trades)

def _stats(t):
    if t is None or t.empty: return dict(n=0)
    g, pe = t.loc[t.pnl_eur > 0, "pnl_eur"].sum(), -t.loc[t.pnl_eur < 0, "pnl_eur"].sum()
    eq = t["pnl_eur"].cumsum()
    return dict(n=len(t), reussite=(t.pnl_eur > 0).mean(), pnl_total_eur=t.pnl_eur.sum(),
                esperance_eur=t.pnl_eur.mean(), pts_moy=t.pts_nets.mean(), taille_moy=t.taille.mean(),
                profit_factor=g / pe if pe > 0 else np.inf, drawdown_eur=(eq - eq.cummax()).min())

# ── Notifications ───────────────────────────────────────────────────────────
def envoyer_telegram(texte):
    tok, chat = _secret("TELEGRAM_TOKEN"), _secret("TELEGRAM_CHAT_ID")
    if not (tok and chat): return False
    try:
        return requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                             data={"chat_id": chat, "text": texte}, timeout=10).ok
    except Exception:
        return False

def envoyer_email(sujet, texte):
    host, user, pwd, dest = (_secret(k) for k in ("EMAIL_SMTP", "EMAIL_USER", "EMAIL_PASS", "EMAIL_DEST"))
    if not (host and user and pwd and dest): return False
    msg = EmailMessage(); msg["Subject"], msg["From"], msg["To"] = sujet, user, dest; msg.set_content(texte)
    with smtplib.SMTP(host, int(_secret("EMAIL_PORT", "587")), timeout=15) as s:
        s.starttls(); s.login(user, pwd); s.send_message(msg)
    return True

def notifier(titre, message, muet=False):
    st.session_state.setdefault("log_notifications", []).append(
        f"{pd.Timestamp.now(tz='Europe/Paris'):%H:%M:%S} — {titre} — {message.replace(chr(10), ' | ')}")
    if muet: return
    if POPUP_ACTIF: st.toast(f"{titre} : {message}", icon="🎯")
    if TELEGRAM_ACTIF:
        try: envoyer_telegram(f"{titre}\n\n{message}")
        except Exception as e: st.warning(f"Telegram : {e}")
    if EMAIL_ACTIF:
        try: envoyer_email(titre, message)
        except Exception as e: st.warning(f"Email : {e}")

# ── Journal SQLite ─────────────────────────────────────────────────────────
def journaliser(ts, type_, sens="", config="", prix=np.nan, details="", p=None):
    try:
        os.makedirs(ETAT_DOSSIER, exist_ok=True)
        with contextlib.closing(sqlite3.connect(DB_JOURNAL)) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS signaux (barre TEXT, symbole TEXT, intervalle TEXT, type TEXT, sens TEXT,
                          config TEXT, prix REAL, details TEXT, enregistre_le TEXT, PRIMARY KEY (barre, symbole, intervalle, type))""")
            cx.execute("INSERT OR IGNORE INTO signaux VALUES (?,?,?,?,?,?,?,?,?)",
                       (str(ts), (p or st.session_state.params)["symbole"], (p or st.session_state.params)["intervalle"],
                        type_, sens, config, float(prix), details,
                        str(pd.Timestamp.now(tz=(p or st.session_state.params)["fuseau"]))))
            cx.commit()
    except Exception as e:
        st.warning(f"Journal SQLite : {e}")

def lire_journal(p):
    if not os.path.exists(DB_JOURNAL): return pd.DataFrame()
    with contextlib.closing(sqlite3.connect(DB_JOURNAL)) as cx:
        return pd.read_sql("SELECT * FROM signaux WHERE symbole=? AND intervalle=? ORDER BY enregistre_le DESC",
                           cx, params=(p["symbole"], p["intervalle"]))

# ── Logique d'alerte live ───────────────────────────────────────────────────
def niveau_stop(pos):
    if not (SYS["stop_atr"] and pos.get("atr")): return float("nan")
    return pos["ext"] - pos["side"] * SYS["stop_atr"] * pos["atr"]

def side_entree(d, s=SYS):
    l = d.iloc[-1]; dr = int(l["direction_observee"]); pv = int(d["direction_observee"].iloc[-2])
    if dr == 0 or pv != 0 or not (abs(l["z"]) >= s["z_min"]): return 0
    if s["filtre_contexte"] and not l["contexte_ok"]: return 0
    if s["bloquer_contre_tendance"] and l["align_score"] * dr < 0: return 0
    q = l["quantile_rendement"]
    return dr if ((dr == 1 and q >= s["q_long"]) or (dr == -1 and q <= s["q_short"])) else 0

def motif_sortie(d, pos, s=SYS):
    l = d.iloc[-1]; side = pos["side"]; q, z, dr = l["quantile_rendement"], l["z"], int(l["direction_observee"])
    if s["stop_atr"] and pos.get("atr"):
        stop = niveau_stop(pos)
        if (side == 1 and l["low"] <= stop) or (side == -1 and l["high"] >= stop):
            return "stop ATR suiveur" if (s["suiveur"] and pos["ext"] != pos["ref"]) else "stop ATR initial"
        if s["suiveur"]: pos["ext"] = float(max(pos["ext"], l["high"]) if side == 1 else min(pos["ext"], l["low"]))
    hm = d.index[-1].hour * 60 + d.index[-1].minute; dur = int(st.session_state.params["intervalle"].replace("m", ""))
    if side == 1:
        if q < s["q_sortie"]: return "TP quantile (pression retournée)"
        if z <= -s["z_stop"]: return "stop momentum"
        if dr == -1: return "stop direction"
    else:
        if q > s["q_sortie"]: return "TP quantile (pression retournée)"
        if z >= s["z_stop"]: return "stop momentum"
        if dr == 1: return "stop direction"
    if hm + dur >= s["fin_session_min"]: return "fin de session 17:30"
    return None

def verifier_alerte(d, p, muet=False):
    """Gère position/entrée/sortie sur la dernière barre. État dans st.session_state."""
    ss = st.session_state
    l = d.iloc[-1]; ts = d.index[-1]; t = f"{ts:%H:%M}"
    nom = lambda s: "ACHAT" if s == 1 else "VENTE"
    if ss.get("position") is not None:
        position = ss["position"]; side = position["side"]; m = motif_sortie(d, position)
        pts = side * (l["close"] - position["ref"]); stop = niveau_stop(position)
        if m:
            msg = (f"{t} | clôture {l['close']:.2f}\nCOUPER le {nom(side)} ouvert à {position['heure']}\n"
                   f"Motif : {m}\nRésultat indicatif : {pts:+.1f} pts\n(sortie à l'ouverture de la prochaine barre)")
            ss["bandeau"] = ("#e67e22", f"✖ COUPER le {nom(side)} — {m} — {pts:+.1f} pts (barre {t})")
            journaliser(ts, "SORTIE", nom(side), m, l["close"], f"pts={pts:+.1f}", p)
            ss["position"] = None
            notifier(f"SORTIE {p['symbole']}", msg, muet)
        else:
            stx = f" — stop ATR {stop:.1f}" if not np.isnan(stop) else ""
            ss["bandeau"] = ("#2c3e50", f"● {nom(side)} en cours depuis {position['heure']} ({pts:+.1f} pts){stx} — barre {t}")
        return
    side = side_entree(d)
    if side:
        atr = float(l["atr"]) if pd.notna(l["atr"]) else None
        taille = taille_position(atr, int(l["score_confluence"]), p)
        ss["position"] = dict(side=int(side), ref=float(l["close"]), heure=t, taille=float(taille), atr=atr,
                             ext=float(l["close"]), jour=str(ts.date()), barre=str(ts))
        stop = niveau_stop(ss["position"])
        stx = f"\nStop ATR initial : {stop:.1f}{' (suiveur)' if SYS['suiveur'] else ''}" if not np.isnan(stop) else ""
        msg = (f"{t} | clôture {l['close']:.2f}\n{nom(side)} à l'ouverture de la prochaine barre\n"
               f"Taille : {taille} €/pt (confluence {int(l['score_confluence'])}){stx}\n"
               f"Anomalies : {l['type_anomalie']}\nSorties : quantile {'<' if side == 1 else '>'} {SYS['q_sortie']}, "
               f"z {'≤' if side == 1 else '≥'} {-side * SYS['z_stop']:+.0f}, direction {-side:+d}, stop ATR, ou 17:30")
        ss["bandeau"] = ("#1b8a3a" if side == 1 else "#c0392b", f"🎯 {nom(side)} — taille {taille} €/pt — barre {t}")
        journaliser(ts, "ENTREE", nom(side), l["type_anomalie"], l["close"],
                    f"taille={taille};confluence={int(l['score_confluence'])};tendance={l['signal_tendance']};prob={l['probabilite_directionnelle']}", p)
        notifier(f"ENTRÉE {nom(side)} {p['symbole']}", msg, muet)
        return
    stx, sp = l["signal_tendance"], l["signal_prioritaire"]
    if sp:
        ss["bandeau"] = ("#5b2c6f", f"⚡ {sp} — {stx} — prob. {l['probabilite_directionnelle'] or '—'} — momentum "
                          f"{l['score_momentum'] if pd.notna(l['score_momentum']) else '—'} ({l['momentum']}) — barre {t}")
        journaliser(ts, "TENDANCE", sp, stx, l["close"],
                    f"prob={l['probabilite_directionnelle']};momentum={l['score_momentum']}", p)
        if ALERTE_TENDANCE_PRIORITAIRE:
            notifier(f"TENDANCE {p['symbole']}", f"{t} | {sp}\n{stx} | prob. {l['probabilite_directionnelle'] or '—'} | clôture {l['close']:.2f}", muet)
    elif stx != "ATTENTE":
        ss["bandeau"] = ("#b7950b", f"🟨 {stx} — prob. {l['probabilite_directionnelle'] or '—'} — momentum "
                          f"{l['score_momentum'] if pd.notna(l['score_momentum']) else '—'} ({l['momentum']}) — barre {t}")
    else:
        ss["bandeau"] = None

def traiter_nouvelles_barres(d, p):
    ss = st.session_state
    dur = int(p["intervalle"].replace("m", ""))
    if ss.get("position") and ss["position"].get("jour") != str(d.index[-1].date()):
        ss["position"] = None   # position d'une séance précédente : ignorée
    if ss.get("derniere_traitee") is None: idx = [d.index[-1]]
    else: idx = [t for t in d.index[d.index > ss["derniere_traitee"]] if t.date() == d.index[-1].date()]
    maintenant = pd.Timestamp.now(tz=p["fuseau"])
    for t in idx:
        muet = (maintenant - t) > pd.Timedelta(minutes=3 * dur)
        verifier_alerte(d.loc[:t], p, muet); ss["derniere_traitee"] = t

# ── Graphiques ─────────────────────────────────────────────────────────────
def figure_graphiques(d, p, tab_evt, trades_A, trades_B, n_last=600):
    w = d.iloc[-n_last:]; fig, ax = plt.subplots(4, 2, figsize=(16, 15)); ax = ax.ravel()
    ax[0].plot(w.index, w["close"], lw=1, c="k"); cols = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for k in range(1, 6):
        m = w[f"a{k}_on"]; ax[0].scatter(w.index[m], w["close"][m], s=18, c=cols[k-1], label=NOMS[k-1], zorder=3)
    if "signal_tendance" in w.columns:
        for cfg, coul, mk in [("ACHAT FORT", "#1e8449", "o"), ("ACHAT", "#7dcea0", "o"),
                              ("VENTE", "#f1948a", "v"), ("VENTE FORTE", "#922b21", "v")]:
            m = w["signal_tendance"] == cfg
            if m.any(): ax[0].scatter(w.index[m], w["close"][m], s=45, c=coul, marker=mk, label=f"tend: {cfg}",
                                      zorder=4, edgecolors="k", linewidths=.4)
    ax[0].set_title("Prix, anomalies et configurations de tendance"); ax[0].legend(fontsize=7)
    ax[1].plot(w.index, w["z"], lw=.8); ax[1].plot(w.index, w["z_robuste"], lw=.6, alpha=.6)
    for y in (p["z_seuil"], -p["z_seuil"]): ax[1].axhline(y, c="orange", ls="--")
    for y in (p["z_extreme"], -p["z_extreme"]): ax[1].axhline(y, c="r", ls="--")
    ax[1].set_title("Z-score (classique / robuste) et seuils")
    ax[2].plot(w.index, w["vol_realisee"], lw=1); a2 = ax[2].twinx()
    a2.plot(w.index, w["cusum_pos"], c="r", lw=.7); a2.plot(w.index, -w["cusum_neg"], c="g", lw=.7)
    a2.axhline(p["cusum_h"], c="r", ls=":"); a2.axhline(-p["cusum_h"], c="g", ls=":")
    ax[2].set_title("Volatilité réalisée et CUSUM (+ rouge / − vert)")
    if (d["volume"].fillna(0) > 0).mean() > .5:
        ax[3].bar(w.index, w["volume"].fillna(0), width=0.0005 if est_intraday(p) else 0.8,
                  color=np.where(w["a3_on"], "tab:red", "lightgray")); ax[3].set_title("Volume (pics en rouge)")
    else: ax[3].text(.5, .5, "Volume indisponible", ha="center"); ax[3].set_title("Volume")
    ax[4].plot(w.index, w["ret"], lw=.5, c="gray"); ax[4].plot(w.index, w["q_bas_val"], c="r", lw=.8)
    ax[4].plot(w.index, w["q_haut_val"], c="g", lw=.8)
    ax[4].set_title(f"Rendements et quantiles P{p['q_bas']*100:g} / P{p['q_haut']*100:g} glissants (passé)")
    for nom, tr, coul in (("A retour moyenne", trades_A, "tab:blue"), ("B continuation", trades_B, "tab:orange")):
        if tr is not None and len(tr):
            eq = (1 + tr["ret"]).cumprod(); ax[5].step(tr["exit_time"], eq, label=nom, c=coul)
            ax[6].fill_between(tr["exit_time"], eq / eq.cummax() - 1, step="pre", alpha=.4, color=coul)
    ax[5].set_title("Equity curve (trades séquentiels, net de coûts)"); ax[5].legend(); ax[6].set_title("Drawdown")
    if tab_evt is not None and len(tab_evt):
        hm = horizons(p)[len(horizons(p)) // 2]
        t = tab_evt[(tab_evt.horizon == hm) & tab_evt.anomalie.str.match(r"a\d_")].dropna(subset=["cont_exc_bp"])
        if len(t):
            lab = (t.anomalie.str[3:] + " " + t.direction).values; y = t.cont_exc_bp.values
            ax[7].barh(lab, y, xerr=[y - t.ic_bas_bp.values, t.ic_haut_bp.values - y], color="tab:blue")
            ax[7].axvline(0, c="k"); ax[7].set_title(f"Continuation excédentaire (bp), h={hm}, IC95%")
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════
def porte_acces():
    code_attendu = _secret("CODE_ACCES", None)
    if code_attendu is None:
        st.error("Secret CODE_ACCES manquant : ajoutez-le dans les secrets de l'app "
                 "(Settings → Secrets → CODE_ACCES = \"votre-code\").")
        st.stop()
    ss = st.session_state
    if ss.get("accedant"): return True
    st.markdown("<h1 style='text-align:center'>🔐 Accès protégé</h1>", unsafe_allow_html=True)
    with st.form("acces"):
        saisie = st.text_input("Code d'accès", type="password")
        if st.form_submit_button("Entrer", use_container_width=True):
            if saisie == str(code_attendu):
                ss["accedant"] = True; st.rerun()
            else:
                st.error("Code incorrect.")
    st.stop()

def panneau_params(p):
    with st.sidebar:
        st.header("⚙️ Paramètres")
        p["symbole"] = st.selectbox("Indice", ["^GDAXI", "^FCHI"], index=0 if p["symbole"] == "^GDAXI" else 1,
                                   format_func=lambda s: "DAX 40" if s == "^GDAXI" else "CAC 40")
        p["intervalle"] = st.selectbox("Intervalle", ["5m", "15m", "1d"], index=["5m", "15m", "1d"].index(p["intervalle"]))
        with st.expander("Seuils d'anomalie"):
            p["z_seuil"] = st.slider("Z-score : seuil", 1.0, 4.0, p["z_seuil"], 0.1)
            p["z_extreme"] = st.slider("Z-score : extrême", 2.0, 5.0, p["z_extreme"], 0.1)
            p["vol_seuil"] = st.slider("Volatilité : seuil", 0.5, 4.0, p["vol_seuil"], 0.1)
            p["volume_z_seuil"] = st.slider("Volume : Z seuil", 1.0, 5.0, p["volume_z_seuil"], 0.1)
            p["q_bas"] = st.select_slider("Quantile bas", [0.001, 0.005, 0.01, 0.02, 0.05], value=p["q_bas"])
            p["q_haut"] = 1 - p["q_bas"]
        with st.expander("Coûts & risque"):
            p["cout_bp"] = st.number_input("Coût (bp)", 0.0, 10.0, p["cout_bp"], 0.5)
            p["slippage_bp"] = st.number_input("Slippage (bp/côté)", 0.0, 10.0, p["slippage_bp"], 0.5)
            p["capital"] = st.number_input("Capital (€)", 100, 1_000_000, p["capital"], 100)
            p["risque_pct"] = st.slider("Risque par trade (%)", 0.1, 3.0, p["risque_pct"], 0.1)
        with st.expander("Filtres de contexte"):
            p["filtres_contexte"] = st.checkbox("Activer", p["filtres_contexte"])
            p["heure_min_signal"] = st.text_input("Heure min", p["heure_min_signal"])
            p["heure_max_signal"] = st.text_input("Heure max", p["heure_max_signal"])
            p["filtre_alignement"] = st.selectbox("Alignement", ["declasse", "bloque", "off"],
                                                   index=["declasse", "bloque", "off"].index(p["filtre_alignement"]))
            annonces = st.text_area("Annonces macro (une par ligne, \"AAAA-MM-JJ HH:MM\")",
                                    "\n".join(p["annonces"]))
            p["annonces"] = [a.strip() for a in annonces.splitlines() if a.strip()]
        with st.expander("Système DQZ"):
            SYS["z_min"] = st.slider("Z min entrée", 0.5, 3.0, SYS["z_min"], 0.1)
            SYS["q_long"] = st.slider("Quantile entrée long", 0.5, 0.95, SYS["q_long"], 0.05)
            SYS["q_short"] = round(1 - SYS["q_long"], 2)
            SYS["q_sortie"] = st.slider("Quantile sortie", 0.3, 0.7, SYS["q_sortie"], 0.05)
            SYS["stop_atr"] = st.slider("Stop (ATR)", 0.5, 5.0, SYS["stop_atr"], 0.5)
            SYS["suiveur"] = st.checkbox("Stop suiveur", SYS["suiveur"])
        refresh_min = st.slider("Rafraîchissement auto (min)", 1, 15, max(1, INTERVALLE_REFRESH_S // 60))
        st.caption("⚠ Outil d'analyse : aucun conseil en investissement.")
    return refresh_min

def page_live(p, refresh_min):
    st.header("📡 Suivi live")
    if not st.toggle("▶ Activer le suivi automatique", value=st.session_state.get("live_on", False)):
        st.session_state["live_on"] = False
        st.info("Activez le suivi pour rafraîchir automatiquement et recevoir les alertes (Telegram / e-mail).")
        if st.button("🔄 Analyser maintenant"):
            analyse_et_affiche(p, notifier_muet=False)
        return
    st.session_state["live_on"] = True
    st.caption(f"Rafraîchissement toutes les {refresh_min} min — laissez l'onglet ouvert.")
    fragment_live(refresh_min, p)

def analyse_et_affiche(p, notifier_muet=True):
    ss = st.session_state
    ss["params"] = p
    try:
        df, meta = charger_donnees(p["symbole"], p["intervalle"], p["debut"], p["fin"], p["fuseau"],
                                   p["ignorer_derniere_barre_incomplete"], PARAMS_DEFAUT)
    except Exception as e:
        st.error(f"Échec du téléchargement Yahoo Finance : {e}"); return None
    if ss.get("derniere_barre") != str(meta["fin"]):
        d = detecter_anomalies(df, p, meta["volume_ok"])
        if ss.get("jour_calib") != d.index[-1].date():
            with st.spinner("Calibration des probabilités..."):
                d = calibrer_et_appliquer(d, p, verbose=False)
            ss["jour_calib"] = d.index[-1].date()
        else:
            d = finaliser_tendance(d, p)
        if not notifier_muet:                      # analyse manuelle : on traite la dernière barre
            traiter_nouvelles_barres(d, p)
        ss["d"], ss["meta"], ss["derniere_barre"] = d, meta, str(meta["fin"])
    d, meta = ss.get("d"), ss.get("meta")
    if d is None: st.warning("Pas encore de données."); return None
    affiche_dashboard(d, p, meta)
    return d

def affiche_dashboard(d, p, meta):
    ss = st.session_state
    st.caption(f"{meta['symbole']} {meta['intervalle']} — {meta['n']} barres — {meta['debut']} → {meta['fin']}"
               + ("" if meta["volume_ok"] else " — volume indisponible"))
    b = ss.get("bandeau")
    if b: st.markdown(f"<div style='padding:12px;border-radius:8px;background:{b[0]};"
                      f"color:white;font-weight:bold;text-align:center'>{b[1]}</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    last = d.iloc[-1]
    c1.metric("Clôture", f"{last['close']:.2f}", f"{last['ret']:+.3%}" if pd.notna(last["ret"]) else None)
    c2.metric("Z-score", f"{last['z']:+.2f}", f"robuste {last['z_robuste']:+.2f}")
    c3.metric("Tendance", last["signal_tendance"], last["filtre_motif"] or None)
    c4.metric("Momentum", f"{last['score_momentum'] if pd.notna(last['score_momentum']) else '—'}",
              last["momentum"])
    with st.expander("📋 Détails de la dernière observation", expanded=True):
        lignes = resume_last(d)
        cols = st.columns(3)
        for i, (k, v) in enumerate(lignes.items()):
            cols[i % 3].markdown(f"**{k}** : {v}")
    st.subheader("Tableau de la séance")
    st.dataframe(tableau_tendance(d, jour=d.index[-1].date()), use_container_width=True, height=320)
    st.subheader("Signaux de tendance (non ATTENTE)")
    s = d[d["signal_tendance"] != "ATTENTE"]
    if len(s):
        st.dataframe(s[["signal_tendance", "signal_prioritaire", "probabilite_directionnelle",
                        "score_momentum", "momentum", "confirmation_zscore"]].tail(25),
                     use_container_width=True)
    else:
        st.info("Aucun signal de tendance aujourd'hui.")
    fig = figure_graphiques(d, p, None, None, None, n_last=400)
    st.pyplot(fig)
    logs = ss.get("log_notifications", [])
    if logs:
        with st.expander(f"🔔 Notifications ({len(logs)})"):
            for l in logs[-30:]:
                st.text(l)

def fragment_live(refresh_min, p):
    # st.fragment : run_every en secondes → rafraîchissement automatique du bloc uniquement
    @st.fragment(run_every=refresh_min * 60)
    def _frag():
        analyse_et_affiche(p, notifier_muet=False)
    _frag()

def page_analyse_complete(p):
    st.header("🔬 Analyse complète (backtest + études d'événements)")
    if st.button("Lancer l'analyse complète", type="primary"):
        with st.spinner("Téléchargement + détection + validation... (1 à 3 min)"):
            try:
                df, meta = charger_donnees(p["symbole"], p["intervalle"], p["debut"], p["fin"], p["fuseau"],
                                           p["ignorer_derniere_barre_incomplete"], PARAMS_DEFAUT)
                ok, problemes = controle_fuites(df, p, meta["volume_ok"])
                if ok: st.success("Contrôle des fuites : OK (aucune donnée future utilisée)")
                else: st.error(f"Contrôle des fuites : ÉCHEC — {problemes}")
                d = detecter_anomalies(df, p, meta["volume_ok"])
                d = calibrer_et_appliquer(d, p, verbose=False)
                masques, (t1, t2) = decouper(d, p)
                st.info(f"Découpage : apprentissage < {t1} ≤ validation < {t2} ≤ test")
                c1, c2 = st.columns(2)
                c1.markdown("**Anomalies détectées (barres)**")
                c1.dataframe(d[[f"a{k}_on" for k in range(1, 6)]].sum()
                             .rename(dict(zip([f"a{k}_on" for k in range(1, 6)], NOMS))).to_frame("n_barres"))
                c2.markdown("**Configurations de tendance (finales)**")
                c2.dataframe(d["signal_tendance"].value_counts().to_frame("n_barres"))
                st.subheader("Étude d'événements — découverte (apprentissage + validation)")
                dm = d[masques["apprentissage"] | masques["validation"]]
                t_disc = etude_evenements(dm, {**flags_anomalies(dm), **flags_combinaisons(dm)}, p)
                show = t_disc[t_disc["n"] >= p["n_min_evenements"]]
                st.dataframe(show.sort_values("p_adj").head(25) if len(show) else
                             pd.DataFrame([{"info": "Aucun groupe avec assez d'observations."}]),
                             use_container_width=True)
                st.subheader("Confirmation hors échantillon (test)")
                dt = d[masques["test"]]
                t_test = etude_evenements(dt, flags_anomalies(dt), p)
                tt = t_test[t_test["n"] >= p["n_min_evenements"]]
                st.dataframe(tt if len(tt) else
                             pd.DataFrame([{"info": "Échantillon test trop court (n < seuil)."}]), use_container_width=True)
                st.subheader("Stratégies A/B")
                tabA, trA, _, exA = evaluer_strategie(d, p, "A")
                tabB, trB, _, exB = evaluer_strategie(d, p, "B")
                cA, cB = st.columns(2)
                cA.markdown("**A — Retour à la moyenne**"); cA.dataframe(tabA)
                cB.markdown("**B — Continuation**"); cB.dataframe(tabB)
                st.caption(f"Buy & hold (test) : {exA['bh']:+.2%} | entrées aléatoires (test) : "
                           f"{exA['baseline'][0]:.2f} ± {exA['baseline'][1]:.2f} bp/trade")
                st.subheader("Walk-forward")
                cA, cB = st.columns(2)
                cA.markdown("**A**"); cA.dataframe(walk_forward(d, p, "A"))
                cB.markdown("**B**"); cB.dataframe(walk_forward(d, p, "B"))
                st.subheader("Graphiques")
                st.pyplot(figure_graphiques(d, p, t_disc, trA["tous"], trB["tous"]))
                st.subheader("Décision")
                tab_strat = tabA if p["strategie"] == "A" else tabB
                dec, expl = decision_finale(d, p, tab_strat)
                if dec == "ACHAT": st.success(f">>> {dec} : {expl}")
                elif dec == "VENTE": st.error(f">>> {dec} : {expl}")
                else: st.warning(f">>> {dec} : {expl}")
                st.caption("Rappel : un bon résultat in-sample ou un taux de réussite élevé ne prouve aucune "
                           "rentabilité réelle. Vérifiez coûts, financement, spreads et liquidité de votre instrument.")
                st.session_state["d_complet"] = d
                st.session_state["p_complet"] = p
            except Exception as e:
                st.error(f"Erreur : {e}")
    elif st.session_state.get("d_complet") is not None:
        st.caption("Résultats de la dernière analyse complète affichés ci-dessus (relancez pour actualiser).")

def page_backtest_dqz(p):
    st.header("🤖 Backtest système DQZ (Direction + Quantile + Z-score)")
    if st.button("Lancer le backtest DQZ", type="primary"):
        with st.spinner("Backtest en cours..."):
            try:
                df, meta = charger_donnees(p["symbole"], p["intervalle"], p["debut"], p["fin"], p["fuseau"],
                                           p["ignorer_derniere_barre_incomplete"], PARAMS_DEFAUT)
                d = detecter_anomalies(df, p, meta["volume_ok"])
                d = finaliser_tendance(d, p)
                t = systeme_dqz(d, p, SYS)
                if t.empty:
                    st.info("Aucun trade."); return
                _, (t1, t2) = decouper(d, p)
                t["segment"] = np.where(t.entry_time < t1, "apprentissage",
                                        np.where(t.entry_time < t2, "validation", "test"))
                st.markdown(f"Coûts {p['cout_bp']} bp + slippage {p['slippage_bp']} bp/côté | sizing {SYS['sizing']} | "
                            f"stop ATR {SYS['stop_atr']} ({'suiveur' if SYS['suiveur'] else 'fixe'})")
                stats = pd.DataFrame({**{k: _stats(g) for k, g in t.groupby("segment")}, "TOUS": _stats(t)}).T
                st.dataframe(stats, use_container_width=True)
                for titre, col in [("Par sens", "sens"), ("Par motif de sortie", "sortie_motif"),
                                   ("Par confluence", "confluence"), ("Par régime", "regime")]:
                    st.markdown(f"**{titre}**")
                    st.dataframe(t.groupby(col)["pnl_eur"].agg(n="count", total="sum", moy="mean"))
                st.markdown("**Par séance**")
                st.dataframe(t.groupby(t.entry_time.dt.date)["pnl_eur"].agg(n="count", pnl="sum"))
                fig, ax = plt.subplots(1, 2, figsize=(14, 4))
                ax[0].step(t["exit_time"], t["pnl_eur"].cumsum(), c="tab:blue"); ax[0].axhline(0, c="k", lw=.5)
                ax[0].set_title("Equity (€, net de coûts)")
                ax[1].hist(t["pnl_eur"], bins=30, color="gray"); ax[1].set_title("Distribution des P&L (€)")
                plt.tight_layout(); st.pyplot(fig)
                csv = t.to_csv(sep=";", decimal=",", index=False).encode("utf-8-sig")
                st.download_button("💾 Télécharger les trades (CSV)", csv,
                                   file_name=f"trades_dqz_{p['symbole'].replace('^','')}_{p['intervalle']}.csv",
                                   mime="text/csv")
            except Exception as e:
                st.error(f"Erreur : {e}")

def page_journal(p):
    st.header("🧾 Journal des signaux")
    st.caption("⚠ Sur Streamlit Cloud, le journal est éphémère (redémarrage de l'app = perte). "
               "Pour la persistance, les notifications Telegram/e-mail font foi.")
    j = lire_journal(p)
    if j is None or j.empty:
        st.info("Journal vide.")
        return
    st.dataframe(j, use_container_width=True)
    st.download_button("💾 Télécharger le journal (CSV)",
                       j.to_csv(sep=";", decimal=",", index=False).encode("utf-8-sig"),
                       file_name="journal_signaux.csv", mime="text/csv")

# ── Point d'entrée ─────────────────────────────────────────────────────────
def main():
    porte_acces()
    st.title("📈 Détecteur d'anomalies DAX / CAC 40")
    ss = st.session_state
    ss.setdefault("params", dict(PARAMS_DEFAUT))
    p = ss["params"]
    refresh_min = panneau_params(p)
    onglets = st.tabs(["📡 Suivi live", "🔬 Analyse complète", "🤖 Backtest DQZ", "🧾 Journal"])
    with onglets[0]: page_live(p, refresh_min)
    with onglets[1]: page_analyse_complete(p)
    with onglets[2]: page_backtest_dqz(p)
    with onglets[3]: page_journal(p)

main()