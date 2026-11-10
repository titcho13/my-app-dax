# ============================================================================
# dax_core.py — Moteur complet DAX/CAC 40, adapté pour GitHub Actions.
# Source : vos CELLULES 1 à 11b (notebook Colab), avec les adaptations :
#   - bloc IPython/Colab remplacé par des stubs headless (aucune dépendance notebook)
#   - CELLULE 12 (boucle live Colab) retirée : remplacée par dax_live.py
#   - matplotlib conservé mais jamais appelé en mode automatique
# Ce fichier s'utilise avec dax_live.py (voir canvas correspondant).
# ============================================================================

# CELLULE 1 — Installation et imports
import subprocess, sys
def _pip(pk): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pk])
for _m, _pk in [("yfinance", "yfinance"), ("statsmodels", "statsmodels"), ("scipy", "scipy")]:
    try: __import__(_m)
    except ImportError: _pip(_pk)

import os, io, json, time, sqlite3, smtplib, contextlib, warnings
from email.message import EmailMessage
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.4g}")

try:                                   # accélération optionnelle (pip install numba)
    from numba import njit
except Exception:
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k: return a[0]
        return lambda f: f

# ── Adaptation headless (remplace le bloc IPython de Colab) ──
EN_NOTEBOOK = False
def clear_output(*a, **k): pass
def display(*a, **k): pass

# ── Outils vectorisés : ligne i = fenêtre du PASSÉ strict x[i-w:i] ──
def _fenetres_passees(x, w):
    x = np.asarray(x, float); n = len(x)
    return sliding_window_view(np.r_[np.full(w, np.nan), x], w)[:n]

def _stat_passee(x, w, minp, f):
    """f(W) -> statistique par ligne (axis=1) ; NaN si moins de minp valeurs dans la fenêtre."""
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
    """Rang (0-1) de x[i] parmi les w valeurs PRÉCÉDENTES."""
    x = np.asarray(x, float); win = _fenetres_passees(x, w)
    valid = ~np.isnan(win); cnt = valid.sum(axis=1); less = ((win < x[:, None]) & valid).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"): out = less / cnt
    out[(cnt < minp) | np.isnan(x)] = np.nan
    return out

def _par_creneau(x, slot, fn):
    """Applique fn (tableau -> tableau de même longueur) séparément à chaque tranche horaire."""
    x = np.asarray(x, float); slot = np.asarray(slot); out = np.full(len(x), np.nan)
    for s in np.unique(slot):
        m = slot == s; out[m] = fn(x[m])
    return out

# CELLULE 2 — Paramètres centralisés
PARAMS = {
    # --- Données ---
    "symbole": "^GDAXI",          # DAX : ^GDAXI | CAC 40 : ^FCHI
    "intervalle": "5m",           # "5m", "15m" ou "1d"
    "debut": None, "fin": None, "fuseau": "Europe/Paris",
    "ignorer_derniere_barre_incomplete": True,
    # --- Anomalie 1 : Z-score ---
    "z_fenetre": 100, "z_seuil": 2.0, "z_extreme": 3.0, "z_base": "classique",  # ou "robuste"
    # --- Anomalie 2 : volatilité / CUSUM ---
    "vol_fenetre": 20, "vol_ref": 100, "vol_seuil": 2.0,
    "cusum_k": 0.5, "cusum_h": 5.0,
    "compression_ratio": 0.7, "expansion_ratio": 1.3,
    # --- Anomalie 3 : volume ---
    "volume_fenetre": 50, "volume_z_seuil": 2.0, "volume_slot_fenetre": 20, "volume_rel_slot_min": 1.5,
    # --- Anomalie 4 : quantiles ---
    "q_fenetre": 250, "q_bas": 0.01, "q_haut": 0.99,
    # --- Anomalie 5 : divergences ---
    "div_fenetre": 10, "div_prix_seuil": 1.0, "div_activite_baisse": 0.8,
    "div_vol_score": 1.0, "div_ralentissement": 0.25,
    # --- Régime ---
    "ema_rapide": 50, "ema_lente": 200, "er_fenetre": 20, "er_tendance": 0.30, "er_range": 0.15,
    "regime_vol_haut": 1.5, "regime_vol_bas": -1.0,
    # --- Stratégies A/B ---
    "strategie": "A",
    "anomalies_strategie": ["a1", "a4"], "anomalie_memoire": 3,
    "breakout_n": 3, "atr_fenetre": 14, "stop_atr": 1.5, "target_atr": 2.0,
    "cible_sma": 20, "tenue_max": 12,
    # --- Coûts (points de base) ---
    "cout_bp": 1.0, "slippage_bp": 0.5,
    # --- Validation ---
    "horizons": None, "n_min_evenements": 20, "alpha": 0.10, "bootstrap": 1000,
    "split": (0.5, 0.25, 0.25), "seed": 42, "rf_annuel": 0.0,
    "exiger_validation_test": True,
    "capital": 10000, "risque_pct": 0.5,
    # --- [3] Normalisation par tranche horaire ---
    "normaliser_creneau": True,
    "creneau_fenetre": 40,
    "creneau_min": 15,
    # --- [4] Filtres de contexte ---
    "filtres_contexte": True,
    "heure_min_signal": "09:15",
    "heure_max_signal": "17:00",
    "annonces": [],
    "annonce_avant_min": 15, "annonce_apres_min": 30,
    "filtre_alignement": "declasse",
    "ema_tendance_h": 20,
    # --- [2] Calibration des probabilités ---
    "calib_horizon": 3, "calib_n_min": 30,
    "graphiques": True,
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

# CELLULE 3 — Récupération Yahoo Finance + vérifications
def charger_donnees(p):
    itv = p["intervalle"]
    if itv not in FREQ: raise ValueError(f"Intervalle non géré : {itv}")
    today = pd.Timestamp.today().normalize()
    fin = pd.Timestamp(p["fin"]) if p["fin"] else today + pd.Timedelta(days=1)
    mj = FREQ[itv]["max_jours"]
    debut = pd.Timestamp(p["debut"]) if p["debut"] else fin - pd.Timedelta(days=mj or 3650)
    if mj and debut < today - pd.Timedelta(days=mj):
        debut = today - pd.Timedelta(days=mj)
        print(f"⚠ Yahoo limite l'intraday {itv} à ~{mj} jours : début ramené au {debut.date()}")
    try:
        raw = yf.download(p["symbole"], start=debut, end=fin, interval=itv,
                          auto_adjust=False, progress=False, threads=False)
    except Exception as e:
        raise RuntimeError(f"Échec du téléchargement Yahoo Finance : {e}")
    if raw is None or raw.empty:
        raise RuntimeError("Aucune donnée reçue (symbole, période ou limite Yahoo ?).")
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated()].sort_index()
    if df.index.tz is None:
        if est_intraday(p): df.index = df.index.tz_localize("UTC").tz_convert(p["fuseau"])
    else:
        df.index = df.index.tz_convert(p["fuseau"])
    df.index.name = "timestamp"
    if p["ignorer_derniere_barre_incomplete"] and len(df) > 1:
        now = pd.Timestamp.now(tz=p["fuseau"]) if df.index.tz is not None else pd.Timestamp.now()
        if est_intraday(p):
            if df.index[-1] + pd.Timedelta(itv.replace("m", "min")) > now: df = df.iloc[:-1]
        elif df.index[-1].normalize() == now.normalize() and now.hour < 18:
            df = df.iloc[:-1]
    vol = df["volume"].fillna(0)
    volume_ok = bool((vol > 0).mean() > 0.5 and vol.std() > 0)
    meta = dict(symbole=p["symbole"], intervalle=itv, debut=df.index[0], fin=df.index[-1],
                n=len(df), volume_ok=volume_ok)
    print(f"Données {p['symbole']} {itv} : {meta['n']} barres, {meta['debut']} → {meta['fin']}")
    if not volume_ok:
        print("⚠ Volume absent ou inutilisable : anomalie 3 désactivée ; anomalie 5 en variante prix + volatilité.")
    if len(df) < 400: print("⚠ Historique court : les fenêtres glissantes laisseront peu d'observations exploitables.")
    return df, meta

def preparer(df, p):
    d = df.copy()
    d["ret"] = np.log(d["close"] / d["close"].shift(1))
    if est_intraday(p):  # 1re barre de séance : rendement ouverture→clôture
        jour = pd.Series(d.index.date, index=d.index)
        prem = (jour != jour.shift(1)).values
        d.loc[prem, "ret"] = np.log(d["close"] / d["open"])[prem]
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(p["atr_fenetre"]).mean()
    return d

# CELLULE 4 — Anomalies 1 et 2 : Z-score, volatilité / CUSUM
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
    sp, sm, st = _cusum(x, float(p["cusum_k"]), float(p["cusum_h"]))
    typ = np.select([st == 1, (score >= p["vol_seuil"]) & (st != 1), (st == -1) | (score <= -p["vol_seuil"])],
                    ["augmentation_persistante", "expansion_soudaine", "baisse_inhabituelle"], "aucune")
    phase = np.select([ratio <= p["compression_ratio"], ratio >= p["expansion_ratio"]],
                      ["compression", "expansion"], "neutre")
    out = pd.DataFrame(index=d.index)
    out["vol_realisee"], out["vol_score"], out["vol_ratio"] = vol, score, ratio
    out["cusum_pos"], out["cusum_neg"], out["etat_cusum"] = sp, sm, st
    out["phase_vol"], out["anomalie_vol"] = phase, typ
    out["a2_on"] = out["anomalie_vol"] != "aucune"
    out["a2_dir"] = np.sign(r).where(out["a2_on"], 0).fillna(0).astype(int)
    return out

# CELLULE 5 — Anomalies 3, 4 et 5 : volume, quantiles, divergences
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
    if est_intraday(p):  # référence par tranche horaire (passé uniquement)
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

# CELLULE 5b — Configurations de tendance + calibration + contexte
TEND = dict(
    q_achat=0.90, q_achat_fort=0.95, q_vente=0.10, q_vente_fort=0.05,
    q_surv_long=(0.80, 0.90), q_surv_short=(0.10, 0.20),
    bonus_vol="expansion_soudaine", bonus_conf=10, conf_max=95,
    z_conf_forte=1.0, z_exceptionnel=2.0, z_clip=3.0,
    poids_quantile=0.8, poids_zscore=0.2,
    calib=None,
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
        ["LONG PRIORITAIRE RENFORCE", "LONG PRIORITAIRE", "SHORT PRIORITAIRE RENFORCE"], default="")
    d["signal_prioritaire"] = np.where(negatif & (q <= t["q_vente"]) & (d["signal_prioritaire"] == ""),
                                       "SHORT PRIORITAIRE", d["signal_prioritaire"])

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
        ["Exceptionnel : Z > 2", "Exceptionnel : Z < -2", "Confirmation forte (Z > 1)", "Confirmation forte (Z < -1)"],
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

# ---- [2] Calibration des probabilités ----
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
        print(f"\n=== Calibration des probabilités (horizon {p['calib_horizon']} barres) ===")
        print(tab if len(tab) else "Pas assez d'événements.")
        tt, _ = calibrer_probas(d, p, masques["test"])
        print("\nContrôle hors échantillon (test) :"); print(tt if len(tt) else "Test trop court.")
        print("→ % utilisés :", calib if calib else "aucun (n insuffisant : valeurs manuelles conservées)")
    TEND["calib"] = calib or None
    return finaliser_tendance(d, p)

# CELLULE 6 — Régime, confluence, signaux et moteur commun
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
    ok = True
    for c in cols:
        a, b = d_full[c].iloc[:k].values.astype(float), d_cut[c].values.astype(float)
        if not np.allclose(a, b, equal_nan=True, atol=1e-9): print(f"✗ fuite possible : {c}"); ok = False
    print("✓ Contrôle des fuites : OK" if ok else "✗ Contrôle des fuites : ÉCHEC")
    return ok

# CELLULE 7 — Étude d'événements
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
    print(f"Synthèse des événements (horizon {h} barres, non chevauchants) : {len(r)} événements")
    return out

# CELLULE 8 — Stratégies A/B : simulation, métriques, validation
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
    if verbose:
        print(f"\n=== Stratégie {strat} ({'retour à la moyenne' if strat=='A' else 'continuation'}) — coûts aller-retour "
              f"{p['cout_bp']} bp + slippage {p['slippage_bp']} bp/côté ===")
        print(tab)
        if len(tous):
            print("\nLongs / Shorts :"); print(tous.groupby("dir")["ret"].agg(n="count", moy_bp=lambda x: x.mean()*1e4,
                                                                             reussite=lambda x: (x > 0).mean()))
            print("\nPar régime :"); print(tous.groupby("regime")["ret"].agg(n="count", moy_bp=lambda x: x.mean()*1e4))
            print("\nPar année :"); print(tous.groupby(tous["entry_time"].dt.year)["ret"].agg(n="count", moy_bp=lambda x: x.mean()*1e4))
        mt = masques["test"]; bh = d["close"][mt].iloc[-1] / d["close"][mt].iloc[0] - 1
        print(f"\nRéférence buy & hold (test) : {bh:+.2%} | drawdown {_maxdd(d['close'][mt].pct_change().dropna().values):.2%}")
        bm, bs = baseline_aleatoire(d, p, tab.loc["test", "n_signaux"], mt)
        if not np.isnan(bm): print(f"Référence entrées aléatoires (test) : espérance {bm:.2f} ± {bs:.2f} bp/trade "
                                   f"vs stratégie {tab.loc['test','esperance_nette_bp']:.2f} bp")
    return tab, trades, sig

def walk_forward(d, p, strat, k=5):
    sig = generer_signaux(d, p, strat); edges = np.linspace(0, len(d), k + 1, dtype=int); rows = []
    for f in range(k):
        m = np.zeros(len(d), bool); m[edges[f]:edges[f + 1]] = True
        tr = simuler(d, sig, p, m); rows.append({"bloc": f + 1, "debut": d.index[edges[f]].date(),
                                                 **metriques(tr, p, d.index[edges[f]], d.index[edges[f+1]-1])})
    return pd.DataFrame(rows).set_index("bloc")

def sensibilite(df, p, volume_ok, strat, grille_z=(1.5, 2.0, 2.5, 3.0), grille_stop=(1.0, 1.5, 2.0)):
    rows = []
    for z in grille_z:
        for s in grille_stop:
            q = {**p, "z_seuil": z, "stop_atr": s, "strategie": strat}
            d = detecter_anomalies(df, q, volume_ok, tendance=False)
            tab, _, _ = evaluer_strategie(d, q, strat, verbose=False)
            rows.append(dict(z_seuil=z, stop_atr=s, n_app=tab.loc["apprentissage", "n_trades"], esp_app_bp=tab.loc["apprentissage", "esperance_nette_bp"],
                             n_val=tab.loc["validation", "n_trades"], esp_val_bp=tab.loc["validation", "esperance_nette_bp"],
                             esp_test_bp=tab.loc["test", "esperance_nette_bp"], n_test=tab.loc["test", "n_trades"]))
    t = pd.DataFrame(rows); ok = t[t["n_val"] >= 5]
    best = ok.sort_values("esp_val_bp", ascending=False).head(1) if len(ok) else t.head(0)
    print("Sensibilité (choix sur VALIDATION) :")
    print(t.drop(columns="esp_test_bp"))
    if len(best): print("\nJeu retenu sur validation, évalué sur test :"); print(best[["z_seuil", "stop_atr", "esp_val_bp", "esp_test_bp", "n_test"]])
    return t

# CELLULE 10 — Pipeline complet (utile pour analyse manuelle hors boucle live)
def tableau_de_bord(d, p, meta, tab_evt, tab_strat):
    last = d.iloc[-1]; hs = horizons(p); hm = hs[len(hs) // 2]; dec, expl = decision_finale(d, p, tab_strat)
    print("=" * 78); print(f"DERNIÈRE OBSERVATION : {d.index[-1]}  |  {meta['symbole']} {meta['intervalle']}  |  clôture {last['close']:.2f}")
    print(f"Régime : {last['regime']} / {last['regime_vol']}   |   confluence : {int(last['score_confluence'])}/4 familles")
    print(f"Tendance : {last['signal_tendance']} (brut {last['signal_tendance_brut']}, {last['filtre_motif'] or 'aucun filtre'})   |   "
          f"probabilité : {last['probabilite_directionnelle'] or '—'}   |   momentum : {last['score_momentum'] if pd.notna(last['score_momentum']) else '—'} ({last['momentum']})")
    print("-" * 78)
    actives = [(k, NOMS[k-1]) for k in range(1, 6) if last[f"a{k}_on"]]
    if not actives: print("  aucune anomalie active")
    for k, nom in actives:
        print(f"  • {nom} | mouvement observé {int(last[f'a{k}_dir']):+d}")
    print("-" * 78); print(f">>> DÉCISION : {dec}"); print(f"    {expl}"); print("=" * 78)
    return dec

def decision_finale(d, p, tab_strat):
    last = d.iloc[-1]; s = int(last["sig"])
    if s == 0: return "ATTENTE", "Règles de la stratégie non satisfaites."
    if p["exiger_validation_test"]:
        t = tab_strat.loc["test"]
        if t["n_trades"] < 5 or not (t["esperance_nette_bp"] > 0):
            return "ATTENTE", (f"Signal {'ACHAT' if s>0 else 'VENTE'} présent mais avantage non démontré hors échantillon "
                               f"(test : {int(t['n_trades'])} trades, espérance {t['esperance_nette_bp']:.2f} bp).")
    return ("ACHAT" if s > 0 else "VENTE"), last["motif_signal"]

def executer(p, controle=True, donnees=None, complet=True):
    df, meta = donnees if donnees is not None else charger_donnees(p)
    if controle: controle_fuites(df, p, meta["volume_ok"])
    d = detecter_anomalies(df, p, meta["volume_ok"])
    d = calibrer_et_appliquer(d, p, verbose=complet)
    masques, (t1, t2) = decouper(d, p)
    print(f"\nDécoupage chronologique : apprentissage < {t1} ≤ validation < {t2} ≤ test")
    print("\nAnomalies détectées :"); print(d[[f"a{k}_on" for k in range(1, 6)]].sum().rename(dict(zip([f"a{k}_on" for k in range(1, 6)], NOMS))).to_frame("n_barres"))
    print("\nConfigurations de tendance (finales) :"); print(d["signal_tendance"].value_counts().to_frame("n_barres"))
    if not complet: return d, meta, None, None
    print("\n=== Étude d'événements — DÉCOUVERTE ===")
    dm = d[masques["apprentissage"] | masques["validation"]]
    t_disc = etude_evenements(dm, {**flags_anomalies(dm), **flags_combinaisons(dm)}, p)
    show = t_disc[t_disc["n"] >= p["n_min_evenements"]]
    print(show.sort_values("p_adj").head(25) if len(show) else "Aucun groupe d'événements avec assez d'observations.")
    print("\n=== Étude d'événements — CONFIRMATION HORS ÉCHANTILLON (test) ===")
    dt = d[masques["test"]]; t_test = etude_evenements(dt, flags_anomalies(dt), p)
    print(t_test[t_test["n"] >= p["n_min_evenements"]] if (t_test["n"] >= p["n_min_evenements"]).any() else "Échantillon test trop court.")
    print("\n=== Synthèse isolées / simultanées / suivies d'un mouvement ==="); print(synthese_evenements(d, p))
    tabA, trA, _ = evaluer_strategie(d, p, "A"); tabB, trB, _ = evaluer_strategie(d, p, "B")
    tab_strat = tabA if p["strategie"] == "A" else tabB
    tableau_de_bord(d, p, meta, t_disc, tab_strat)
    print("\nRappel : un bon résultat in-sample ou un taux de réussite élevé ne prouve aucune rentabilité réelle.")
    return d, meta, t_disc, t_test

# CELLULE 11 — TESTS SYNTHÉTIQUES
def donnees_synthetiques(n=3000, seed=0):
    rng = np.random.default_rng(seed); r = rng.normal(0, 0.0008, n); r[1500:] *= 2.5
    r[800] = -0.012; r[2000] = 0.015
    c = 100 * np.exp(np.cumsum(r)); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0003, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    v = rng.lognormal(10, 0.3, n); v[1200] *= 10
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="15min", tz="Europe/Paris")
    return pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=v), index=idx)

def tests_synthetiques():
    print("[SYNTHÉTIQUE] Tests unitaires sur données artificielles")
    p = {**PARAMS, "intervalle": "15m", "filtres_contexte": False, "normaliser_creneau": False}
    anc = TEND["calib"]; TEND["calib"] = None
    df = donnees_synthetiques(); d = detecter_anomalies(df, p, True)
    x = np.random.default_rng(1).normal(size=500); w = 50
    ref = pd.Series(x).rolling(w + 1).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True).values
    assert np.allclose(_rang_passe(x, w, w), ref, equal_nan=True), "_rang_passe diverge de rolling.apply"
    assert d["a1_on"].iloc[800] and d["a1_dir"].iloc[800] == -1, "Z-score : choc baissier non détecté"
    assert d["a1_on"].iloc[2000] and d["a1_dir"].iloc[2000] == 1, "Z-score : choc haussier non détecté"
    assert (d["etat_cusum"].iloc[1500:1560] == 1).any(), "CUSUM : changement de volatilité non détecté"
    assert d["a3_on"].iloc[1200], "Volume : pic non détecté"
    assert d["a4_on"].iloc[800] and d["a4_dir"].iloc[800] == -1, "Quantiles : excès baissier non détecté"
    assert d["signal_tendance"].iloc[2000] in ("ACHAT", "ACHAT FORT", "SURVEILLANCE LONG"), "Tendance : choc haussier non classé LONG"
    assert d["signal_tendance"].iloc[800] in ("VENTE", "VENTE FORTE", "SURVEILLANCE SHORT"), "Tendance : choc baissier non classé SHORT"
    assert controle_fuites(df, p, True), "Fuite de données futures"
    dn = detecter_anomalies(df.assign(volume=0.0), p, False)
    assert (dn["anomalie_volume"] == "indisponible").all() and not dn["a3_on"].any(), "Volume absent mal géré"
    p2 = {**p, "normaliser_creneau": True, "filtres_contexte": True, "annonces": ["2024-01-20 14:30"]}
    d2 = detecter_anomalies(df, p2, True)
    assert {"contexte_ok", "align_score", "signal_tendance_brut", "filtre_motif"} <= set(d2.columns), "Colonnes de contexte absentes"
    assert not d2.loc[d2.index.strftime("%H:%M") < "09:15", "contexte_ok"].any(), "Filtre horaire inactif"
    assert controle_fuites(df, p2, True), "Fuite de données futures (mode tranche horaire)"
    tab, calib = calibrer_probas(d, p); assert isinstance(calib, dict)
    TEND["calib"] = anc
    print(f"✓ Tous les tests synthétiques passent | faux positifs Z≥2 hors chocs : {d['a1_on'].iloc[200:].mean():.2%}")
    return d

# CELLULE 11b — Système « Direction + Quantile + Z-score »
SYS = dict(z_min=1.0, q_long=0.6, q_short=0.4, q_sortie=0.5, z_stop=1.0,
           fin_session_min=17 * 60 + 30,
           taille_base=0.5, taille_conf=1.0, conf_min=2,
           sizing="atr",
           taille_conf_mult=2.0,
           taille_min=0.1, taille_max=3.0,
           stop_atr=2.0,
           suiveur=True,
           filtre_contexte=True,
           bloquer_contre_tendance=False)

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

def backtest_dqz(d, p, s=SYS):
    t = systeme_dqz(d, p, s)
    if t.empty: print("Aucun trade."); return t
    _, (t1, t2) = decouper(d, p)
    t["segment"] = np.where(t.entry_time < t1, "apprentissage", np.where(t.entry_time < t2, "validation", "test"))
    print(f"=== Système Direction+Quantile+Z — {p['symbole']} {p['intervalle']} ===")
    print(pd.DataFrame({**{k: _stats(g) for k, g in t.groupby("segment")}, "TOUS": _stats(t)}).T)
    print("\nPar motif de sortie :"); print(t.groupby("sortie_motif")["pnl_eur"].agg(n="count", total="sum", moy="mean"))
    return t

def signal_live(d, p=None, s=SYS):
    p = p or PARAMS
    l, dr, pv = d.iloc[-1], int(d["direction_observee"].iloc[-1]), int(d["direction_observee"].iloc[-2])
    if dr == 0 or pv != 0: return "Pas d'entrée (pas de 1re occurrence)."
    if s["filtre_contexte"] and not l["contexte_ok"]: return f"Direction {dr:+d} ignorée : {l['filtre_motif'] or 'hors plage / annonce'}."
    ok_z = abs(l["z"]) >= s["z_min"]
    ok_q = (l["quantile_rendement"] >= s["q_long"]) if dr == 1 else (l["quantile_rendement"] <= s["q_short"])
    if not (ok_z and ok_q):
        return f"Direction {dr:+d} refusée : |z|={abs(l['z']):.2f} (≥{s['z_min']}), quantile={l['quantile_rendement']:.2f}."
    atr = float(l["atr"]) if pd.notna(l["atr"]) else None; taille = taille_position(atr, int(l["score_confluence"]), p, s)
    stop = f" | stop ATR {l['close'] - dr * s['stop_atr'] * atr:.1f}" if (s["stop_atr"] and atr) else ""
    return (f"{'ACHAT' if dr == 1 else 'VENTE'} à l'ouverture de la prochaine barre | taille {taille} €/pt{stop} | "
            f"sorties : quantile {'<' if dr == 1 else '>'} {s['q_sortie']}, z {'≤' if dr == 1 else '≥'} {-dr * s['z_stop']:+.0f}, "
            f"direction {-dr:+d}, stop ATR suiveur, ou 17:30")
