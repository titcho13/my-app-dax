#!/usr/bin/env python3
# ============================================================================
# dax_live.py — Version headless de la boucle live, pour GitHub Actions.
#
# Prérequis dans le dépôt :
#   - dax_core.py (même dossier) : CELLULES 1 à 11b, sans la CELLULE 12, avec
#     en tête : EN_NOTEBOOK = False / def clear_output(*a, **k): pass
#   - .github/workflows/dax-live.yml
# Secrets GitHub : TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
# ============================================================================
import contextlib, io, json, os, re, sys, time
from datetime import datetime

# Rend dax_core importable quel que soit le répertoire courant
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- Configuration locale ----------------------------------------------------
INTERVALLE_REFRESH_S = 30
HEURE_DEBUT = 9          # 09:00 Paris
HEURE_FIN   = 17.5       # 17:30 Paris
SEUIL_PROBA = 90         # alerte si probabilité directionnelle >= 90 %
FICHIER_ETAT = os.path.join(BASE_DIR, "etat", "etat_live.json")


def _secret(nom):
    return os.environ.get(nom) or ""


def envoyer_telegram(texte):
    tok, chat = _secret("TELEGRAM_TOKEN"), _secret("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        print("⚠ TELEGRAM_TOKEN / TELEGRAM_CHAT_ID absents : alerte non envoyée.")
        return False
    import requests
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": texte}, timeout=10)
    if not r.ok:
        print(f"⚠ Telegram a répondu {r.status_code} : {r.text[:200]}")
    return r.ok


def maintenant_paris():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Europe/Paris"))


def fenetre_marche(tl=None):
    """True si jour ouvré dans la fenêtre [09:00 ; 17:30] Paris."""
    tl = tl or maintenant_paris()
    h = tl.hour + tl.minute / 60
    return tl.weekday() < 5 and HEURE_DEBUT <= h <= HEURE_FIN


def apres_cloture():
    tl = maintenant_paris()
    return tl.weekday() < 5 and (tl.hour + tl.minute / 60) >= HEURE_FIN


# --- État persistant ---------------------------------------------------------
def charger_etat():
    try:
        with open(FICHIER_ETAT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"derniere_barre_alertee": None, "jour_calib": None}


def sauver_etat(etat):
    try:
        os.makedirs(os.path.dirname(FICHIER_ETAT), exist_ok=True)
        with open(FICHIER_ETAT, "w", encoding="utf-8") as f:
            json.dump(etat, f)
        print(f"💾 État sauvegardé : {FICHIER_ETAT}")
    except Exception as e:
        print(f"⚠ Sauvegarde de l'état impossible : {e}")


# --- Alertes -----------------------------------------------------------------
def lire_proba(txt):
    """'LONG 90%' -> 90 ; '' ou None -> None."""
    if not txt or not isinstance(txt, str):
        return None
    m = re.search(r"(\d+)\s*%", txt)
    return int(m.group(1)) if m else None


def construire_message(l, ts, pct):
    mom = f"{l['score_momentum']:.1f} ({l['momentum']})" if l.get("momentum") else "—"
    return (f"🎯 SIGNAL {l.get('signal_tendance', '—')} — probabilité {pct}%\n"
            f"{l.get('signal_prioritaire') or '—'}\n"
            f"{ts:%A %d/%m %H:%M} (Paris) | clôture {l['close']:.2f}\n"
            f"Momentum : {mom} | {l.get('confirmation_zscore', '—')}\n"
            f"Anomalies : {l.get('type_anomalie', '—')} | régime {l.get('regime', '—')}")


def barres_a_alerter(d, etat):
    """Barres du jour postérieures à la dernière barre déjà alertée."""
    aujourdhui = maintenant_paris().date()
    derniere = etat.get("derniere_barre_alertee")
    for ts in d.index:
        if ts.date() != aujourdhui:
            continue
        if derniere and str(ts) <= derniere:   # ISO : l'ordre texte = l'ordre temps
            continue
        yield ts


def boucle_live():
    from dax_core import (PARAMS, charger_donnees, detecter_anomalies,
                          calibrer_et_appliquer)

    etat = charger_etat()
    d = None
    derniere_barre = None
    jour_calib = etat.get("jour_calib")
    print(f"▶ Démarrage {PARAMS['symbole']} {PARAMS['intervalle']} à "
          f"{maintenant_paris():%H:%M:%S} Paris | alertes Telegram : proba >= {SEUIL_PROBA}%")
 envoyer_telegram("✅ Test dax_live depuis GitHub Actions")
    while True:
        tl = maintenant_paris()
        if apres_cloture():
            print("✔ Clôture passée (17:30 Paris) : arrêt propre du job.")
            break
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df_new, meta_new = charger_donnees(PARAMS)
            if derniere_barre is None or meta_new["fin"] != derniere_barre:
                derniere_barre = meta_new["fin"]
                print(f"🔄 {tl:%H:%M:%S} — nouvelle barre {derniere_barre}")
                with contextlib.redirect_stdout(io.StringIO()):
                    d = detecter_anomalies(df_new, PARAMS, meta_new["volume_ok"])
                    if jour_calib is None or str(d.index[-1].date()) != jour_calib:
                        d = calibrer_et_appliquer(d, PARAMS, verbose=False)
                        jour_calib = str(d.index[-1].date())

                if fenetre_marche():
                    for ts in barres_a_alerter(d, etat):
                        l = d.loc[ts]
                        if getattr(l, "ndim", 1) > 1:      # index dupliqué
                            l = l.iloc[-1]
                        pct = lire_proba(l.get("probabilite_directionnelle", ""))
                        if pct is not None and pct >= SEUIL_PROBA:
                            msg = construire_message(l, ts, pct)
                            print(f"💬 {msg}")
                            envoyer_telegram(msg)

                etat["derniere_barre_alertee"] = str(d.index[-1])
                etat["jour_calib"] = jour_calib
                sauver_etat(etat)
            else:
                print(f"⏱ {tl:%H:%M:%S} — vérifié, dernière barre : {derniere_barre}")
        except Exception as e:
            print(f"⚠ {tl:%H:%M:%S} — erreur (nouvel essai au prochain cycle) : {e}")
        time.sleep(INTERVALLE_REFRESH_S)


if __name__ == "__main__":
    boucle_live()
