"""
OBSIDIAN — Epaississement selectif (etape A4, ex-"A7") + mesures QA.

Repris de epaissir_selectif.py v2 (valide le 25 oct. 2026 sur A2_S1_2K_1 :
507 zones / 15,1 % encre / trait 0,62 mm / 1,6 % du trace < 0,40 mm).

Principe :
  1. Agrandissement x2 + leger flou (debruitage).
  2. Binarisation (pixel < seuil = encre) ; poussiere < 0,05 mm2 supprimee.
  3. Squelette + largeur locale : seuls les traits plus fins que trait_min
     sont re-traces a trait_min ; les contours deja epais ne bougent pas.
  4. Bords anti-aliases. Sortie a l'echelle de travail (x2) ou a la taille
     d'origine (x1, comportement exact du script valide).

Changement par rapport au script valide : la suppression de poussiere utilise
cv2.connectedComponentsWithStats (connectivite 4, aire <= seuil) au lieu de
skimage.remove_small_objects, dont les parametres changent selon la version.
Le resultat est identique (meme connectivite, meme regle de taille).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize


def _retirer_poussiere(ink: np.ndarray, aire_max_px: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=4)
    garder = st[:, cv2.CC_STAT_AREA] > aire_max_px
    garder[0] = False  # fond
    return garder[lab]


def epaissir(g0: np.ndarray, trait_min=0.40, seuil=200, hauteur_in=11.0, up=2, sortie_x=2) -> np.ndarray:
    """g0 = image en niveaux de gris (uint8). Retourne l'image traitee (uint8)."""
    mm = hauteur_in * 25.4 / g0.shape[0] / up  # mm par pixel a l'echelle de travail

    # 1. agrandissement + debruitage
    g = cv2.resize(g0, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (0, 0), up * 0.7)

    # 2. binarisation + suppression de la poussiere
    ink = g < seuil
    del g
    ink = _retirer_poussiere(ink, int(0.05 / mm / mm))

    # 3. re-trace des seuls traits trop fins
    sk = skeletonize(ink)
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    fins = sk & (dt < trait_min / 2 / mm)  # equivalent a 2*dt*mm < trait_min
    del sk, dt
    d = cv2.distanceTransform((~fins).astype(np.uint8), cv2.DIST_L2, 5)
    del fins
    out = ink | (d <= (trait_min / 2) / mm)
    del ink, d

    # 4. bords anti-aliases
    img = np.where(out, 0, 255).astype(np.uint8)
    del out
    img = cv2.GaussianBlur(img, (0, 0), up * 0.6)
    if sortie_x == 1:
        img = cv2.resize(img, (g0.shape[1], g0.shape[0]), interpolation=cv2.INTER_AREA)
    return img


def mesurer(g: np.ndarray, hauteur_in=11.0, seuil=160) -> dict:
    """Mesures QA (meme methode que mesurer.py) : zones >= 1 mm2, connectivite 4,
    micro-zones < 7 mm2, trait = 2 x transformee de distance sur le squelette."""
    mm = hauteur_in * 25.4 / g.shape[0]
    ink = g < seuil

    blanc = (~ink).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(blanc, connectivity=4)
    aires = np.sort(st[1:, cv2.CC_STAT_AREA] * mm * mm)[:-1]  # retire le fond
    z = aires[aires >= 1]

    sk = skeletonize(ink)
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    w = 2 * dt[sk] * mm

    return {
        "zones": int(len(z)),
        "micro_zones": int((z < 7).sum()),
        "encre_pct": round(float(ink.mean() * 100), 1),
        "trait_moyen_mm": round(float(w.mean()), 2) if w.size else 0.0,
        "trait_fin_pct": round(float((w < 0.40).mean() * 100), 1) if w.size else 0.0,
    }
