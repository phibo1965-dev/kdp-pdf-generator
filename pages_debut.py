"""OBSIDIAN - pages de debut multilingues (v2, nov. 2026).

Polices : fichiers .ttf dans le dossier fonts/ a cote de ce fichier.
Une "famille" = un prefixe de fichier : NotoSerif -> fonts/NotoSerif-Regular.ttf
(+ fonts/NotoSerif-Bold.ttf si present). Les polices sont integrees au PDF.

Pages produites (4) :
  1 (recto) titre      2 (verso) a propos + copyright + depot legal
  3 (recto) "ce livre appartient a" + avis      4 (verso) page d'essai des couleurs
=> le premier dessin tombe toujours en page 5 (recto).
"""
import logging
import re
from pathlib import Path

from reportlab.lib.colors import Color, black
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)
FONTS_DIR = Path(__file__).parent / "fonts"
DEFAULT_TITLE = "NotoSerif"
DEFAULT_TEXT = "NotoSerif"
DEFAULT_FALLBACK = "NotoSans"
GREY = Color(0.35, 0.35, 0.35)


def police_initiale():
    """Police de depart du canvas (evite Helvetica non integree dans le PDF)."""
    ps = f"{DEFAULT_TEXT}-Regular"
    if ps not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(ps, str(FONTS_DIR / f"{ps}.ttf")))
    return ps


class Polices:
    """Enregistre les familles demandees ; repli automatique caractere par ligne."""

    def __init__(self, titre, texte, secours):
        self.avertissements = []
        self.titre = self._famille(titre, DEFAULT_TITLE)
        self.texte = self._famille(texte, DEFAULT_TEXT)
        self.secours = self._famille(secours, DEFAULT_FALLBACK)

    def _famille(self, nom, defaut):
        nom = (nom or "").strip() or defaut
        if not (FONTS_DIR / f"{nom}-Regular.ttf").exists():
            self.avertissements.append(f"police {nom} absente, remplacee par {defaut}")
            nom = defaut
        styles = {}
        for style in ("Regular", "Bold"):
            f = FONTS_DIR / f"{nom}-{style}.ttf"
            if f.exists():
                psname = f"{nom}-{style}"
                if psname not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(psname, str(f)))
                styles[style] = psname
        styles.setdefault("Bold", styles["Regular"])
        return styles

    @staticmethod
    def _couvre(psname, texte):
        cmap = pdfmetrics.getFont(psname).face.charToGlyph
        return all(ord(ch) in cmap for ch in texte if not ch.isspace())

    def choisir(self, famille, texte, gras=False):
        """Police de la famille si elle couvre tout le texte, sinon police de secours."""
        ps = famille["Bold" if gras else "Regular"]
        if self._couvre(ps, texte):
            return ps
        ps2 = self.secours["Bold" if gras else "Regular"]
        if not self._couvre(ps2, texte):
            self.avertissements.append(f"caracteres non couverts : {texte[:30]!r}")
        return ps2

    def utilisees(self):
        return f"{self.titre['Regular']}|{self.texte['Regular']}|{self.secours['Regular']}"


def _centre(c, pol, famille, texte, cx, y, taille, gras=False, couleur=black, larg_max=None):
    """Texte centre, coupe en lignes si besoin. Renvoie le y sous le bloc."""
    ps = pol.choisir(famille, texte, gras)
    lignes = simpleSplit(texte, ps, taille, larg_max) if larg_max else [texte]
    c.setFillColor(couleur)
    for ligne in lignes:
        c.setFont(ps, taille)
        c.drawCentredString(cx, y, ligne)
        y -= taille * 1.35
    c.setFillColor(black)
    return y


def _paragraphes(c, pol, famille, texte, cx, y, taille, larg, couleur=black):
    for para in [p.strip() for p in texte.split("\n")]:
        if not para:
            y -= taille * 0.6
            continue
        y = _centre(c, pol, famille, para, cx, y, taille, couleur=couleur, larg_max=larg)
    return y


def _ornement(c, cx, y, demi):
    c.setLineWidth(0.8)
    c.line(cx - demi, y, cx - 9, y)
    c.line(cx + 9, y, cx + demi, y)
    p = c.beginPath()
    p.moveTo(cx, y + 5); p.lineTo(cx + 5, y); p.lineTo(cx, y - 5); p.lineTo(cx - 5, y); p.close()
    c.drawPath(p, stroke=1, fill=0)


def _mots(*textes):
    return sum(len(re.findall(r"\w+", t or "")) for t in textes)


def generer(c, req, page_w, page_h, zone):
    """Dessine les 4 pages. zone(page_number) -> (x, largeur) de la zone utile.
    Renvoie (nb_pages, pct_mots_langue_principale, polices)."""
    pol = Polices(req.font_title, req.font_text, req.font_fallback)
    L = [dict(l) for l in (req.langues or [])]
    if not L:
        raise ValueError("langues vide : au moins la langue principale est requise")
    P, autres = L[0], L[1:]
    fmt = {"year": req.year, "author": req.author, "collection": req.collection,
           "title": req.title}
    g = lambda d, k: (d.get(k) or "").format(**fmt) if "{" in (d.get(k) or "") else (d.get(k) or "")
    sep = "  ·  "

    # ---------- page 1 : titre (recto)
    x, w = zone(1); cx = x + w / 2
    y = page_h - 1.6 * inch
    _centre(c, pol, pol.titre, req.collection.upper(), cx, y, 16, gras=True)
    _ornement(c, cx, y - 18, w * 0.18)
    y = page_h * 0.60
    y = _centre(c, pol, pol.titre, req.title, cx, y, 34, gras=True, larg_max=w * 0.85)
    if req.subtitle:
        y = _centre(c, pol, pol.texte, req.subtitle, cx, y - 10, 15, larg_max=w * 0.8)
    _ornement(c, cx, y - 14, w * 0.25)
    _centre(c, pol, pol.texte, req.author, cx, 2.0 * inch, 14)
    c.showPage()

    # ---------- page 2 : a propos + copyright (verso)
    x, w = zone(2); cx = x + w / 2
    y = page_h - 2.2 * inch
    if g(P, "about"):
        y = _paragraphes(c, pol, pol.texte, g(P, "about"), cx, y, 11, w * 0.78)
    y = 4.2 * inch
    y = _paragraphes(c, pol, pol.texte, g(P, "copyright"), cx, y, 9, w * 0.8)
    if req.d_number:
        y = _centre(c, pol, pol.texte, f"{P.get('legal_deposit') or 'Legal deposit'}: {req.d_number}",
                    cx, y - 4, 9)
    droits = sep.join(g(l, "rights") for l in autres if g(l, "rights"))
    if droits:
        _centre(c, pol, pol.texte, droits, cx, y - 8, 7.5, couleur=GREY, larg_max=w * 0.85)
    c.showPage()

    # ---------- page 3 : appartient a + avis (recto)
    x, w = zone(3); cx = x + w / 2
    y = page_h - 1.5 * inch
    y = _centre(c, pol, pol.titre, g(P, "belongs"), cx, y, 20, gras=True, larg_max=w * 0.9)
    trad = sep.join(g(l, "belongs") for l in autres if g(l, "belongs"))
    if trad:
        y = _centre(c, pol, pol.texte, trad, cx, y - 2, 8, couleur=GREY, larg_max=w * 0.9)
    c.setLineWidth(0.8)
    for i in range(2):
        yy = y - 0.55 * inch - i * 0.5 * inch
        c.line(cx - w * 0.3, yy, cx + w * 0.3, yy)
    y = y - 1.9 * inch
    _ornement(c, cx, y + 0.35 * inch, w * 0.12)
    y = _centre(c, pol, pol.titre, g(P, "notice_title"), cx, y, 15, gras=True)
    y = _paragraphes(c, pol, pol.texte, g(P, "notice"), cx, y - 4, 10.5, w * 0.8)
    y -= 10
    for l in autres:
        if g(l, "notice"):
            y = _centre(c, pol, pol.texte, f"{l.get('lang', '').upper()} — {g(l, 'notice')}",
                        cx, y, 7.5, couleur=GREY, larg_max=w * 0.88)
            y -= 2
    if y < 0.75 * inch:
        pol.avertissements.append("page 3 trop chargee : raccourcir les avis")
    c.showPage()

    # ---------- page 4 : essai des couleurs (verso)
    x, w = zone(4); cx = x + w / 2
    y = page_h - 1.3 * inch
    y = _centre(c, pol, pol.titre, g(P, "test_title"), cx, y, 18, gras=True)
    trad = sep.join(g(l, "test_title") for l in autres if g(l, "test_title"))
    if trad:
        y = _centre(c, pol, pol.texte, trad, cx, y - 2, 8, couleur=GREY, larg_max=w * 0.9)
    cols, rows, gap = 5, 7, 0.18 * inch
    top, bottom = y - 0.3 * inch, 1.0 * inch
    cell_w = (w * 0.86 - gap * (cols - 1)) / cols
    cell_h = (top - bottom - gap * (rows - 1)) / rows
    x0 = cx - (cell_w * cols + gap * (cols - 1)) / 2
    c.setLineWidth(1.0)
    for r in range(rows):
        for k in range(cols):
            c.roundRect(x0 + k * (cell_w + gap), top - (r + 1) * cell_h - r * gap,
                        cell_w, cell_h, 6, stroke=1, fill=0)
    c.showPage()

    # ---------- part de la langue principale (regle KDP : > 50 %, 60 % conseille)
    champs = ("belongs", "notice_title", "notice", "test_title", "rights", "copyright", "about")
    m_p = _mots(req.title, req.subtitle, *(g(P, k) for k in champs))
    m_a = sum(_mots(*(g(l, k) for k in champs)) for l in autres)
    pct = round(100 * m_p / max(1, m_p + m_a), 1)
    if pct < 60:
        pol.avertissements.append(f"langue principale {pct} % des mots (< 60 %)")
    for a in pol.avertissements:
        logger.warning(f"Pages de debut : {a}")
    return 4, pct, pol
