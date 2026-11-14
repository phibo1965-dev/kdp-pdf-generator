"""
Service FastAPI de génération de PDFs pour livres de coloriage KDP
Formats supportés: 5x8, 6x9, 8x10, 8.5x11 inches
v3.9.0 (14 nov. 2026) : code mort retire (/svg_to_png, cairosvg, anciennes pages Helvetica)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel, HttpUrl
from typing import List
import requests
import os
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import inch
from PIL import Image
import io
from io import BytesIO
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import gc
import resource
import numpy as np
import cv2
from thicken import epaissir, mesurer
import pages_debut
import datetime as _dt

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KDP PDF Generator",
    description="Service de génération de PDFs pour livres de coloriage Amazon KDP (OBSIDIAN)",
    version="3.9.0"
)

# Formats KDP en inches → points (1 inch = 72 points) pour PDF
KDP_FORMATS = {
    "5x8": (5 * inch, 8 * inch),
    "6x9": (6 * inch, 9 * inch),
    "8x10": (8 * inch, 10 * inch),
    "8.5x11": (8.5 * inch, 11 * inch)
}

# Telechargement au fil de l'eau (v3.8.0, 14 nov. 2026) : au plus PREFETCH images
# en memoire en meme temps (avant : les 43 d'un coup -> depassement des 512 Mo Render)
PREFETCH = 2


# =============================================================================
# MODÈLES DE REQUÊTE
# =============================================================================

class PDFRequest(BaseModel):
    """Modèle de requête pour la génération de PDF"""
    urls: List[HttpUrl]
    book_id: str
    format: str = "8.5x11"
    title: str = ""
    author: str = ""                  # obligatoire (Collection_DNA M AUTEUR, lu par A8)
    # --- Mise en page KDP (OBSIDIAN, nov. 2026) ---
    recto_only: bool = True           # page blanche apres chaque dessin (feutres)
    hauteur_dessin_in: float = 10.0   # hauteur imprimee du dessin = hauteur_in de /thicken
    marge_ext_in: float = 0.375       # marge exterieure (KDP mini 0,25 sans fond perdu)
    marge_haut_in: float = 0.375      # marge du haut
    numeroter: bool = True            # numero de page sous le dessin, hors du dessin
    dpi_impression: int = 300         # resolution integree au PDF (KDP : 300 ppp)
    niveaux_de_gris: bool = True      # dessins integres en gris (3x plus leger que RGB)
    # --- Pages de debut v2 multilingues (OBSIDIAN, 13 nov. 2026) ---
    # obligatoire : 4 pages v2 (titre, copyright, appartient+avis, essai couleurs).
    # langues[0] = langue principale (anglais).
    langues: List[dict] = []
    collection: str = "OBSIDIAN"
    subtitle: str = ""
    year: str = ""
    d_number: str = ""
    font_title: str = ""      # famille dans fonts/ (ex. CinzelDecorative) ; vide = NotoSerif
    font_text: str = ""       # vide = NotoSerif
    font_fallback: str = ""   # vide = NotoSans (langues non couvertes)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def telecharger_image(url: str, index: int, timeout: int = 60) -> bytes:
    """Telecharge une image (synchrone, appele depuis un fil du pool)."""
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
    except requests.HTTPError:
        raise HTTPException(400, f"Image {index + 1}: impossible de telecharger (HTTP {r.status_code}): {url}")
    except requests.Timeout:
        raise HTTPException(408, f"Image {index + 1}: delai depasse au telechargement: {url}")
    except requests.RequestException as e:
        raise HTTPException(400, f"Image {index + 1}: erreur de telechargement: {e}")
    ct = r.headers.get("content-type", "")
    if ct.startswith("text/html"):
        raise HTTPException(400, f"Image {index + 1}: le lien renvoie une page web, pas une image "
                                 f"(fichier Drive non partage ou ID faux): {url}")
    return r.content


def memoire_max_mo() -> int:
    """Pic de memoire du processus depuis son demarrage (Linux : ru_maxrss en Ko)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def gouttiere_kdp(nb_pages: int) -> float:
    """Marge interieure KDP (pouces) selon le nombre total de pages."""
    if nb_pages <= 150:
        return 0.375
    if nb_pages <= 300:
        return 0.5
    if nb_pages <= 500:
        return 0.625
    if nb_pages <= 700:
        return 0.75
    return 0.875


def create_pdf(urls: List[str], request: PDFRequest):
    """Cree le PDF en telechargeant les images une a une (memoire basse).
    Renvoie (pdf_bytes, info)."""
    pool = ThreadPoolExecutor(max_workers=PREFETCH)
    try:
        format_name = request.format
        if format_name not in KDP_FORMATS:
            raise ValueError(
                f"Format non supporté: {format_name}. "
                f"Formats disponibles: {', '.join(KDP_FORMATS.keys())}"
            )
        
        page_width, page_height = KDP_FORMATS[format_name]
        logger.info(f"Création PDF format {format_name}: {page_width}x{page_height} points")
        
        pdf_buffer = io.BytesIO()
        init_font = pages_debut.police_initiale()
        c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height), initialFontName=init_font)
        
        page_number = 1
        # ---- pages de debut v2 : 4 pages, polices integrees
        nb_img = len(urls)
        nb_total = 4 + (2 * nb_img - 1 if request.recto_only else nb_img)
        g_v2 = gouttiere_kdp(nb_total) * inch
        ext_v2 = request.marge_ext_in * inch
        def zone(n):
            if n % 2 == 1:
                return g_v2, page_width - g_v2 - ext_v2
            return ext_v2, page_width - g_v2 - ext_v2
        if not request.year:
            request.year = str(_dt.date.today().year)
        nb, pct, pol = pages_debut.generer(c, request, page_width, page_height, zone)
        page_number += nb
        police_num = pol.texte["Regular"]
        info = {"pct": str(pct), "polices": pol.utilisees(),
                "avert": " ; ".join(pol.avertissements)}

        # Images de coloriage
        nb_front = page_number - 1
        nb_img = len(urls)
        # premier dessin sur une page impaire (recto) : page blanche si besoin
        pad = 1 if (request.recto_only and page_number % 2 == 0) else 0
        nb_pages = nb_front + pad + (2 * nb_img - 1 if request.recto_only else nb_img)
        gout = gouttiere_kdp(nb_pages) * inch
        ext = request.marge_ext_in * inch
        haut = request.marge_haut_in * inch
        h_cible = request.hauteur_dessin_in * inch
        bas = page_height - haut - h_cible
        if bas < 0.25 * inch + (0.3 * inch if request.numeroter else 0):
            raise ValueError(f"hauteur_dessin_in trop grande : marge du bas {bas / inch:.2f} in")
        logger.info(f"Mise en page : {nb_pages} pages, gouttiere {gout / inch} in, "
                    f"dessin {request.hauteur_dessin_in} in de haut, recto seul={request.recto_only}")
        if pad:
            c.showPage()
            page_number += 1

        # file d'attente : PREFETCH telechargements d'avance, pas plus
        futs = deque()
        suivant = [0]
        def lancer():
            if suivant[0] < nb_img:
                futs.append(pool.submit(telecharger_image, str(urls[suivant[0]]), suivant[0]))
                suivant[0] += 1
        for _ in range(PREFETCH):
            lancer()

        for idx in range(nb_img):
            logger.info(f"Traitement de l'image {idx + 1}/{nb_img}")
            data = futs.popleft().result()
            lancer()
            img = Image.open(io.BytesIO(data))
            del data

            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1])
                img = background
            if request.niveaux_de_gris and img.mode != 'L':
                img = img.convert('L')
            elif img.mode not in ('L', 'RGB'):
                img = img.convert('RGB')

            # page impaire = recto : reliure a gauche ; page paire : reliure a droite
            if page_number % 2 == 1:
                zone_x, zone_w = gout, page_width - gout - ext
            else:
                zone_x, zone_w = ext, page_width - gout - ext

            ratio = img.width / img.height
            draw_height = h_cible
            draw_width = draw_height * ratio
            if draw_width > zone_w:  # image trop large : on reduit (signale)
                logger.warning(f"Image {idx + 1} trop large, reduite a la largeur utile")
                draw_width = zone_w
                draw_height = draw_width / ratio
            # ramener a dpi_impression a la taille imprimee (jamais d'agrandissement)
            cible_h = round(draw_height / inch * request.dpi_impression)
            if img.height > cible_h * 1.02:
                cible_w = round(img.width * cible_h / img.height)
                img = img.resize((cible_w, cible_h), Image.LANCZOS)
            x = zone_x + (zone_w - draw_width) / 2
            y = bas + (h_cible - draw_height)  # dessin colle a la marge du haut

            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=95)
            del img
            img_buffer.seek(0)
            c.drawImage(ImageReader(img_buffer), x, y, width=draw_width, height=draw_height)
            del img_buffer
            gc.collect()

            if request.numeroter:
                c.setFont(police_num, 10)
                c.drawCentredString(zone_x + zone_w / 2, 0.35 * inch, str(page_number))

            if idx < nb_img - 1:
                c.showPage()
                page_number += 1
                if request.recto_only:  # verso blanc
                    c.showPage()
                    page_number += 1

        c.save()
        
        pdf_bytes = pdf_buffer.getvalue()
        logger.info(f"PDF créé avec succès: {len(pdf_bytes)} bytes, {page_number} pages")
        
        return pdf_bytes, info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la création du PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la création du PDF: {str(e)}"
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Endpoint racine - Informations sur l'API"""
    return {
        "service": "KDP PDF Generator",
        "version": "3.9.0",
        "status": "operational",
        "features": ["Pages de debut v2 multilingues", "Memoire basse", "Epaississement /thicken"],
        "formats_supported": list(KDP_FORMATS.keys()),
        "endpoints": {
            "health": "/health",
            "generate": "/generate-pdf (POST)",
            "thicken": "/thicken (POST, multipart)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "KDP PDF Generator",
        "version": "3.9.0",
        "formats_available": list(KDP_FORMATS.keys())
    }


@app.post("/generate-pdf")
async def generate_pdf(request: PDFRequest):
    """
    Génère un PDF à partir d'une liste d'URLs d'images
    Version optimisée avec téléchargements parallèles
    
    Args:
        request: PDFRequest contenant les URLs et paramètres
        
    Returns:
        Response: Le PDF en données binaires
    """
    try:
        logger.info(f"═══════════════════════════════════════════════")
        logger.info(f"Début de génération PDF: {request.book_id}")
        logger.info(f"Format: {request.format}")
        logger.info(f"Nombre d'images: {len(request.urls)}")
        logger.info(f"Auteur: {request.author} ; langues: {len(request.langues)}")
        logger.info(f"═══════════════════════════════════════════════")
        
        # Validations
        if not request.urls:
            raise HTTPException(status_code=400, detail="La liste d'URLs ne peut pas être vide")
        
        if not request.langues:
            raise HTTPException(status_code=400, detail="'langues' obligatoire (pages de debut v2) : TEXTES_PAGES K2 vide ?")
        if not request.author.strip():
            raise HTTPException(status_code=400, detail="'author' vide : remplir Collection_DNA M (AUTEUR)")

        if len(request.urls) > 500:
            raise HTTPException(status_code=400, detail="Maximum 500 images par PDF")
        
        if request.format not in KDP_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté: {request.format}. "
                       f"Formats disponibles: {', '.join(KDP_FORMATS.keys())}"
            )
        
        # Telechargement au fil de l'eau + creation du PDF (dans un fil, sans bloquer le serveur)
        import time
        start_time = time.time()
        loop = asyncio.get_running_loop()
        pdf_bytes, info = await loop.run_in_executor(None, create_pdf, request.urls, request)
        total_time = time.time() - start_time

        logger.info(f"═══════════════════════════════════════════════")
        logger.info(f"✓ PDF généré avec succès: {request.book_id}")
        logger.info(f"  Temps TOTAL: {total_time:.2f}s ; memoire max {memoire_max_mo()} Mo")
        logger.info(f"  Taille PDF: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
        logger.info(f"═══════════════════════════════════════════════")
        
        return Response(
            content=pdf_bytes,
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{request.book_id}.pdf"',
                'X-PDF-Recto-Seul': str(request.recto_only),
                'X-PDF-Hauteur-Dessin-In': str(request.hauteur_dessin_in),
                'X-PDF-DPI': str(request.dpi_impression),
                'X-PDF-Duree-s': f"{total_time:.1f}",
                'X-PDF-Memoire-Max-Mo': str(memoire_max_mo()),
                'X-PDF-Langue-Principale-Pct': info["pct"],
                'X-PDF-Polices': info["polices"],
                'X-PDF-Avertissements': info["avert"].encode("ascii", "replace").decode()[:900],
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur: {str(e)}")


@app.post("/thicken")
async def thicken(
    file: UploadFile = File(...),
    nom: str = Form("page"),
    trait_min: float = Form(0.40),
    seuil: int = Form(200),
    hauteur_in: float = Form(11.0),
    sortie_x: int = Form(2),
):
    """OBSIDIAN A4 : epaississement selectif d'une page raster (JPEG/PNG).
    Renvoie un PNG + mesures QA dans les en-tetes X-QA-*."""
    logger.info(f"🖊️ Thicken: {nom}")
    data = await file.read()
    g0 = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    del data
    if g0 is None:
        raise HTTPException(400, "Image illisible (fichier vide, corrompu ou pas une image)")
    if sortie_x not in (1, 2):
        raise HTTPException(400, "sortie_x doit valoir 1 ou 2")
    try:
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(
            None, lambda: epaissir(g0, trait_min, seuil, hauteur_in, 2, sortie_x))
        # mesures toujours a la taille d'origine : comparables a la cible validee
        # (moins de memoire, et memes chiffres que sortie_x=1)
        img_mes = img if sortie_x == 1 else cv2.resize(
            img, (g0.shape[1], g0.shape[0]), interpolation=cv2.INTER_AREA)
        m = await loop.run_in_executor(None, lambda: mesurer(img_mes, hauteur_in))
        del img_mes

        dpi = round(img.shape[0] / hauteur_in)
        buf = BytesIO()
        Image.fromarray(img).save(buf, format="PNG", optimize=True, dpi=(dpi, dpi))
        filename = f"{nom}_A4.png"
        logger.info(f"✅ Thicken terminé: {filename} {img.shape[1]}x{img.shape[0]} {m}")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Image-Width": str(img.shape[1]),
                "X-Image-Height": str(img.shape[0]),
                "X-Image-DPI": str(dpi),
                "X-QA-Zones": str(m["zones"]),
                "X-QA-Micro-Zones": str(m["micro_zones"]),
                "X-QA-Encre-Pct": str(m["encre_pct"]),
                "X-QA-Trait-Moyen-mm": str(m["trait_moyen_mm"]),
                "X-QA-Trait-Fin-Pct": str(m["trait_fin_pct"]),
            },
        )
    except Exception as e:
        logger.error(f"❌ Erreur thicken: {e}")
        raise HTTPException(500, str(e))


@app.get("/formats")
async def get_formats():
    """Retourne la liste des formats KDP supportés avec leurs dimensions"""
    return {
        "formats": {
            name: {
                "width_inches": width / 72,
                "height_inches": height / 72,
                "width_points": width,
                "height_points": height,
                "usage": {
                    "5x8": "Romans compacts",
                    "6x9": "Standard universel",
                    "8x10": "Livres illustrés",
                    "8.5x11": "Livres de coloriage"
                }.get(name, "")
            }
            for name, (width, height) in KDP_FORMATS.items()
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
