"""
Service FastAPI de génération de PDFs pour livres de coloriage KDP
Formats supportés: 5x8, 6x9, 8x10, 8.5x11 inches
Version OPTIMISÉE avec téléchargements parallèles + Conversion SVG→PNG
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
import httpx
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
import cairosvg
import numpy as np
import cv2
from thicken import epaissir, mesurer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KDP PDF Generator",
    description="Service de génération de PDFs pour livres de coloriage Amazon KDP - Version Optimisée + SVG→PNG",
    version="3.7.0"
)

# Formats KDP en inches → points (1 inch = 72 points) pour PDF
KDP_FORMATS = {
    "5x8": (5 * inch, 8 * inch),
    "6x9": (6 * inch, 9 * inch),
    "8x10": (8 * inch, 10 * inch),
    "8.5x11": (8.5 * inch, 11 * inch)
}

# Formats KDP en inches pour conversion SVG→PNG
KDP_FORMATS_INCHES = {
    "A4": (8.27, 11.69),
    "a4": (8.27, 11.69),
    "8.5x11": (8.5, 11.0),
    "8.5X11": (8.5, 11.0),
    "LETTER": (8.5, 11.0),
    "6x9": (6.0, 9.0),
    "5x8": (5.0, 8.0),
    "5.5x8.5": (5.5, 8.5),
    "7x10": (7.0, 10.0),
    "8x10": (8.0, 10.0),
    "8.5x8.5": (8.5, 8.5),
}

# Configuration pour téléchargements parallèles
MAX_CONCURRENT_DOWNLOADS = 10  # Nombre de téléchargements simultanés


# =============================================================================
# MODÈLES DE REQUÊTE
# =============================================================================

class PDFRequest(BaseModel):
    """Modèle de requête pour la génération de PDF"""
    urls: List[HttpUrl]
    book_id: str
    format: str = "8.5x11"
    title: str = ""
    author: str = "Xavier Legrand"
    include_title_page: bool = False
    include_ownership_page: bool = False
    include_copyright_page: bool = False
    include_notice_page: bool = False


class SVGToPNGRequest(BaseModel):
    """Modèle de requête pour la conversion SVG→PNG"""
    book_id: str
    image_number: str
    format: str
    svg_url: str
    dpi: Optional[int] = 300
    margins_mm: Optional[float] = 0


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_format_dimensions(format_str: str, dpi: int = 300):
    """Calcule les dimensions en pixels pour un format KDP donné"""
    format_clean = format_str.strip().upper().replace(",", ".").replace(" ", "")
    
    for key, (w, h) in KDP_FORMATS_INCHES.items():
        if key.upper() == format_clean:
            return int(w * dpi), int(h * dpi)
    
    # Parsing manuel
    if "X" in format_clean:
        try:
            parts = format_clean.split("X")
            return int(float(parts[0]) * dpi), int(float(parts[1]) * dpi)
        except:
            pass
    
    # Défaut A4
    return int(8.27 * dpi), int(11.69 * dpi)


async def download_image_async(url: str, index: int, timeout: int = 30) -> tuple[int, bytes]:
    """
    Télécharge une image de manière asynchrone
    
    Args:
        url: URL de l'image
        index: Index de l'image dans la liste
        timeout: Timeout en secondes
        
    Returns:
        Tuple (index, contenu de l'image en bytes)
    """
    try:
        logger.info(f"[{index}] Téléchargement de l'image: {url}")
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"[{index}] Type de contenu inhabituel: {content_type}")
            
            logger.info(f"[{index}] ✓ Image téléchargée: {len(response.content)} bytes")
            return (index, response.content)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"[{index}] Erreur HTTP {e.response.status_code} pour {url}")
        raise HTTPException(
            status_code=400,
            detail=f"Image {index}: Impossible de télécharger (HTTP {e.response.status_code}): {url}"
        )
    except httpx.TimeoutException:
        logger.error(f"[{index}] Timeout lors du téléchargement de {url}")
        raise HTTPException(
            status_code=408,
            detail=f"Image {index}: Timeout lors du téléchargement: {url}"
        )
    except Exception as e:
        logger.error(f"[{index}] Erreur lors du téléchargement de {url}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Image {index}: Erreur lors du téléchargement: {str(e)}"
        )


async def download_images_parallel(urls: List[str]) -> List[bytes]:
    """
    Télécharge plusieurs images en parallèle
    
    Args:
        urls: Liste des URLs d'images
        
    Returns:
        Liste des images en bytes (dans le bon ordre)
    """
    logger.info(f"Téléchargement parallèle de {len(urls)} images (max {MAX_CONCURRENT_DOWNLOADS} simultanés)")
    
    # Créer un semaphore pour limiter les téléchargements concurrents
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    
    async def download_with_semaphore(url: str, index: int):
        async with semaphore:
            return await download_image_async(url, index)
    
    # Lancer tous les téléchargements en parallèle
    tasks = [download_with_semaphore(str(url), i) for i, url in enumerate(urls)]
    results = await asyncio.gather(*tasks)
    
    # Trier les résultats par index pour garder l'ordre original
    results.sort(key=lambda x: x[0])
    
    # Extraire seulement les données (pas les index)
    images_data = [data for _, data in results]
    
    logger.info(f"✓ Tous les téléchargements terminés")
    return images_data


# =============================================================================
# FONCTIONS PAGES PDF
# =============================================================================

def add_title_page(c: canvas.Canvas, title: str, author: str, width: float, height: float):
    """Ajoute une page de titre"""
    c.setFont("Helvetica-Bold", 28)
    title_y = height * 0.55
    
    max_width = width * 0.8
    if c.stringWidth(title, "Helvetica-Bold", 28) > max_width:
        words = title.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if c.stringWidth(test_line, "Helvetica-Bold", 28) <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for i, line in enumerate(lines):
            c.drawCentredString(width / 2, title_y - (i * 35), line)
    else:
        c.drawCentredString(width / 2, title_y, title)
    
    c.setFont("Helvetica", 18)
    c.drawCentredString(width / 2, height * 0.40, f"Par {author}")
    
    c.setLineWidth(2)
    c.line(width * 0.2, height * 0.35, width * 0.8, height * 0.35)


def add_ownership_page(c: canvas.Canvas, width: float, height: float):
    """Ajoute une page "Ce livre appartient à" """
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height * 0.7, "Ce livre appartient à:")
    
    c.setLineWidth(1)
    line_y = height * 0.5
    line_width = width * 0.6
    line_x_start = (width - line_width) / 2
    
    for i in range(3):
        y = line_y - (i * 50)
        c.line(line_x_start, y, line_x_start + line_width, y)


def add_copyright_page(c: canvas.Canvas, author: str, width: float, height: float):
    """Ajoute une page de copyright"""
    import datetime
    current_year = datetime.datetime.now().year
    
    text_y = height * 0.5
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, text_y + 40, f"© {current_year} {author}")
    
    c.setFont("Helvetica", 10)
    c.drawCentredString(width / 2, text_y, "Tous droits réservés.")
    
    c.setFont("Helvetica", 9)
    text_lines = [
        "Aucune partie de ce livre ne peut être reproduite,",
        "distribuée ou transmise sous quelque forme que ce soit",
        "sans l'autorisation écrite préalable de l'auteur."
    ]
    
    for i, line in enumerate(text_lines):
        c.drawCentredString(width / 2, text_y - 40 - (i * 15), line)


def add_notice_page(c: canvas.Canvas, width: float, height: float):
    """Ajoute une page d'avis important"""
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height * 0.75, "Avis Important")
    
    c.setLineWidth(1)
    c.line(width * 0.2, height * 0.72, width * 0.8, height * 0.72)
    
    c.setFont("Helvetica", 11)
    text_lines = [
        "Ce livre de coloriage est conçu pour les adultes.",
        "",
        "Les images peuvent contenir des détails complexes",
        "qui ne conviennent pas aux jeunes enfants.",
        "",
        "Utilisez des crayons de couleur, des feutres",
        "ou des stylos adaptés au papier.",
        "",
        "Prenez votre temps et profitez du processus créatif !",
        "",
        "Bon coloriage !"
    ]
    
    text_y = height * 0.6
    for i, line in enumerate(text_lines):
        c.drawCentredString(width / 2, text_y - (i * 20), line)


def create_pdf(images_data: List[bytes], request: PDFRequest) -> bytes:
    """
    Crée un PDF à partir d'une liste d'images
    
    Args:
        images_data: Liste des images en bytes
        request: Requête contenant les paramètres du PDF
        
    Returns:
        Contenu du PDF en bytes
    """
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
        c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height))
        
        page_number = 1
        
        # Pages supplémentaires
        if request.include_title_page and request.title:
            logger.info("Ajout de la page de titre")
            add_title_page(c, request.title, request.author, page_width, page_height)
            c.showPage()
            page_number += 1
        
        if request.include_ownership_page:
            logger.info("Ajout de la page d'appartenance")
            add_ownership_page(c, page_width, page_height)
            c.showPage()
            page_number += 1
        
        if request.include_copyright_page:
            logger.info("Ajout de la page de copyright")
            add_copyright_page(c, request.author, page_width, page_height)
            c.showPage()
            page_number += 1
        
        if request.include_notice_page:
            logger.info("Ajout de la page d'avis important")
            add_notice_page(c, page_width, page_height)
            c.showPage()
            page_number += 1
        
        # Images de coloriage
        for idx, img_data in enumerate(images_data, 1):
            logger.info(f"Traitement de l'image {idx}/{len(images_data)}")
            
            img = Image.open(io.BytesIO(img_data))
            
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height
            page_aspect_ratio = page_width / page_height
            
            if aspect_ratio > page_aspect_ratio:
                draw_width = page_width
                draw_height = page_width / aspect_ratio
            else:
                draw_height = page_height
                draw_width = page_height * aspect_ratio
            
            x = (page_width - draw_width) / 2
            y = (page_height - draw_height) / 2
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            img_reader = ImageReader(img_buffer)
            
            c.drawImage(img_reader, x, y, width=draw_width, height=draw_height)
            
            c.setFont("Helvetica", 10)
            c.drawCentredString(page_width / 2, 0.5 * inch, str(page_number))
            
            if idx < len(images_data):
                c.showPage()
                page_number += 1
        
        c.save()
        
        pdf_bytes = pdf_buffer.getvalue()
        logger.info(f"PDF créé avec succès: {len(pdf_bytes)} bytes, {page_number} pages")
        
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la création du PDF: {str(e)}"
        )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Endpoint racine - Informations sur l'API"""
    return {
        "service": "KDP PDF Generator",
        "version": "3.7.0 (Optimisé + SVG→PNG)",
        "status": "operational",
        "features": ["Téléchargements parallèles", "Conversion SVG→PNG"],
        "formats_supported": list(KDP_FORMATS.keys()),
        "endpoints": {
            "health": "/health",
            "generate": "/generate-pdf (POST)",
            "svg_to_png": "/svg_to_png (POST)",
            "thicken": "/thicken (POST, multipart)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "KDP PDF Generator",
        "version": "3.7.0",
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
        logger.info(f"Pages supplémentaires: T={request.include_title_page}, "
                   f"O={request.include_ownership_page}, "
                   f"C={request.include_copyright_page}, "
                   f"N={request.include_notice_page}")
        logger.info(f"═══════════════════════════════════════════════")
        
        # Validations
        if not request.urls:
            raise HTTPException(status_code=400, detail="La liste d'URLs ne peut pas être vide")
        
        if len(request.urls) > 500:
            raise HTTPException(status_code=400, detail="Maximum 500 images par PDF")
        
        if request.format not in KDP_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté: {request.format}. "
                       f"Formats disponibles: {', '.join(KDP_FORMATS.keys())}"
            )
        
        # Télécharger toutes les images EN PARALLÈLE
        import time
        start_time = time.time()
        
        images_data = await download_images_parallel(request.urls)
        
        download_time = time.time() - start_time
        logger.info(f"⚡ Téléchargements terminés en {download_time:.2f}s "
                   f"({len(request.urls) / download_time:.1f} images/sec)")
        
        # Générer le PDF
        pdf_start = time.time()
        pdf_bytes = create_pdf(images_data, request)
        pdf_time = time.time() - pdf_start
        
        total_time = time.time() - start_time
        
        logger.info(f"═══════════════════════════════════════════════")
        logger.info(f"✓ PDF généré avec succès: {request.book_id}")
        logger.info(f"  Temps téléchargements: {download_time:.2f}s")
        logger.info(f"  Temps création PDF: {pdf_time:.2f}s")
        logger.info(f"  Temps TOTAL: {total_time:.2f}s")
        logger.info(f"  Taille PDF: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
        logger.info(f"═══════════════════════════════════════════════")
        
        return Response(
            content=pdf_bytes,
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{request.book_id}.pdf"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur: {str(e)}")


@app.post("/svg_to_png")
async def svg_to_png(request: SVGToPNGRequest):
    """Convertit un SVG en PNG 300 DPI au format KDP"""
    
    logger.info(f"🎨 SVG→PNG: {request.book_id}_{request.image_number}")
    
    try:
        # 1. Télécharger le SVG
        response = requests.get(request.svg_url, timeout=60, allow_redirects=True)
        if response.status_code != 200:
            raise HTTPException(400, f"Erreur téléchargement SVG: {response.status_code}")
        
        svg_content = response.content
        
        # 2. Calculer dimensions
        width_px, height_px = get_format_dimensions(request.format, request.dpi)
        
        # 3. Convertir SVG → PNG
        png_data = cairosvg.svg2png(
            bytestring=svg_content,
            output_width=width_px,
            output_height=height_px,
            dpi=request.dpi
        )
        
        # 4. Optimiser (RGBA → RGB avec fond blanc)
        img = Image.open(BytesIO(png_data))
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        output = BytesIO()
        img.save(output, format='PNG', optimize=True)
        final_png = output.getvalue()
        
        # 5. Retourner le PNG
        filename = f"{request.book_id}_{request.image_number}_KDP.png"
        
        logger.info(f"✅ SVG→PNG terminé: {filename} ({img.size[0]}x{img.size[1]} @ {request.dpi} DPI)")
        
        return Response(
            content=final_png,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Image-Width": str(img.size[0]),
                "X-Image-Height": str(img.size[1]),
                "X-Image-DPI": str(request.dpi)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur SVG→PNG: {e}")
        raise HTTPException(500, str(e))


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
    if g0 is None:
        raise HTTPException(400, f"Image illisible ({len(data)} octets recus)")
    if sortie_x not in (1, 2):
        raise HTTPException(400, "sortie_x doit valoir 1 ou 2")
    try:
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(
            None, lambda: epaissir(g0, trait_min, seuil, hauteur_in, 2, sortie_x))
        m = await loop.run_in_executor(None, lambda: mesurer(img, hauteur_in))

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
