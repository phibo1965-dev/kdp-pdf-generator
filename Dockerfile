# Utiliser Python 3.11 slim comme image de base
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app.py .

# Exposer le port (Render.com définira automatiquement PORT)
EXPOSE 8000

# Variable d'environnement par défaut
ENV PORT=8000

# Commande de démarrage
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
