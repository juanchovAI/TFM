FROM python:3.11-slim

# Evita la creación de archivos pyc y fuerza salida sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Instala dependencias del sistema necesarias en caso de que alguna librería lo requiera
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copia e instala dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia el resto del código (incluye la carpeta modelos/ con los .pkl)
COPY . .

# Exponer puerto por convención
EXPOSE 8000

# Comando por defecto: Gunicorn con workers de Uvicorn.
# Usamos la forma shell para que Docker expanda la variable de entorno $PORT que Render provee.
# Ajusta main:app si tu app está en otro módulo.
CMD gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --workers 2
