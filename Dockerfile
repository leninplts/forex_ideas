FROM python:3.13-slim

WORKDIR /app

# Dependencias de sistema necesarias para compilar paquetes Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para aprovechar cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Crear directorios necesarios
RUN mkdir -p data_cache forex_bot/models/saved forex_bot/logs forex_bot/analysis

# No correr como root
RUN useradd -m botuser && chown -R botuser:botuser /app
USER botuser

# Entrypoint: el bot principal
CMD ["python", "forex_bot/main.py"]
