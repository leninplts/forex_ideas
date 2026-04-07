FROM python:3.13-slim

WORKDIR /app

# Dependencias de sistema necesarias para compilar paquetes Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para aprovechar cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# mt5linux + rpyc compatible (separado para evitar cache de Docker)
RUN pip install --no-cache-dir mt5linux>=1.0.0 rpyc==5.2.3 \
    && python -c "from mt5linux import MetaTrader5; print('mt5linux OK')"

# Copiar el resto del proyecto
COPY . .

# Crear usuario no-root y directorios necesarios
RUN useradd -m botuser \
    && mkdir -p data_cache forex_bot/models/saved forex_bot/logs forex_bot/analysis \
    && chown -R botuser:botuser /app

# Crear entrypoint que arregla permisos de bind mounts y ejecuta como botuser
RUN printf '#!/bin/sh\nchown -R botuser:botuser /app/data_cache /app/forex_bot/models/saved /app/forex_bot/logs 2>/dev/null\nexec gosu botuser "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Instalar gosu para drop de privilegios
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "forex_bot/main.py"]
