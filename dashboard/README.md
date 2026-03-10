# Dashboard – Predicción de Prematurez

Aplicación desarrollada en Python Dash para visualización de indicadores clínicos del proyecto Fundación Canguro.

## Ejecutar localmente

Instalar dependencias:

pip install -r requirements.txt

Ejecutar aplicación:

python app.py

Abrir en navegador:
http://127.0.0.1:8050/

## Despliegue en la nube (Render)

Este dashboard está listo para desplegarse como servicio web (WSGI) usando `gunicorn`.

1. Subir el repo a GitHub.
2. En Render: **New +** → **Web Service** → conectar el repositorio.
3. Configurar:
   - **Root Directory**: `dashboard`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:server`
4. Render define automáticamente `PORT`. El dashboard lo toma desde esa variable.