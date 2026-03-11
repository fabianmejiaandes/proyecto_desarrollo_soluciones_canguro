# Manual de Instalación – Proyecto Predicción de Prematurez

## 1. Introducción

Este documento describe el proceso de instalación y ejecución del sistema desarrollado para el proyecto **Predicción de nacimiento prematuro en infantes con control prenatal**.

El proyecto integra modelos de aprendizaje automático, una API de predicción y un dashboard interactivo desarrollado en Python.

---

## 2. Requisitos del sistema

Para ejecutar el proyecto se requiere:

- Python 3.10 o superior
- Git
- Acceso al repositorio de GitHub
- Conexión a internet

Opcional:

- Anaconda o Miniconda
- Visual Studio Code

---

## 3. Clonar el repositorio

Clonar el repositorio oficial del proyecto utilizando el siguiente comando:

git clone https://github.com/fabianmejiaandes/proyecto_desarrollo_soluciones_canguro.git

Luego ingresar al directorio del proyecto:

cd proyecto_desarrollo_soluciones_canguro

---

## 4. Crear un entorno virtual

Se recomienda crear un entorno virtual de Python para gestionar las dependencias del proyecto.

Crear entorno virtual:

python -m venv venv

Activar entorno virtual.

En Windows:

venv\Scripts\activate

En Linux o Mac:

source venv/bin/activate

---

## 5. Instalar dependencias

Instalar las librerías necesarias utilizando el archivo de requerimientos del dashboard.

pip install -r app/dashboard/requirements.txt

---

## 6. Ejecutar el Dashboard

Ubicarse en la carpeta del dashboard:

cd app/dashboard

Ejecutar la aplicación:

python dashboard.py

---

## 7. Acceder a la aplicación

Una vez iniciada la aplicación, abrir un navegador web y acceder a la siguiente dirección:

http://127.0.0.1:8050/

Esto abrirá el dashboard interactivo donde se podrán visualizar los indicadores clínicos y la información de los pacientes.

---

## 8. Fuente de datos

Actualmente el dashboard utiliza archivos Excel como fuente de datos.

Estos archivos contienen registros clínicos anonimizados utilizados únicamente para fines académicos dentro del proyecto.

En futuras versiones se planea integrar una base de datos y un sistema de ingesta automatizado.

---

## 9. Control de versiones

El proyecto utiliza las siguientes herramientas para el control de versiones y manejo de datos:

- Git para versionamiento del código.
- DVC para versionamiento de datos.
- AWS S3 para almacenamiento remoto de los datasets.

---

## 10. Soporte

Para soporte técnico o consultas sobre el sistema, contactar al equipo de desarrollo del proyecto.