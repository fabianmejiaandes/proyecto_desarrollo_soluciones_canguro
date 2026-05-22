# Identificación de factores de riesgo asociados a la malnutrición en bebés-canguro mediante modelos de aprendizaje automático, evaluados a 12 meses de edad corregida

**Programa Madre Canguro Integral — Fundación Canguro**
Universidad de los Andes, Bogotá, Colombia

---

## Integrantes

| Nombre | Correo |
|--------|--------|
| Jhoner Stiven Calderon Rojas | js.calderonr1@uniandes.edu.co |
| Juan Andrés Guzmán Pineda | ja.guzmanp@uniandes.edu.co |
| Hernando José Jiménez Díaz | h.jimenezd@uniandes.edu.co |
| Leison Fabián Mejia García | lf.mejiag1@uniandes.edu.co |

---

## Descripción del proyecto

La malnutrición en neonatos prematuros o con bajo peso al nacer constituye un reto clínico de considerable envergadura, particularmente durante el primer año de vida, periodo en el cual el crecimiento se modula por una interacción de factores prenatales, neonatales, intrahospitalarios y ambulatorios.

Este proyecto desarrolla un **sistema predictivo para la detección temprana del riesgo nutricional a los 12 meses de edad corregida**, focalizado en la cohorte de pacientes del Programa Madre Canguro Integral de la Fundación Canguro. Se empleó una base de datos histórica compuesta por aproximadamente 70 000 registros clínicos.

Se establecieron cuatro desenlaces basados en criterios antropométricos de la OMS:

| Desenlace | Descripción |
|-----------|-------------|
| Stunting | Baja talla para la edad |
| Bajo peso | Bajo peso para la edad |
| Wasting | Desnutrición aguda |
| Mixta | Combinación de los anteriores |

La metodología se fundamenta en una **arquitectura de cascada temporal** que integra 28 modelos LightGBM estructurados en 7 fases clínicas acumulativas, desde la información prenatal y del parto hasta el control ambulatorio a los 9 meses de edad corregida.

### Resultados principales (AUC — fase 9 meses)

| Desenlace | AUC |
|-----------|-----|
| Bajo peso | 0.964 |
| Condición mixta | 0.965 |
| Baja talla (Stunting) | 0.929 |
| Desnutrición aguda (Wasting) | 0.926 |

---

## Estructura del repositorio

```
proyecto_desarrollo_soluciones_canguro/
├── Datos/              # Muestra y estructura de los datos utilizados
├── Modelos/            # Modelos LightGBM entrenados (.lgb), features JSON y metadata
├── notebooks/          # Cuadernos de análisis, entrenamiento y evaluación
├── dashboard/          # Aplicación web interactiva (Dash)
│   ├── app.py
│   ├── Dashboard_demo.xlsx
│   └── assets/
├── documentacion/      # Propuesta del proyecto y manuales
└── Articulo/           # Publicación del estudio
```

---

## Dependencias

Python **3.10 o superior**.

Instalar todas las dependencias con:

```bash
pip install -r requirements.txt
```

Librerías principales:

| Librería | Uso |
|----------|-----|
| `lightgbm` | Carga y predicción de modelos |
| `pandas` | Manipulación de datos |
| `numpy` | Operaciones numéricas |
| `plotly` | Visualizaciones interactivas |
| `dash` | Framework del dashboard web |
| `dash-bootstrap-components` | Componentes UI del dashboard |
| `openpyxl` | Lectura de archivos Excel |
| `pyarrow` | Lectura de archivos Parquet |

---

## Entorno de ejecución

- **Sistema operativo:** Windows, macOS o Linux
- **Python:** 3.10+
- **Memoria recomendada:** 4 GB RAM mínimo
- **Almacenamiento:** ~500 MB para modelos y datos

---

## Acceso al dashboard

El dashboard está desplegado en AWS y disponible en:

**[https://kmcfundacioncanguro.duckdns.org](https://kmcfundacioncanguro.duckdns.org)**

No se requiere instalación local para usar la aplicación.

---

## Pasos para ejecutar localmente (opcional)

### 1. Clonar el repositorio

```bash
git clone https://github.com/fabianmejiaandes/proyecto_desarrollo_soluciones_canguro.git
cd proyecto_desarrollo_soluciones_canguro
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejemplo de uso

1. Ingresar al dashboard en [http://ec2-52-73-108-72.compute-1.amazonaws.com/](http://ec2-52-73-108-72.compute-1.amazonaws.com/)
2. En la interfaz, seleccionar un **ID de paciente** del panel izquierdo.
3. El sistema muestra:
   - Perfil clínico del paciente (peso, talla, perímetro cefálico por fase).
   - **Predicción de riesgo nutricional** para cada uno de los 4 desenlaces.
   - Comparación del paciente frente a la cohorte completa.
   - Trayectorias de puntajes Z a lo largo del tiempo.

---

## Modelos

Los 28 modelos entrenados se encuentran en la carpeta `Modelos/` con la nomenclatura:

```
modelo_<Desenlace>_<Fase>.lgb
```

**Fases disponibles:** F0 Prenatal/Parto → F1 Nacimiento → F2 Hospitalización → F3 40 semanas → F4 3 meses → F5 6 meses → F6 9 meses

Cada fase incorpora acumulativamente las variables de las fases anteriores, siguiendo la arquitectura de cascada temporal.

---

## Análisis de resultados e interpretación de métricas

Reporte interactivo con el análisis completo del proceso de modelado, interpretación de métricas y resultados:

**URL:** [https://dm6b7gcbsczrr.cloudfront.net/](https://dm6b7gcbsczrr.cloudfront.net/)

| Campo | Valor |
|-------|-------|
| Usuario | `canguro` |
| Contraseña | `%VSkH5m3mU7#Wec5pCJE` |

---

## Enlace al repositorio

[https://github.com/fabianmejiaandes/proyecto_desarrollo_soluciones_canguro](https://github.com/fabianmejiaandes/proyecto_desarrollo_soluciones_canguro)
