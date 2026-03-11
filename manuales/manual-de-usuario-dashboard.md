# Manual de Usuario – Dashboard de Predicción de Prematurez

## 1. Introducción

El presente manual describe el uso del dashboard desarrollado para el proyecto **Predicción de nacimiento prematuro en infantes con control prenatal**.

La herramienta fue diseñada como un sistema de apoyo para especialistas médicos, permitiendo visualizar información clínica relevante y explorar patrones asociados al riesgo de nacimiento prematuro.

El sistema integra información proveniente de registros clínicos anonimizados administrados por la **Fundación Canguro**, los cuales contienen información del control prenatal, características maternas y seguimiento del recién nacido.

Actualmente el sistema se encuentra en fase de prototipo y utiliza archivos **Excel con datos reales anonimizados** para alimentar las visualizaciones del dashboard.

---

## 2. Acceso al Dashboard

El dashboard se ejecuta localmente utilizando Python Dash.

Una vez iniciado el sistema, el usuario podrá acceder desde un navegador web en la siguiente dirección:

http://127.0.0.1:8050/

---

## 3. Estructura del Dashboard

El sistema cuenta con dos vistas principales diseñadas para facilitar el análisis de la información clínica.

### 3.1 Vista Poblacional

La vista poblacional permite analizar la información de todos los pacientes de manera agregada.

Entre las funcionalidades disponibles se encuentran:

- Visualización de indicadores clave (KPIs)
- Distribución de variables demográficas
- Análisis de lactancia materna exclusiva
- Distribución de edad gestacional
- Tabla de indicadores clínicos
- Gráficos descriptivos de la población

Esta vista permite identificar patrones generales dentro de la población analizada y apoyar el monitoreo clínico.

---

### 3.2 Vista por Paciente

La vista por paciente permite explorar la información individual de cada caso clínico.

Las funcionalidades principales incluyen:

- Selección de paciente mediante un menú desplegable
- Visualización del perfil clínico
- Factores de riesgo registrados
- Información del nacimiento
- Evolución de indicadores de crecimiento
- Visualizaciones de seguimiento clínico

Esta sección está diseñada para apoyar el análisis clínico individual de cada paciente.

---

## 4. Uso del Dashboard

Para utilizar el sistema se deben seguir los siguientes pasos:

1. Ejecutar la aplicación del dashboard.
2. Abrir el navegador web.
3. Acceder a la dirección http://127.0.0.1:8050/.
4. Seleccionar la vista que se desea analizar.
5. En la vista por paciente seleccionar el identificador del paciente desde el menú desplegable.
6. Analizar los indicadores y visualizaciones presentadas.

---

## 5. Consideraciones sobre los datos

Los datos utilizados en el sistema provienen de registros clínicos anonimizados proporcionados por la Fundación Canguro.

Por razones de confidencialidad:

- Los datos no pueden ser compartidos fuera del equipo de desarrollo.
- El acceso al repositorio es restringido.
- Los datos se utilizan únicamente con fines académicos y de investigación.

---

## 6. Desarrollo futuro

En versiones posteriores del sistema se planea:

- Integrar el modelo de machine learning directamente en el dashboard.
- Conectar el sistema con una base de datos clínica.
- Desplegar la aplicación en infraestructura cloud.
- Implementar autenticación de usuarios.