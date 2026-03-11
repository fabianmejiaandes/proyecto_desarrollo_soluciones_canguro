# Manual de Usuario – Dashboard de Predicción de Prematurez

## 1. Introducción

El presente manual describe el uso del dashboard desarrollado para el proyecto **Predicción de nacimiento prematuro en infantes con control prenatal**.

La solución fue diseñada como un sistema de apoyo a la toma de decisiones clínicas, orientado a facilitar la consulta de información relevante sobre los pacientes y la visualización de resultados asociados al riesgo de nacimiento prematuro.

El sistema integra información proveniente de registros clínicos anonimizados administrados por la **Fundación Canguro**, junto con un modelo de aprendizaje automático empaquetado y desplegado mediante una **API**. Esta arquitectura permite que el procesamiento de la información y la inferencia del modelo se realicen de forma centralizada, mientras el usuario interactúa únicamente con la interfaz del dashboard.

La solución se encuentra desplegada sobre infraestructura en la nube, utilizando servicios de **AWS**, con el propósito de facilitar el acceso, la escalabilidad y la integración entre la interfaz y el componente predictivo.

---

## 2. Acceso al sistema

El usuario accede al sistema a través de la interfaz del dashboard desplegada para el proyecto.

Desde esta interfaz es posible:

- Consultar la vista poblacional de los datos
- Navegar a la vista por paciente
- Visualizar indicadores clínicos relevantes
- Consultar resultados generados por el sistema predictivo

El usuario no requiere ejecutar scripts manualmente ni interactuar directamente con el modelo, ya que la comunicación con este se realiza a través de la API desplegada.

---

## 3. Funcionamiento general de la solución

La arquitectura del sistema está compuesta por tres elementos principales:

- **Dashboard**: interfaz visual con la que interactúa el usuario
- **API**: servicio intermedio que recibe las solicitudes del dashboard
- **Modelo empaquetado**: componente que procesa las variables de entrada y genera la inferencia

El flujo general es el siguiente:

1. El usuario accede al dashboard
2. El dashboard consulta la información disponible
3. Cuando se requiere una predicción, el dashboard envía las variables a la API
4. La API procesa la solicitud y ejecuta el modelo empaquetado
5. El resultado de la inferencia retorna al dashboard
6. El usuario visualiza la probabilidad o clasificación generada

Este enfoque permite separar la lógica visual de la lógica de predicción, mejorando la mantenibilidad y escalabilidad del sistema.

---

## 4. Estructura del dashboard

El sistema cuenta con dos vistas principales diseñadas para facilitar el análisis clínico y el monitoreo de la población.

### 4.1 Vista Poblacional

La vista poblacional permite analizar la información de todos los pacientes de forma agregada.

Entre las funcionalidades disponibles se encuentran:

- Visualización de indicadores clave (KPIs)
- Tabla resumen de indicadores clínicos
- Gráficos descriptivos de la población
- Distribución por edad gestacional
- Distribución por sexo
- Distribución por edad materna
- Análisis de factores prenatales y variables de seguimiento

Esta vista permite identificar patrones generales dentro de la población estudiada y apoyar procesos de monitoreo y análisis clínico.

---

### 4.2 Vista por Paciente

La vista por paciente permite explorar la información individual de cada caso clínico.

Entre sus funcionalidades se encuentran:

- Selección de paciente
- Consulta del perfil clínico
- Visualización de factores de riesgo
- Revisión de variables de seguimiento
- Presentación de resultados e inferencias asociadas al caso

Esta vista fue diseñada para concentrar en una sola pantalla la información más relevante del paciente, facilitando la lectura clínica y el apoyo a la decisión.

---

## 5. Uso del sistema

Para utilizar la herramienta, el usuario debe:

1. Acceder al dashboard desplegado
2. Seleccionar la vista deseada
3. Consultar la información poblacional o individual
4. En los casos donde aplique, revisar el resultado generado por el sistema predictivo

El proceso de inferencia no requiere interacción técnica por parte del usuario, ya que este se ejecuta automáticamente a través de la API integrada con el modelo.

---

## 6. Consideraciones sobre los datos

Los datos utilizados por el sistema corresponden a registros clínicos reales anonimizados proporcionados por la Fundación Canguro.

Por razones de confidencialidad:

- El acceso a los datos está restringido
- La información se utiliza únicamente con fines académicos y de investigación
- La visualización y gestión de la información se encuentra controlada dentro del entorno del proyecto

---

## 7. Alcance actual

En esta versión del sistema, el dashboard permite validar la arquitectura funcional de la solución, la navegación entre vistas y la integración con el componente predictivo.

La solución sienta las bases para versiones posteriores con mayores niveles de automatización, integración de datos y despliegue productivo.

---

## 8. Desarrollo futuro

En versiones posteriores del sistema se espera:

- Consolidar el despliegue completo de la solución en la nube
- Integrar nuevas fuentes de datos
- Mejorar la trazabilidad de predicciones
- Implementar mecanismos de autenticación y control de acceso
- Fortalecer la integración entre dashboard, API y modelo