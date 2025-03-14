# huella-carbono
Proyecto digital para medir el impacto de la huella de carbono en diferentes escenarios para la comercialización de combustibles

# Apuntes y módulos que deberían tener en cuenta

🔧 Módulos y Librerías Esenciales
1️⃣ Manejo de Datos

pandas → Para estructurar y analizar datos.
bash
Copiar
Editar
pip install pandas
numpy → Para cálculos numéricos eficientes.
bash
Copiar
Editar
pip install numpy
2️⃣ Cálculo de Huella de Carbono

pycarbon (si está disponible) → Para cálculos de emisiones.
co2eq → Para estimar emisiones de CO₂.
bash
Copiar
Editar
pip install co2eq
3️⃣ Modelado y Optimización

scikit-learn → Para análisis de datos y modelos de predicción.
bash
Copiar
Editar
pip install scikit-learn
pulp → Para optimización de costos vs impacto ambiental.
bash
Copiar
Editar
pip install pulp
4️⃣ Visualización de Datos

matplotlib y seaborn → Para gráficos comparativos.
bash
Copiar
Editar
pip install matplotlib seaborn
plotly → Para dashboards interactivos.
bash
Copiar
Editar
pip install plotly
5️⃣ Interfaz de Usuario

streamlit → Para crear una aplicación web sin necesidad de desarrollo complejo.
bash
Copiar
Editar
pip install streamlit
dash → Para una interfaz más avanzada basada en Flask.
bash
Copiar
Editar
pip install dash
🚀 Flujo de Desarrollo
Recolectar datos de combustibles (costos, emisiones, regulaciones).
Crear un modelo de decisión que relacione impacto ambiental con costos.
Construir la APP con streamlit o dash.
Visualizar los resultados con gráficos.

1️⃣ Introducción a Python para la Industria Energética
Breve repaso sobre Python y sus aplicaciones en análisis de datos
Librerías clave:
pandas (manejo de datos)
numpy (cálculo numérico)
matplotlib y seaborn (visualización)
requests y BeautifulSoup (web scraping para regulaciones)
scikit-learn (modelos predictivos)
2️⃣ Regulaciones y Marco Legal (Web Scraping y API’s)
Obtener datos de normativas ambientales
Uso de requests para consumir datos de fuentes como la Agencia Internacional de Energía (IEA) o ministerios locales.
Uso de BeautifulSoup para extraer información de sitios web gubernamentales.
Ejemplo práctico: Descargar un documento de regulaciones sobre emisiones y extraer las secciones clave en Python.
📌 Recurso:

API de la European Environment Agency (EEA) → https://www.eea.europa.eu/data-and-maps
3️⃣ Medición de Huella de Carbono con Python
Fuentes de datos: bases de datos de emisiones de CO₂ (Ej: Our World in Data, EIA, World Bank).
Conversión de unidades de carbono: Uso de pandas para manipular datos de consumo energético y calcular emisiones.
Cálculo de huella de carbono (CO₂ por barril de petróleo vendido).
Ejemplo práctico: Calcular la huella de carbono de una empresa según su producción.
📌 Recurso:

Base de datos de emisiones de carbono: https://ourworldindata.org/co2-emissions
4️⃣ Modelos Predictivos para la Toma de Decisiones
Introducción a Machine Learning para análisis predictivo.
Uso de scikit-learn para modelar tendencias de emisiones con datos históricos.
Ejemplo práctico: Predicción de huella de carbono en los próximos 5 años según consumo energético y regulaciones.
📌 Recurso:

Documentación oficial de scikit-learn: https://scikit-learn.org/stable/
5️⃣ Visualización de Datos para Reportes
Uso de matplotlib y seaborn para crear gráficos de tendencias de emisiones y consumos.
Ejemplo práctico: Gráfica de reducción de CO₂ según diferentes políticas ambientales aplicadas.
📌 Recurso:

Curso gratuito de visualización de datos en Python: https://datavizcatalogue.com/
📂 Recursos Extra
Curso de Python para Data Science: https://www.datacamp.com/courses/
Dataset de emisiones por país y sector: https://datahub.io/core/co2-fossil-global
Paper sobre predicciones en energía y huella de carbono: https://arxiv.org/

Aquí tienes un código de ejemplo en Python que descarga datos de emisiones de carbono, realiza un análisis de tendencias y genera una predicción para los próximos años usando pandas, matplotlib y scikit-learn.

📌 Funcionalidades del código:
✅ Descarga datos de emisiones de CO₂ de un dataset global.
✅ Realiza un análisis visual de las tendencias.
✅ Aplica un modelo de regresión lineal para predecir emisiones futuras.

python
Copiar
Editar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1️⃣ Cargar datos de emisiones de CO₂ (Fuente: Our World in Data)
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
df = pd.read_csv(url)

# 2️⃣ Filtrar datos de un país específico (Ej: Colombia) y las columnas clave
pais = "Colombia"
df_colombia = df[df["country"] == pais][["year", "co2"]].dropna()

# 3️⃣ Visualizar datos históricos
plt.figure(figsize=(10,5))
plt.plot(df_colombia["year"], df_colombia["co2"], marker='o', linestyle='-', color='b', label="CO₂ real")
plt.xlabel("Año")
plt.ylabel("Emisiones CO₂ (millones de toneladas)")
plt.title(f"Emisiones de CO₂ en {pais}")
plt.legend()
plt.show()

# 4️⃣ Preparar datos para modelo predictivo
X = df_colombia["year"].values.reshape(-1, 1)
y = df_colombia["co2"].values.reshape(-1, 1)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Aplicar regresión lineal para predecir emisiones futuras
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones para años futuros
años_futuros = np.array(range(2024, 2035)).reshape(-1, 1)
predicciones = modelo.predict(años_futuros)

# 6️⃣ Visualizar predicción
plt.figure(figsize=(10,5))
plt.scatter(X, y, color='blue', label="Datos reales")
plt.plot(años_futuros, predicciones, color='red', linestyle="dashed", label="Predicción 2024-2035")
plt.xlabel("Año")
plt.ylabel("Emisiones CO₂ (millones de toneladas)")
plt.title(f"Predicción de emisiones de CO₂ en {pais}")
plt.legend()
plt.show()
🔍 Explicación rápida del código
1️⃣ Carga datos de emisiones de CO₂ desde un archivo CSV en línea.
2️⃣ Filtra los datos para Colombia (puedes cambiar el país).
3️⃣ Grafica la evolución de emisiones.
4️⃣ Prepara los datos y aplica regresión lineal con scikit-learn.
5️⃣ Predice emisiones futuras hasta 2035 y visualiza la tendencia.

📊 Posibles mejoras
✅ Integrar datos de consumo energético.
✅ Aplicar modelos más avanzados de Machine Learning.
✅ Obtener regulaciones ambientales y compararlas con emisiones.

Aquí tienes un paso a paso para que cada estudiante trabaje en ramas individuales dentro de un mismo repositorio:

1. Crear el repositorio y dar acceso a los estudiantes
Si aún no tienes un repositorio:

Ve a GitHub y crea un nuevo repositorio.
Comparte el enlace del repositorio con los estudiantes.
En Settings > Collaborators, agrégalos como colaboradores con permisos de escritura.
2. Clonar el repositorio (cada estudiante)
Cada estudiante debe clonar el repositorio en su computadora:

bash
Copiar código
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
3. Crear una rama individual
Cada estudiante debe crear su propia rama con su nombre:

bash
Copiar código
git checkout -b nombre-del-estudiante
Ejemplo:

bash
Copiar código
git checkout -b juan-perez
4. Hacer cambios y confirmar (commit)
Cada estudiante puede editar archivos y luego hacer commit:

bash
Copiar código
git add .
git commit -m "Agregando mis cambios"
5. Subir la rama al repositorio remoto
Cada estudiante debe subir su rama al repositorio en GitHub:

bash
Copiar código
git push origin nombre-del-estudiante
6. Crear un Pull Request (PR) en GitHub
Ir al repositorio en GitHub.
Hacer clic en Pull Requests > New Pull Request.
Seleccionar su rama y compararla con main.
Agregar un título y una descripción, luego hacer clic en Create Pull Request.
7. Revisar y fusionar los cambios
Tú (o un estudiante asignado) revisa los Pull Requests.
Si todo está bien, se aprueba y se fusiona en main.
Luego, cada estudiante debe actualizar su código local:
bash
Copiar código
git checkout main
git pull origin main
git branch -d nombre-del-estudiante  # Borra la rama local después de fusionar
Consejos Adicionales
✅ Recomienda a los estudiantes hacer pull antes de trabajar:

bash
Copiar código
git pull origin main
✅ Para evitar conflictos, asigna tareas claras a cada estudiante.
✅ Usa git status y git log para verificar cambios antes de hacer commit.

Este flujo asegura que cada estudiante trabaja de forma independiente y permite revisar los cambios antes de fusionarlos. ¿Necesitas que ajuste algo para tu clase? 😊

1. Configurar la terminal predeterminada en VS Code
Abre Visual Studio Code.
Presiona Ctrl + Shift + P para abrir la paleta de comandos.
Escribe "Configurar perfil de terminal predeterminado" y selecciónalo.
Elige WSL en la lista de opciones.
Esto hará que cada vez que abras la terminal en VS Code (Ctrl + Ñ), se abra WSL en lugar del símbolo de sistema de Windows.
