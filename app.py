import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# CONFIGURACIÓN
# ==================================================

st.set_page_config(
    page_title="SimulIA - Novatech",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# CARGA DE DATOS
# ==================================================

@st.cache_data
def cargar_datos():
    return pd.read_excel("dataset_entrenamiento.xlsx")

@st.cache_resource
def cargar_modelos():
    clasificador = joblib.load("modelo_clasificador.pkl")
    regresor = joblib.load("modelo_regresor.pkl")
    return clasificador, regresor

df = cargar_datos()

clasificador, regresor = cargar_modelos()

# ==================================================
# TÍTULO
# ==================================================

st.title("📊 SimulIA - Novatech")
st.markdown("### Dashboard Inteligente de Simulación y Predicción")

# ==================================================
# VARIABLES DEL MODELO
# ==================================================

X = df.drop(
    columns=[
        "Tiempo de ciclo total",
        "Producción",
        "Cuello de botella"
    ]
)

y = df["Tiempo de ciclo total"]

predicciones = regresor.predict(X)

# ==================================================
# MÉTRICAS DEL MODELO
# ==================================================

mae = mean_absolute_error(y, predicciones)

rmse = mean_squared_error(
    y,
    predicciones
) ** 0.5

r2 = r2_score(
    y,
    predicciones
)

st.subheader("📈 Indicadores del Modelo")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("MAE", f"{mae:.2f}")

with c2:
    st.metric("RMSE", f"{rmse:.2f}")

with c3:
    st.metric("R²", f"{r2:.4f}")

# ==================================================
# VALIDACIÓN DEL MODELO
# ==================================================

st.subheader("📉 Verificación de Predicciones")

comparacion = pd.DataFrame({
    "Valor Real": y,
    "Valor Predicho": predicciones
})

comparacion["Error"] = (
    comparacion["Valor Real"]
    - comparacion["Valor Predicho"]
)

error_promedio = comparacion["Error"].abs().mean()

st.metric(
    "Error Promedio",
    f"{error_promedio:.2f} min"
)

fig_validacion = px.scatter(
    comparacion,
    x="Valor Real",
    y="Valor Predicho",
    title="Valores Reales vs Predichos",
    trendline="ols"
)

fig_validacion.add_shape(
    type="line",
    x0=comparacion["Valor Real"].min(),
    y0=comparacion["Valor Real"].min(),
    x1=comparacion["Valor Real"].max(),
    y1=comparacion["Valor Real"].max(),
    line=dict(color="red")
)

st.plotly_chart(
    fig_validacion,
    use_container_width=True
)

st.write("### Muestra de Validación")

st.dataframe(
    comparacion.head(20),
    use_container_width=True
)

st.write("### Distribución del Error")

fig_error = px.histogram(
    comparacion,
    x="Error",
    nbins=20,
    title="Errores de Predicción"
)

st.plotly_chart(
    fig_error,
    use_container_width=True
)

if r2 > 0.90:
    st.success(
        f"Modelo validado correctamente (R² = {r2:.4f})"
    )
else:
    st.warning(
        "El modelo requiere mejoras."
    )

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Parámetros del proceso")

llegada = st.sidebar.slider(
    "Llegada (min)",
    8.0,
    15.0,
    12.0
)

recepcion = st.sidebar.slider(
    "Recepción",
    2.0,
    4.0,
    2.5
)

inspeccion = st.sidebar.slider(
    "Inspección",
    2.0,
    5.0,
    3.5
)

ensamble = st.sidebar.slider(
    "Ensamble",
    12.0,
    25.0,
    18.0
)

prueba = st.sidebar.slider(
    "Prueba",
    8.0,
    18.0,
    12.0
)

calidad = st.sidebar.slider(
    "Calidad",
    2.0,
    6.0,
    4.0
)

empaque = st.sidebar.slider(
    "Empaque",
    2.0,
    5.0,
    3.0
)

almacen = st.sidebar.slider(
    "Almacén",
    1.0,
    3.0,
    1.5
)

operadores = st.sidebar.number_input(
    "Operadores ensamble",
    min_value=1,
    max_value=3,
    value=2
)

# ==================================================
# SIMULACIÓN
# ==================================================

if st.button("Ejecutar Simulación"):

    entrada = pd.DataFrame(
        [[
            llegada,
            recepcion,
            inspeccion,
            ensamble,
            prueba,
            calidad,
            empaque,
            almacen,
            operadores
        ]],
        columns=[
            "Llegada min",
            "Recepción",
            "Inspeción",
            "Ensamble",
            "Prueba",
            "Calidad",
            "Empaque",
            "Almacén",
            "Operadores ensamble"
        ]
    )

    tiempo_estimado = regresor.predict(
        entrada
    )[0]

    estaciones = {
        "Recepción": recepcion,
        "Inspección": inspeccion,
        "Ensamble": ensamble,
        "Prueba": prueba,
        "Calidad": calidad,
        "Empaque": empaque,
        "Almacén": almacen
    }

    cuello_estimado = max(
        estaciones,
        key=estaciones.get
    )

    tiempo_cuello = estaciones[cuello_estimado]

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "⏱ Tiempo de ciclo estimado",
            f"{tiempo_estimado:.2f} min"
        )

    with c2:
        st.metric(
            "🚨 Cuello de botella",
            f"{cuello_estimado} ({tiempo_cuello:.1f} min)"
        )

# ==================================================
# DATASET
# ==================================================

st.subheader("📋 Dataset de Entrenamiento")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# ==================================================
# PRODUCCIÓN
# ==================================================

st.subheader("🏭 Tiempo de Ciclo por Estación")

tiempos = {
    "Recepción": recepcion,
    "Inspección": inspeccion,
    "Ensamble": ensamble,
    "Prueba": prueba,
    "Calidad": calidad,
    "Empaque": empaque,
    "Almacén": almacen
}
tiempo_max = max(tiempos.values())

cuellos = [
    estacion
    for estacion, tiempo in tiempos.items()
    if tiempo >= tiempo_max * 0.95
]

df_tiempos = pd.DataFrame({
    "Estación": list(tiempos.keys()),
    "Tiempo": list(tiempos.values())
})

fig_estaciones = px.bar(
    df_tiempos,
    x="Estación",
    y="Tiempo",
    title="Tiempo Promedio por Estación"
)

st.plotly_chart(
    fig_estaciones,
    use_container_width=True
)

# ==================================================
# CUELLOS DE BOTELLA
# ==================================================

st.subheader("🚨 Cuello de Botella Detectado")

if "cuello_estimado" in locals():

    st.warning(
        f"La estación {cuello_estimado} presenta el mayor tiempo de ciclo "
        f"({tiempo_cuello:.1f} min) y constituye la restricción principal "
        f"del proceso en el escenario actual."
    )
# ==================================================
# CONCLUSIONES
# ==================================================

st.subheader("📌 Conclusiones del Modelo")

st.info(
    f"""
    • El modelo presenta un coeficiente R² de {r2:.4f}.

    • El error promedio es de {error_promedio:.2f} minutos.

    • El modelo puede utilizarse para evaluar escenarios
      futuros de la línea de producción Novatech.

    • Los cuellos de botella identificados permiten
      enfocar mejoras en el proceso.
    """
)

st.header("📌 Recomendaciones de Mejora")

if "cuello_estimado" in locals():

    if cuello_estimado == "Ensamble":

        st.warning("""
        La estación de Ensamble es el cuello de botella actual.

        Recomendaciones:
        - Incrementar operadores de ensamble.
        - Balancear la carga de trabajo.
        - Automatizar tareas repetitivas.
        - Reducir tiempos de preparación.
        """)

    elif cuello_estimado == "Prueba":

        st.warning("""
        La estación de Prueba es el cuello de botella actual.

        Recomendaciones:
        - Incrementar capacidad de prueba.
        - Automatizar pruebas repetitivas.
        - Reducir retrabajos.
        """)

    elif cuello_estimado == "Inspección":

        st.warning("""
        La estación de Inspección es el cuello de botella actual.

        Recomendaciones:
        - Optimizar inspecciones.
        - Aplicar muestreo estadístico.
        - Incorporar visión artificial.
        """)