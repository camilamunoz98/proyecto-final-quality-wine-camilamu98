
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title = 'Predictor de Calidad de Vino',
    page_icon = '🍷',
    layout = 'centered'
)

@st.cache_resource

def cargar_modelo():
    modelo = joblib.load('mejor_modelo_random_forest.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    return modelo, columnas
modelo, columnas_modelo = cargar_modelo()

st.title('Predictor de Calidad de Vino')
st.markdown(
    ' Ingresar las propiedades fisicoquímicas del vino y el modelo ' \
    'predicirá su categoría de calidad: **Bajo**, **Medio** o **Alto**.'
)

st.divider()

st.subheader ('Tipo de vino')
tipo_vino = st.selectbox('Tipo de vino', ['Red', 'White'])

st.subheader('Propiedades fisicoquímicas')

col1, col2 = st.columns(2)

with col1:

    fixed_acidity = st.slider ('Acidez fija (g/L)', 3.8, 15.9, 7.2, 0.1)
    volatile_acidity = st.slider("Acidez volátil (g/L)", 0.08, 1.58, 0.34, 0.01)
    citric_acid = st.slider("Ácido cítrico (g/L)", 0.0, 1.66, 0.32, 0.01)
    residual_sugar = st.slider("Azúcar residual (g/L)", 0.6, 65.8, 5.4, 0.1)
    chlorides = st.slider("Cloruros (g/L)", 0.009, 0.611, 0.056, 0.001)
    free_sulfur_dioxide = st.slider("Dióxido de azufre libre (mg/L)", 1, 289, 30)
 
with col2:
    total_sulfur_dioxide = st.slider("Dióxido de azufre total (mg/L)", 6, 440, 115)
    density = st.slider("Densidad (g/cm³)", 0.987, 1.039, 0.995, 0.0001, format="%.4f")
    pH = st.slider("pH", 2.72, 4.01, 3.22, 0.01)
    sulphates = st.slider("Sulfatos (g/L)", 0.22, 2.0, 0.53, 0.01)
    alcohol = st.slider("Alcohol (% vol.)", 8.0, 14.9, 10.5, 0.1)

st.divider()

if st.button ("🔍 Predecir calidad", type="primary", use_container_width=True) :
  datos = {
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": np.log1p(citric_acid),
    "residual sugar": np.log1p(residual_sugar),      # transformada
    "chlorides": np.log1p(chlorides),                 # transformada
    "free sulfur dioxide": np.log1p(free_sulfur_dioxide),   # transformada
    "total sulfur dioxide": np.log1p(total_sulfur_dioxide), # transformada
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
    'wine_type': 0 if tipo_vino == 'Red' else 1,
  }
  #Convierto el DataFrame con el orden exacto de las columnas que el modelo espera
  wine_data = pd.DataFrame([datos])
  wine_data = wine_data[columnas_modelo]

  #Predicción
  prediccion = modelo.predict (wine_data)[0]
  probabilidades = modelo.predict_proba(wine_data)[0]

  #Mostrar los resultados 
  colores = {"bajo": "🔴", "medio": "🟡", "alto": "🟢"}
  st.subheader(f"Resultado: {colores.get(prediccion, '')} Calidad **{prediccion.upper()}**")

  prob_wine_data = pd.DataFrame({
    'Categoría': modelo.classes_,
    'Probabilidad': probabilidades
    }).sort_values('Probabilidad', ascending = False)

  st.bar_chart(prob_wine_data.set_index('Categoría'))

st.divider()

st.caption(
  'Modelo: Random Forest Optimizado | Dataset: Wine Quality (UCI) | '
  'Proyecto de Machine Learning'
)