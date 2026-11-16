from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Iris Species Predictor",page_icon="🌸")
st.title("🌸 Iris Species Predictor")
st.write("Interactive Decision Tree classification using four flower measurements (centimeters).")
@st.cache_resource
def load_model():
    return joblib.load(ROOT / "iris_model.joblib")
model=load_model()
with st.form("prediction"):
    sepal_length=st.slider("Sepal length (cm)",4.0,8.0,5.8,0.1)
    sepal_width=st.slider("Sepal width (cm)",2.0,4.5,3.0,0.1)
    petal_length=st.slider("Petal length (cm)",1.0,7.0,4.0,0.1)
    petal_width=st.slider("Petal width (cm)",0.1,2.6,1.2,0.1)
    submit=st.form_submit_button("Predict species")
if submit:
    row=pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],columns=["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"])
    pred=int(model.predict(row)[0]);names=["Setosa","Versicolor","Virginica"]
    st.success(f"Predicted species: {names[pred]}")
    probs=model.predict_proba(row)[0]
    st.bar_chart(pd.DataFrame({"Probability":probs},index=names))
st.caption("Educational demonstration only. Model is trained on the public Iris dataset.")
