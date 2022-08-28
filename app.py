import pandas as pd
import streamlit as st
import pickle
st.title("Iris Classification")
pipe=pickle.load(open("iris.pkl","rb"))
df=pd.read_csv("iris.csv")
sepal_len=sorted(df["SepalLengthCm"].unique())
sepal_wid=sorted(df["SepalWidthCm"].unique())
petal_len=sorted(df["PetalLengthCm"].unique())
petal_wid=sorted(df["PetalWidthCm"].unique())

col1,col2=st.columns(2)
with col1:
    sepal_length=st.selectbox("choose Sepal Length",sepal_len)
with col2:
    sepal_width = st.selectbox("choose Sepal width",sepal_wid)

col3,col4=st.columns(2)
with col3:
    petal_length=st.selectbox("Choose petal length",petal_len)
with col4:
    petal_width =st.selectbox("Choose petal width",petal_wid)
if st.button("predict Probabibilty"):
    prediction=pipe.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    st.title("Specie is "+prediction)
st.text("DESIGNED BY :SOLO_DRAGON AKA  KANDADI SAI TEJA")