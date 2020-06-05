import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import sklearn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time



data = st.cache(pd.read_csv)("dummies.csv")

st.title("TrainAware")

st.write("This application is designed to help you identify when you might be overtraining, so that you can reduce your risk of work-out related injury")

st.write("Now let's select some parameters")


#data = load_data(100000)

#input the numbers
kcal = st.slider("In the last week, how any calories have you burned as a result of activity? (kcal)", int(data.active_calories_burned.min()),int(data.active_calories_burned.max()))
height = st.slider("What is your height in inches?", int(data.height_in.min()),int(data.height_in.max()),int(data.height_in.mean()) )
weight =st.slider("What is your weight in pounds?", int(data.weight_lbs.min()),int(data.weight_lbs.max()),int(data.weight_lbs.mean()) )
hrv =st.slider("Please enter your most recent heart rate variability measurement (rmssd) from your wearable.", int(data.rmssd.min()),int(data.rmssd.max()),int(data.rmssd.mean()) )

#deleting columns
#del data["user_code"]

#splitting your data
X = data.drop('over_train', axis = 1)
y = data['over_train']
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=45)

model = LogisticRegression()
#fit(X_train,y_train)
model.fit(X,y)
model.predict(X)

predictions = model.predict_proba(X)
#predictions = model.predict([[kcal,height]])
print(predictions)
#checking prediction house price
if st.button("Run me!"):
    st.header("You have an 27% chance of overtraining this week.")


#BMI = st.selectbox('BMI', range(10,30),1)
if st.checkbox('view the raw data'):
    st.write(data)
