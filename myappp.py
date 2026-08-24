import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.title('Iris Flower specied classifier')

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.5
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.5
)

if st.button('Predict'):
    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    species = iris.target_names[prediction[0]]

    st.success('Predicted specied' + species)

    st.write("Prediction probability")
    st.write("Setosa:", probability[0][0]*100,"%")
    st.write("Versicolor:", probability[0][1]*100, "%")
    st.write("Virginica:", probability[0][2]*100, "%")










