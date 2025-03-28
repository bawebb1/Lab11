import streamlit as st
# Streamlit App
def load_preprocessor():
    with open("preprocessor.pkl", "rb") as f:
        return pickle.load(f)

def predict_load(span_ft, deck_width_ft, age_years, num_lanes, material, condition_rating):
    preprocessor = load_preprocessor()
    model = load_model("tf_bridge_model.h5")
    
    input_data = pd.DataFrame([[span_ft, deck_width_ft, age_years, num_lanes, material, condition_rating]],
                              columns=["Span_ft", "Deck_Width_ft", "Age_Years", "Num_Lanes", "Material", "Condition_Rating"])
    input_transformed = preprocessor.transform(input_data)
    prediction = model.predict(input_transformed)
    return prediction[0][0]

st.title("Bridge Load Capacity Predictor")

span_ft = st.number_input("Span (ft)", min_value=0, value=250)
deck_width_ft = st.number_input("Deck Width (ft)", min_value=0, value=40)
age_years = st.number_input("Age (Years)", min_value=0, value=20)
num_lanes = st.number_input("Number of Lanes", min_value=1, value=2)
material = st.selectbox("Material", ["Steel", "Concrete", "Composite"])
condition_rating = st.slider("Condition Rating", 1, 5, 4)

if st.button("Predict Load Capacity"):
    prediction = predict_load(span_ft, deck_width_ft, age_years, num_lanes, material, condition_rating)
    st.write(f"### Predicted Max Load Capacity: {prediction:.2f} tons")
