#-------------------------------
## STEP -> 1. Import Libraries
#-------------------------------

import pandas as pd 
import streamlit as st 
import joblib
import os


#-----------------------------
## STEP -> 2. Load the model 
#-----------------------------

# BASE DIR SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model aur Scaler files in 'model' subfolder
model_path = os.path.join(BASE_DIR, "Model", "employee_salary_model.joblib")
scaler_path = os.path.join(BASE_DIR, "Model", "scaler_employee.joblib")

# Check and Load
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    st.error("Model ya Scaler file nahi mili! Location verify karein.")
    st.stop()

#---------------------------------
## STEP -> 3.Title ans Subheader
#---------------------------------

st.title("💼 Employee Salary Predictor")
st.subheader("Predict the salary of the employee")
st.write("Enter the details of the employee given below and predict ")


#--------------------------
## STEP -> 4. User Input 
#--------------------------

# name of the employee 

name = st.text_input("Employee Name")
st.write(f"The name of the emplployee is {name}")


# columns(features)

# 1. age 
age = st.number_input("Enter the age ",
                      min_value = 18,
                      max_value = 90,
                      value = 55,
                      step = 1)

st.write(f"The age is {age}")


# 2. Workclass
workclass = st.selectbox("Enter the workclass",["Private","Self-emp-not-inc","Local-gov","?","State-gov","Self-emp-inc","Federal-gov","Without-pay","Never-worked"])
st.write(f"The workclass is {workclass}")

# 3. Education
education_options = [
    "HS-grad",
    "Some-college",
    "Bachelor",
    "Masters",
    "Assoc-voc",
    "11th",
    "Assoc-acdm",
    "10th",
    "7th-8th",
    "Prof-school",
    "9th",
    "12th",
    "Doctorate",
    "5th-6th",
    "1st-4th",
    "Preschool"
]

education = st.selectbox("Education",education_options)

st.write(f"The Education is {education}")


# 4. Marital Status 
maritalst = st.selectbox("Marital Status",["Married-civ-spouse",
                         "Never-married",
                         "Divorced",
                         "Separated",
                         "Widowed",
                         "Married-spouse-absent",
                         "Married-AF-spouse"])

st.write(f"The marital status : {maritalst}")



# 5. Occupation
occupation_categories = [
    "Prof-specialty",
    "Craft-repair",
    "Exec-managerial",
    "Adm-clerical",
    "Sales",
    "Other-service",
    "Machine-op-inspct",
    "?",
    "Transport-moving",
    "Handlers-cleaners",
    "Farming-fishing",
    "Tech-support",
    "Protective-serv",
    "Priv-house-serv",
    "Armed-Forces"
]

occupation = st.selectbox(
    "Occupation",
    occupation_categories
)

st.write("Selected occupation:", occupation)


# 6. Relationship 
rel = st.selectbox("Relationship",["Husband",
                                   "Not-in-family",
                                   "Own-child",
                                   "Unmarried",
                                   "Wife",
                                   "Other-relative"])

st.write(f"The relationshiop : {rel}")


# 7. Race 
race = st.selectbox("Race",["White",
                            "Black",
                            "Asian-Pac-Islander",
                            "Amer-Indian-Eskimo",
                            "Other"])

st.write(f"The Race is {race}")


# 8. Sex
sex = st.radio("Enter the sex ",["Male","Female"])
st.write(f"Sex : {sex}")


# 9. Capital gain
gain = st.number_input("Capital gain",min_value = 0,
                       max_value = 100000,
                       value = 0,
                       step = 1)

st.write(f"The Capital gain is {gain}")


# 10. Capital loss
loss = st.number_input("Capital loss",min_value = 0,
                       max_value = 5000,
                       value = 0,
                       step = 1)

st.write(f"The Capital loss {loss}")


# 11. Hours per weak 

hours = st.number_input("Hours Per Weak",min_value = 0,
                        max_value = 100,
                        value = 67,
                        step = 1)

st.write(f"Hours Per Weak {hours}")


# 12 . native country 
native_country_categories = [
    "United-States",
    "Mexico",
    "?",
    "Philippines",
    "Germany",
    "Canada",
    "Puerto-Rico",
    "El-Salvador",
    "India",
    "Cuba",
    "England",
    "Jamaica",
    "South",
    "China",
    "Italy",
    "Dominican-Republic",
    "Vietnam",
    "Guatemala",
    "Japan",
    "Poland",
    "Columbia",
    "Taiwan",
    "Haiti",
    "Iran",
    "Portugal",
    "Nicaragua",
    "Peru",
    "Greece",
    "France",
    "Ecuador",
    "Ireland",
    "Hong",
    "Cambodia",
    "Trinadad&Tobago",
    "Laos",
    "Thailand",
    "Yugoslavia",
    "Outlying-US(Guam-USVI-etc)",
    "Hungary",
    "Honduras",
    "Scotland",
    "Holand-Netherlands"
]

native_country = st.selectbox(
    "Native Country",
    native_country_categories
)

st.write(f"Country : {native_country}")



# create Total capital income
total_capital = gain - loss



#---------------------------------------------------
## STEP -> 5. Convert the inputs into datasets
#---------------------------------------------------

input_data = pd.DataFrame({
    "age":[age],
    "workclass":[workclass],
    "education":[education],
    "marital.status":[maritalst],
    "occupation":[occupation],
    "relationship":[rel],
    "race":[race],
    "sex":[sex],
    "capital.gain":[gain],
    "capital.loss":[loss],
    "hours.per.week":[hours],
    "native.country":[native_country],
    "total.capital": [total_capital]
})

# Encoding 

#education 
education_mapping = {
        "HS-grad": 0,
        "Some-college": 1,
        "Bachelors": 2,
        "Masters": 3,
        "Assoc-voc": 4,
        "11th": 5,
        "Assoc-acdm": 6,
        "10th": 7,
        "7th-8th": 8,
        "9th": 9,
        "12th": 10,
        "Prof-school": 11,
        "5th-6th": 12,
        "Doctorate": 13,
        "1st-4th": 14,
        "Preschool": 15
    }

input_data["education"] = input_data["education"].map(
        education_mapping
    )

# sex 

input_data["sex"] = input_data["sex"].map({
        "Female": 0,
        "Male": 1
    })


# rest of the features one hot 
categorical_columns = [
        "workclass",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "native.country"
    ]

input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

model_features = model.feature_name_

input_encoded = input_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

# scailing
numerical_columns = [
        "age",
        "education",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

input_encoded[numerical_columns] = scaler.transform(
        input_encoded[numerical_columns]
    )




#-----------------------
## STEP.6 -> Prediction
#-----------------------

if st.button("🚀 Predict Salary "):

    prediction = model.predict(input_encoded)

    probability = model.predict_proba(input_encoded)[0][1]

    st.write(f"Name : {name}")

    if prediction[0] == 1:

        st.success("Income : >50k")
        
    else:
        st.error("Income : <=50k")
        
    st.metric(
        "Salary Probability ",
        f"{probability * 100:.2f}%"
    )