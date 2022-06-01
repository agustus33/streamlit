import streamlit as st
import pickle
import numpy as np
import streamlit.components.v1 as components
import pandas as pd
model_one = pickle.load(open('anemia_new_model.pkl','rb'))
model_two = pickle.load(open('leukemia_model.pkl','rb'))
model_three = pickle.load(open('diabetes_model.pkl','rb'))
anemia_chart_csv = pd.read_csv("Food details.csv")
leukemia_chart_csv = pd.read_csv("leukemia_foodchart.csv")
diabetes_chart_csv = pd.read_csv("diabetes_foodchart.csv")
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


st.set_page_config(
    page_title="Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

def predict_anemia(sex,rbc,pcv,hgb):
    input = np.array([[sex,rbc,pcv,hgb]]).astype(np.float64)
    prediction = model_one.predict(input)
    return int(prediction)

def predict_leukemia(WBC, PLT):
    input = np.array([[WBC, PLT]]).astype(np.float64)
    leukemia_pred = model_two.predict(input)
    return int(leukemia_pred)
def predict_diabetes(Glucose, Bmi, Age):
    input = np.array([[Glucose, Bmi, Age]]).astype(np.float64)
    diabetes_pred = model_three.predict(input)
    return int(diabetes_pred)
def main():
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Blood Disease Prediction </h2>
    </div>
    """
    safe_html_one ="""  
        <div style="background-color:#80ff80; padding:10px >
        <h2 style="color:white;text-align:center;"> ANAMEIA POSITIVE</h2>
        </div>
        """
    warn_html_one ="""  
        <div style="background-color:#F4D03F; padding:10px >
        <h2 style="color:white;text-align:center;"> ANAMEIA NEGATIVE</h2>
        </div>
        """
    safe_html_two ="""  
        <div style="background-color:#80ff80; padding:10px >
        <h2 style="color:white;text-align:center;"> LEUKEMIA POSITIVE</h2>
        </div>
        """
    warn_html_two ="""  
        <div style="background-color:#F4D03F; padding:10px >
        <h2 style="color:white;text-align:center;"> LEUKEMIA NEGATIVE</h2>
        </div>
        """
    safe_html_three ="""  
        <div style="background-color:#80ff80; padding:10px >
        <h2 style="color:white;text-align:center;"> DIABETES POSITIVE</h2>
        </div>
        """
    warn_html_three ="""  
        <div style="background-color:#F4D03F; padding:10px >
        <h2 style="color:white;text-align:center;"> DIABETES NEGATIVE</h2>
        </div>
        """   
    st.markdown(html_temp, unsafe_allow_html = True)

    option = st.selectbox(
     'Select the disease to be predicted:',
     ('','Anemia', 'Leukemia', 'Diabetes'))
    if option == 'Anemia':
        sex = st.selectbox('Gender(1: Female, 0: Male)',('1', '0'))
        hgb = st.text_input("HGB")
        rbc = st.text_input("RBC")
        pcv = st.text_input("PCV")

        pred = st.button("Predict the disease")
        if pred :
            output = predict_anemia(sex,rbc,pcv,hgb)
            st.success('The result is {}'.format(output))
            if output == 1:
                st.markdown(safe_html_one,unsafe_allow_html=True)
                anemia_foodchart = open("Foodchart_Anemia.html", 'r', encoding='utf-8')
                source_code_anemia = anemia_foodchart.read() 
                components.html(source_code_anemia, height = 1400)
                df_anemia = convert_df(anemia_chart_csv)
                st.download_button(label="Download Anemia Report",data=df_anemia,file_name='anemia_foodchart.csv',mime='text/csv',)
            elif output == 0:
                st.markdown(warn_html_one,unsafe_allow_html=True)
    elif option == 'Leukemia' :
        WBC = st.text_input("WBC")
        PLT = st.text_input("PLT")
        pred = st.button("Predict the disease")
        if pred :
            output = predict_leukemia(WBC,PLT)

            st.success('The result is {}'.format(output))
            if output == 1:
                st.markdown(safe_html_two,unsafe_allow_html=True)
                leukemia_foodchart = open("Foodchart_Leukemia.html", 'r', encoding='utf-8')
                source_code_leukemia = leukemia_foodchart.read() 
                components.html(source_code_leukemia, height = 1400)
                df_leukemia = convert_df(leukemia_chart_csv)
                st.download_button(label="Download Leukemia Report",data=df_leukemia ,file_name='leukemia_foodchart.csv',mime='text/csv',)

            elif output == 0:
                st.markdown(warn_html_two,unsafe_allow_html=True)      

    elif option == 'Diabetes' :
        Glucose = st.text_input("Glucose")
        Bmi = st.text_input("Bmi")
        Age = st.text_input("Age")
        pred = st.button("Predict the disease")
        if pred :
            output = predict_diabetes(Glucose, Bmi, Age)

            st.success('The result is {}'.format(output))
            if output == 1:
                st.markdown(safe_html_three,unsafe_allow_html=True)
                diabetes_foodchart = open("Foodchart_Diabetes.html", 'r', encoding='utf-8')
                source_code_diabetes = diabetes_foodchart.read() 
                components.html(source_code_diabetes, height = 1400)
                df_diabetes = convert_df(diabetes_chart_csv)
                st.download_button(label="Download Diabetes Report",data=df_diabetes ,file_name='diabetes_foodchart.csv',mime='text/csv',)

            elif output == 0:
                st.markdown(warn_html_three,unsafe_allow_html=True)  


if __name__=='__main__':
    main()