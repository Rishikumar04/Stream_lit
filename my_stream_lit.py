import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
from pandas_profiling import ProfileReport
from category_encoders import LeaveOneOutEncoder
from sklearn.linear_model import LinearRegression, Lasso, Lars, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from mlxtend.regressor import StackingCVRegressor
import streamlit as st

Linear_Regression = LinearRegression()
Lasso             = Lasso()
Lars              = Lars()
Ridge             = Ridge()
Elastic_Net       = ElasticNet()
KNN               = KNeighborsRegressor()
SVM               = SVR()
Decision_Tree     = DecisionTreeRegressor()
Bagging           = BaggingRegressor()
RandomForest      = RandomForestRegressor()
AdaBoost          = AdaBoostRegressor()
GradientBoosting  = GradientBoostingRegressor()
LGBM              = LGBMRegressor()
XGB               = XGBRegressor()
CAT_Boost         = CatBoostRegressor()
esti = [('el', Elastic_Net), ('cat', CAT_Boost)]

vote = VotingRegressor(estimators=esti)

re1 = ElasticNet()
re2 = CatBoostRegressor()
stack = StackingCVRegressor(regressors=[re1, re2], meta_regressor=RandomForest)

        
list_model = [Linear_Regression, Lasso, Lars, Ridge, Elastic_Net, KNN, SVM, Decision_Tree, Bagging, RandomForest,
             AdaBoost, GradientBoosting, LGBM, XGB, CAT_Boost, vote, stack]



st.set_page_config(layout="wide")
st.title('Hello! This app helps to predict hospital expensive')


st.image('health image.jpeg')


df = pd.read_csv('Hospital_-_Data.xls') #Importing Data

if st.checkbox('Show Table', value = False):
    st.dataframe(df)

expander_bar = st.beta_expander("Tap to See Input Field Details")
expander_bar.markdown("""
* **AGE**	Age of the patient
* **GENDER** 	Gender code for patient 
* **MARITAL STATUS** 	Marital Status of the patient:
* **KEY COMPLAINTS CODE**	Codes given to the key complaints faced by the patient:
* **BODY WEIGHT**	Weight of the patient
* **BODY HEIGHT**	Height of the patient
* **HR PULSE	**  Pulse of patient at the time of admission
* **BP-HIGH**	High BP of patient (Systolic)
* **BP-LOW** 	Low BP of patient (Diastolic)
* **RR**	Respiratory rate of patient
* **PAST MEDICAL HISTORY CODE**	Code given to the past medical history of the patient:
* **HB**	Hemoglobin count of patient
* **UREA**	Urea levels of patient
* **CREATININE** 	Creatinine levels of patient
* **MODE OF ARRIVAL**	Way in which the patient arrived the hospital:
* **STATE AT THE TIME OF ARRIVAL** 	State in which the patient arrived:
* **TYPE OF ADMISSION**	Type of admission for the patient:
* **TOTAL LENGTH OF STAY	** Number of days patient stayed in the hospital
* **LENGTH OF STAY-ICU** 	Number of days patient stayed in the ICU
* **LENGTH OF STAY-WARD**	Number of days patient stayed in the ward
* **IMPLANT USED (Y/N)** 	Any implant done on the patient
* **COST OF IMPLANT**	Total cost of all the implants done on the patient if any
                      """)
 

    

def impute():
    null_index = df[df['KEY COMPLAINTS -CODE'].isna()].index
    for i in null_index:
        if df.loc[i,'MARITAL STATUS'] == 'MARRIED':
            df.loc[i, 'KEY COMPLAINTS -CODE'] = 'CAD-DVD'
        else:
             df.loc[i, 'KEY COMPLAINTS -CODE'] = 'other- heart'
                
impute()

#Since teenage patients have null values in BP Column, I'm calculating mean values for teenage patients.
bp_mean = df[df['AGE']<=19].aggregate({'BP -HIGH': np.mean, 'BP-LOW': np.mean})

#Replacing null values.
df['BP -HIGH'].fillna(bp_mean[0],inplace=True)

#Replacing null values.
df['BP-LOW'].fillna(bp_mean[1],inplace=True)

#Replacing null values by its respective mean value.
df['HB'].fillna(df['HB'].mean(), inplace=True)

df['CREATININE'].fillna(df['CREATININE'].mode()[0], inplace=True)

df['UREA'].fillna(df['UREA'].mean(), inplace=True)

df_drop = df.drop(['SL.', 'PAST MEDICAL HISTORY CODE'], axis=1)


cat_col = df_drop.select_dtypes(exclude=np.number).columns

#Leave one out encoder.
le = LeaveOneOutEncoder()

df_drop[cat_col] = le.fit_transform(X = df_drop[cat_col], y = df_drop['TOTAL COST TO HOSPITAL '])
#Train test split
X = df_drop.drop('TOTAL COST TO HOSPITAL ', axis=1)
y = df_drop['TOTAL COST TO HOSPITAL ']



#Page Layout:

col1 = st.sidebar
col2, col3 = st.beta_columns((1,1))

empty = pd.DataFrame(columns = X.columns)


#Manual Input
age_val                          = col1.slider('Age of the Patient', 0, 120, 30)
gen_val                        = col1.selectbox(' Select Gender of Patient', ('Male', 'Female'))
mar_val                = col1.radio(' Select Marital Status', ('Married', 'Unmarried'))
key_comp_val          = col1.selectbox(' Select Key Compliance Code', ('ACHD', 'CAD-DVD', 'CAD-SVD', 'CAD-TVD', 'CAD-VSD', 'OS-ASD', 'other-heart', 'other-respiratory', 'other-general', 'other-nervous', 'other-tertalogy', 'PM-VSD', 'RHD'))
weigt_val                  = col1.slider('Patient Weight in Kg', 0.0, 150.0, 50.0, 0.5)
heigt_val                   = col1.slider('Patient Height in cm', 0, 250, 145)
pulse_val                      = col1.slider('Patient Pulse at time of Admission', 0, 200, 90)
bp_high_val                      = col1.slider('Patient High BP Value', 50, 250, 110)
bp_low_val                       = col1.slider('Patient Low BP Value', 20, 150, 80)
rr_val                            = col1.slider('Respiratory rate of patient', 10, 60, 25)
mode_of_arrival_val                = col1.radio('Select Mode of Arrival', ('Ambulance', 'Walked In', 'Transferred'))
admns_val                  = col1.selectbox('Type of Admission',('Elective','Emergency'))
time_of_arrival_val  = col1.radio('Select State in which patient Arrived', ('Alert', 'Confused'))
hb_val                 = col1.slider('Hemoglobin count of Patient', 0, 30, 13)
urea_val          = col1.slider('Urea Levels of the patient', 1, 150, 22)
creatine_val                   = col1.slider('Creatinine levels of patient', 0.1, 6.0, 1.0, 0.1)
stay_val          = col1.slider('Length of stay of the patient in the hospital', 0, 50, 2)
icu_val          = col1.slider('Number of days patient stayed in the ICU', 0, 50, 2)
ward_val           = col1.slider('Number of days patient stayed in the ward', 0, 50, 2)
if col1.checkbox('IMPLANT', value=None):
    implant_val = 'Y'
    cost_implant = col1.slider('Total cost of all the implants done on the patient if any', 10, 100000, 1000, 100)
else:
    implant_val = 'N'
    cost_implant = 0
algo              = col1.radio( "Choose Your Algorithm for Prediction", 
                         ('LinearRegression', 'Lasso', 'Lars', 'Ridge', 'ElasticNet','SVR', 'KNN', 'DecisionTree', 'Bagging' 'RandomForest', 'Ada Boost', 'GradientBoost', 'LGBM', 'XGBoost', 'CAT Boost', 'Voting Regressor', 'Stacking Algorithm'))
#col1.dataframe(empty)

data = {'AGE':age_val, 
        'GENDER':gen_val,
        'MARITAL STATUS':mar_val,
        'KEY COMPLAINTS -CODE':key_comp_val, 
        'BODY_WEIGHT':weigt_val, 
        'BODY HEIGHT':heigt_val, 
        'HR PULSE':pulse_val, 
        'BP -HIGH':bp_high_val, 
        'BP-LOW': bp_low_val, 
        'RR': rr_val, 
        'MODE OF ARRIVAL': mode_of_arrival_val, 
        'TYPE OF ADMSN':admns_val, 
        'STATE AT THE TIME OF ARRIVAL': time_of_arrival_val, 
        'HB': hb_val, 
        'UREA':urea_val, 
        'CREATININE': creatine_val,
        'TOTAL LENGTH OF STAY':stay_val, 
        'LENGTH OF STAY - ICU':icu_val, 
        'LENGTH OF STAY- WARD':ward_val,
       'IMPLANT USED (Y/N)': implant_val,
        'COST OF IMPLANT': cost_implant
       }
inp = pd.DataFrame(data, index=[0])
inp[cat_col] = le.transform(X = inp[cat_col])



# Slider
with st.form('Form1'):
    col2.header('Model Performance ')
    col3.header(' ')
    col3.subheader(' ')
    
    col2.subheader('Train Test Split')
    split_val = col2.slider('Choose Train Test Split Percentage', 0.1, 0.6, 0.3, 0.01)
    col3.subheader('Random State')
    random_val = col3.slider('Choose Random State Value', 1, 150, 123, 1)
    
if col1.checkbox('Predict Expense', value = False):
    if algo ==  'LinearRegression':
           model = LinearRegression()
    elif algo ==  'Lasso':
           model = Lasso()
    elif algo ==  'Ridge':
           model = Ridge()
    elif algo ==  'ElasticNet':
           model = ElasticNet()
    elif algo ==  'Lars':
        model = Lars()
    elif algo ==  'SVR':
        model = SVR()
    elif algo ==  'KNN':
        model = KNeighborsRegressor()
    elif algo ==  'DecisionTree':
        model = DecisionTreeRegressor()
    elif algo ==  'Bagging':
        model = BaggingRegressor()
    elif algo ==  'RandomForest':
        model = RandomForestRegressor()
    elif algo ==  'Ada Boost':
        model = AdaBoostRegressor()
    elif algo ==  'GradientBoost':
        model = GradientBoostingRegressor()
    elif algo ==  'LGBM':
        model = LGBMRegressor()
    elif algo ==  'XGBoost':
        model = XGBRegressor()
    elif algo ==  'CAT Boost':
        model = CatBoostRegressor()
    elif algo ==  'Voting Regressor':
        model = vote
    elif algo ==  'Stacking Algorithm':
        model = stack
    col1.write('Running')
    model.fit(X, y)
    predcost = model.predict(inp)
    
    cost = round(predcost[0], 2) 

    col1.subheader('COST PREDICTION')
    col1.markdown('<font color="blue">As per given Data & Preference, Approx cost to Hospital in     INR</font>',unsafe_allow_html=True)
    col1.write (cost)
    
if st.checkbox('Run_Model', value = False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_val, random_state=random_val)




#metrics
    col4, col5 = st.beta_columns((1,1))
    
    def metrics(y_true, y_pred, val):
        
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        if val==0:
            col4.success(f'MAE:{mae}')
            col4.success(f'MAPE:{mape}')
            col4.success(f'MSE:{mse}')
            col4.success(f'RMSE:{rmse}')
            col4.success(f'R Squared:{r2}')
        else:
            col5.success(f'MAE:{mae}')
            col5.success(f'MAPE:{mape}')
            col5.success(f'MSE:{mse}')
            col5.success(f'RMSE:{rmse}')
            col5.success(f'R Squared:{r2}')
    
    def predictions(model, mod_name = None, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, full = False, test = inp):
    
    
        model.fit(X_train, y_train)
    
        train_pred = model.predict(X_train)
    
        test_pred = model.predict(X_test)
    
        actual = [y_train, y_test]
    
        pred = [train_pred, test_pred]
    
        data = ['Train Score', 'Test Score']
        
        if full:
            
            pred = model.predict(test)
            
            col1.write(f'Expenses{pred}')
            
            return pred
        
           
        for i in range(2):
        
            if i==0:
                col4.header(mod_name)
                col4.subheader(data[i])
                
            else:
                col5.header(mod_name)
                col5.subheader(data[i])
            
            metrics(actual[i], pred[i], i)
            
              
    for i in list_model:

        
        if 'VotingRegressor' in str(i):
            mod_name = 'VotingRegressor'
        elif 'StackingCVRegressor' in str(i):
            mod_name = 'StackingRegressor'
        elif 'Cat' in str(i):
            mod_name = str(i)[15:32]
        else:
            mod_name = str(i).split('(')[0]
        predictions(i, mod_name)
    
   

    st.balloons()

