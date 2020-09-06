################# FORECASTING CocaCola Sales  ###########################

##### IMPORTING the required packages and LOADING the requird Dataset ########
import numpy as np
import pandas as pd
import xlrd
import xlwt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
CocaCola= pd.read_csv(r"C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Forecasting\\CocaCola_Data.csv")


#CocaCola_Sales_Rawdatacsv
#Data=pd.DataFrame(CocaCola_Sales_Rawdatacsv)

CocaCola["Sales"]=CocaCola["Sales"].astype(int)
CocaCola
CocaCola.Sales.plot()


### so we have basically 4 Quarters in an year

quarter=['Q1','Q2','Q3','Q4']
n=CocaCola['Quarter'][0]
n[0:2] # 'Q1', indexing 0 and 1 of Q1_86

CocaCola['quarter']=0 # Adding a new column in the data and assigning all values with 0



for i in range(42):
    n=CocaCola['Quarter'][i]
    CocaCola['quarter'][i]=n[0:2]#extracting the Quarters and putting in "quarter" column

dummy=pd.DataFrame(pd.get_dummies(CocaCola['quarter']))#Creating a dataframe of dummy variables of "quarter"

Coca=pd.concat((CocaCola,dummy),axis=1)#Concating the CocaCola and dummy Dataframes together and putting in object Coca
t= np.arange(1,43)
Coca['t']=t # Making a new column "t" and arranging numbers equal to length of data frame

##or##
#Coca["t"]=np.arrange(1,43)



Coca['t_square']=Coca['t']*Coca['t']

#log_Sales=np.log(Coca['Sales'])
#coco['log_Sales']=log_Sales

Coca["log_Sales"] = np.log(Coca["Sales"])
Coca.columns
#Index(['Quarter', 'Sales', 'quarter', 'Q1', 'Q2', 'Q3', 'Q4', 't', 't_square',
 #      'log_Sales']


#Sales=Coca["Sales"].astype(int)
#Sales=pd.DataFrame(Sales)


#Coca['Sales'].round(decimals=0)


#### SPLITTING THE DATA INTO TEST AND TRAIN ########
train= Coca.head(38)
test=Coca.tail(4)
plt.plot(test['Sales'])
Coca.Sales.plot()





########################################################################
################### M O D E L     B U I L D I N G ######################
########################################################################



##################### LINEAR MODEL #########################
linear= smf.ols('Sales~t',data=train).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmselin=np.sqrt((np.mean(np.array(test['Sales'])-np.array(predlin))**2))
rmselin#421.1787876367789

#################### QUADRATIC MODEL #####################
quad=smf.ols('Sales~t+t_square',data=train).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_square']])))
rmsequad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predquad))**2))
rmsequad #475.56183519821434

######################### EXPONENTIAL MODEL #################
expo=smf.ols('log_Sales~t',data=train).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test['t'])))
predexp
rmseexpo=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predexp)))**2))
rmseexpo #466.247973132103

######################## ADDITIVE SEASONALITY ##################
additive= smf.ols('Sales~ Q1+Q2+Q3+Q4',data=train).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))
predadd
rmseadd=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predadd))**2))
rmseadd #1860.0238154374447

###################### ADDITIVE SEASONALITY WITH LINEAR TREND ###########
addlinear= smf.ols('Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))
predaddlinear

rmseaddlinear=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddlinear))**2))
rmseaddlinear #464.98290242804075

################## ADDITIVE SEASONALITY WITH QUADRATIC TREND ###############
addquad=smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=train).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test[['t','t_square','Q1','Q2','Q3','Q4']])))
rmseaddquad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddquad))**2))
rmseaddquad #301.7380072145719

################### MULTIPLICATIVE SEASONALITY ########################
mulsea=smf.ols('log_Sales~Q1+Q2+Q3+Q4',data=train).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))
rmsemul= np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul #1963.3896400563397

################## MULTIPLICATIVE SEASONALITY WITH LINEAR TREND ############
mullin= smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
predmullin= pd.Series(mullin.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))
rmsemulin=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmullin)))**2))
rmsemulin #225.5243905617087

################# MULTIPLICATIVE SEASONALITY WITH QUADRATIC TREND ###########
mul_quad= smf.ols('log_Sales~t+t_square+Q1+Q2+Q3+Q4',data=train).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test[['t','t_square','Q1','Q2','Q3','Q4']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mul_quad)))**2))
rmse_mul_quad # 581.8457189224152



data={'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsemulin','rmsequad']),'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsemulin,rmsequad])}
data

Rmse=pd.DataFrame(data)
Rmse
#           Model       Values
#0  rmse_mul_quad   581.845719
#1        rmseadd  1860.023815
#2  rmseaddlinear   464.982902
#3    rmseaddquad   301.738007
#4       rmseexpo   466.247973
#5        rmselin   421.178788
#6        rmsemul  1963.389640
#7      rmsemulin   225.524391
#8       rmsequad   475.561835

# Therefore the Multiplicative Additive Seasonality have the least mean squared error
#final model with least rmse value


final= smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=Coca).fit()
train_pred= pd.Series(final.predict(train))
actual_pred = np.exp(train_pred)
actual_pred


test_pred= pd.Series(final.predict(test))
actual_pred1 = np.exp(test_pred)
actual_pred1




# Accuracy = Test
np.mean(predmullin==test.log_Sales) #  0.0
 
test["Forecasted_Sales"]=pd.Series(actual_pred1)


Coca["Forecasted_Sales"]=pd.Series(actual_pred1)






# Accuracy = train 

np.mean(predmullin == train.log_Sales)
Coca["Forecasted_Sales"]=pd.Series(actual_pred)




























#from sm.tsa.statespace import sa



# Boxplot for ever
sns.boxplot("Sales",data=Coca)

#sns.factorplot("Quarter","Sales",data=cocola,kind="box")

# moving average for the time series to understand better about the trend character in Amtrak
Coca.Sales.plot(label="org")
for i in range(2,10,2):
    Coca["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=2)

# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(Coca.Sales,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(Coca.Sales,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(Coca.Sales,lags=10)
tsa_plots.plot_pacf(Coca.Sales)





# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(train["Sales"]).fit()
pred_ses = ses_model.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_ses,test.Sales) # 8.272015971453765

# Holt method 
hw_model = Holt(train["Sales"]).fit()
pred_hw = hw_model.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_hw,test.Sales) #  8.820760666027857



# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(train["Sales"],seasonal="add",trend="add",seasonal_periods=4,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_hwe_add_add,test.Sales)#  1.4893772347227439



# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_hwe_mul_add,test.Sales) # 1.7781059176091492



# Visualization of Forecasted values for Test data set using different methods 
plt.plot(train.index, train["Sales"], label='Train',color="black")
plt.plot(test.index, test["Sales"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.legend(loc='best')
