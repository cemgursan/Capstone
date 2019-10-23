import pandas as pd
import numpy as np
import os
import quandl
import time
import math
import datetime
from bokeh.plotting import figure, output_file, show

from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# select a palette
from itertools import cycle
colorlist=["#393b79","#5254a3","#6b6ecf","#9c9ede","#637939","#8ca252","#b5cf6b","#cedb9c","#8c6d31", 
"#bd9e39", "#e7ba52","#e7cb94","#843c39", "#ad494a","#d6616b","#e7969c","#7b4173","#a55194", "#ce6dbd","#de9ed6"]


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline
from matplotlib import style
from pathlib import Path

import io
from io import BytesIO
import base64

style.use("ggplot")


quandl.ApiConfig.api_key = "YOUR API KEY HERE"




def SetupStocks(stockname1,stockname2,stockname3):
    #Initialize dataframe with chosen stocks
    # Version 1.0 = Only 3 stocks used interactively
    data= quandl.get(stockname1, trim_start="2010-12-12", trim_end="2019-01-01")
    data1= quandl.get(stockname2, trim_start="2010-12-12", trim_end="2019-01-01")
    data2=quandl.get(stockname3, trim_start="2010-12-12", trim_end="2019-01-01")
    data=pd.DataFrame(data["Adj. Close"])
    data1=pd.DataFrame(data1["Adj. Close"])
    data2=pd.DataFrame(data2["Adj. Close"])
    data.columns=["Coca Cola"]
    data1.columns=["Apple"]
    data2.columns=["Google"]
    DF= pd.merge(data, data1,  left_index=True, right_index=True)
    DF= pd.merge(DF, data2,  left_index=True, right_index=True)
    return DF

    
def PlotMAStockPrice(df,stockname,ma1,ma2,ma3):
    #Moving Average Plot
    # Version 1.0 = only 10,20,30
    ma_day=[ma1,ma2,ma3]
    dfTemp=df.copy()

    for ma in ma_day:
        column_name="MA %s days for %s" %(str(ma),stockname)
        dfTemp[column_name]=df[stockname].rolling(ma).mean()
    
    
    p = figure(plot_width=800, plot_height=800,x_axis_type='datetime')
    p.line(pd.to_datetime(dfTemp.index), dfTemp[stockname], line_width=2, line_color="blue" , legend=stockname)
    p.line(pd.to_datetime(dfTemp.index), dfTemp['MA '+str(ma1)+' days for ' +stockname], line_width=2, line_color="red", legend='MA '+str(ma1)+' days for ' +stockname)
    p.line(pd.to_datetime(dfTemp.index), dfTemp['MA '+str(ma2)+' days for ' +stockname], line_width=2, line_color="green", legend='MA '+str(ma2)+' days for ' +stockname)
    p.line(pd.to_datetime(dfTemp.index), dfTemp['MA '+str(ma3)+' days for ' +stockname], line_width=2, line_color="purple", legend= 'MA '+str(ma3)+' days for ' +stockname)

    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    return p



def JointPlot(df,stockname1,stockname2):
    # Stock Prices jointplot by seaborn

    stock_return = df.pct_change()
    sns_plot = sns.jointplot(df[stockname1],df[stockname2],stock_return,kind='scatter',color='seagreen')
    return sns_plot
    


def PairPlot(df):
    # Return Percentage's pairplot by seaborn
    stock_return = df.pct_change()
    returns= stock_return.dropna()
    sns_plot = sns.pairplot(returns)
    return sns_plot




def MonteCarloSimulation(df,stockname1,days):
    
    stock_return = df.pct_change()
    returns= stock_return.dropna()
    
    #Time horizon 
    t=days

    #Delta Time
    dt=1/t

    #Mean or expected return
    mu=returns.mean()[stockname1]

    #Standard deviation
    sigma=returns.std()[stockname1]
    
    startprice = df[stockname1].head(1)
    startprice = startprice[0]
    
    #initiate price array
    
    price=np.zeros(days)
    price[0]=startprice
    #Shock and Drift
    shock=np.zeros(days)
    drift=np.zeros(days)
    
    for x in range(1,days):
        
        #Calculate Shock
        #Scale could be 1.0 however std*delta t waves
        #Loc is where where is drift = mu * dt
        
        shock[x]=np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        
        #Calculate Drift
        drift[x]= mu*dt
        
        #Calculate Price
        price[x]= price[x-1]+ (price[x-1]*(drift[x]+shock[x]))
        
    return price

def MonteCarloGraph(df,stockname1,days,simulationruns):
    # The function above runs to get one price
    # I will use for loop to to get a price array within this function
    # Thus I can plot various pathways that Monte Carlo can reveal
    p = figure(plot_width=800, plot_height=800,x_axis_type='datetime')
    base = pd.to_datetime(df.index[-1])
    date_list = [base + datetime.timedelta(days=x) for x in range(days)]
    p = figure(plot_width=800, plot_height=800,x_axis_type='datetime')

    # I want different colors  for the lines, this zip trick helps to iterate through 20 colors 
    zip_list= zip(range(simulationruns), cycle(colorlist)) if simulationruns > len(colorlist) else zip(cycle(range(simulationruns)), colorlist)

    for run in zip_list:
        p.line(date_list, MonteCarloSimulation(df,stockname1,days), line_width=2, line_color=run[1] )
    return p
    

def MonteCarloDistribution(df,stockname1,days,simulationruns,quantile):
    
    simulation = np.zeros(simulationruns)
    
    startprice = df[stockname1].head(1)
    
    
    for run in range(simulationruns):
        simulation[run] = MonteCarloSimulation(df,stockname1,days)[days-1]
  
    # 5% imperical quantile for simulations list : Change quantile for differen confidence intervals
    q=np.percentile(simulation,quantile)

    #ec black makes histogram show vertical lines
    MontePlot = plt.figure()
    plt.hist(simulation,bins=200,histtype='bar',ec='black')

    # Using plt.figtext to fill in some additional information onto the plot

    # Starting Price
    plt.figtext(0.6, 0.8, s="Start price: $%.2f" %startprice[0])
    # Mean ending price
    plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulation.mean())

    # Variance of the price (within 95% confidence interval)
    plt.figtext(0.6, 0.6, "VaR($%.2f): $%.2f" % (1-quantile/100,startprice[0] - q,))

    # Display 5% quantile
    plt.figtext(0.15, 0.6, "q($%.2f): $%.2f" % (1-quantile/100,q))

    # Plot a line at the 5% quantile result
    plt.axvline(x=q, linewidth=4, color='r')

    # Title
    return MontePlot


def SetupStockML(stockname1):
    
    DF = quandl.get(stockname1, trim_start="2010-12-12", trim_end="2019-01-01")
    
    

    DF= DF[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

    DF["High_Low Percentage"] = (DF["Adj. High"] - DF["Adj. Close"])/ DF["Adj. Close"] * 100

    DF["Daily Change Percentage"] = (DF["Adj. Close"] - DF["Adj. Open"])/ DF["Adj. Open"] * 100

    # Features are here
    DF = DF[["Adj. Close", "High_Low Percentage","Daily Change Percentage", "Adj. Volume" ]]


    
    forecast_column = "Adj. Close"
    DF.fillna(-99999, inplace=True)

    # Forecast 1 % of the data I have
    # This variable identifies the number of days which would be forecasted!
    forecasted_days= int(math.ceil(0.05*len(DF)))

    #Label is here
    # We need minus shift because in this way we will go 10% to the future instead of backwards
    # we extended the forecast this way
    # Basically Label is 10% days to the future from Adj. Close

    DF["Label"] = DF[forecast_column].shift(-forecasted_days)
    
    
    # Let's train test and predict
    # X = features set
    # Y = Label Set
    X = np.array(DF.drop(["Label"],1))
    # Predict against this x lately we dont have a y value for these yet. we will forecast these !
    X = preprocessing.scale(X)

    X = X[:-forecasted_days]
    # 30 days of data is here we can use this data to predict
    X_lately = X[-forecasted_days:]

    # Use later steps for building test and train from your previous scikit project!
    # therefore, scaling the data brings all your values onto one scale eliminating the sparsity.
    # In regards to know how it works in mathematical detail, this follows the same concept of 
    #Normalization and Standardization

    DF.dropna(inplace=True)
    y = np.array(DF["Label"])
    #y = np.array(DF["Label"])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = LinearRegression()
    # support vector regression
    # classifier = svm.SVR()

    classifier.fit(X_train,y_train)
    # lets see accuracy

    accuracy = classifier.score(X_test,y_test)
    #print(accuracy)
    # squared error ! is accuracy
    

    
    # Lets predict 
    # We can use predict with a single variable or array.
    # We pass this variable to our model_
    # Then our model predicts with those features!

    forecast_set = classifier.predict(X_lately)

    
    
    #null first we will forecast on this column
    DF["Forecast"] = np.nan
    # date calculations
    lastday= DF.iloc[-1].name
    lastunix= lastday.timestamp()
    onedaysecs= 86400
    nextunix= lastunix+onedaysecs

    for i in forecast_set:
        nextdate = datetime.datetime.fromtimestamp(nextunix)
        nextunix+=onedaysecs
        DF.loc[nextdate] = [np.nan for count in range(len(DF.columns)-1)] + [i] 


    

 


    y1=pd.Series(DF["Adj. Close"])
    y2=pd.Series(DF["Forecast"])
    yearsforpoints =[y1.last_valid_index(),y2.first_valid_index()]

    x1=DF.loc[DF["Adj. Close"] == DF["Adj. Close"][y1.last_valid_index()]]
    x2=DF.loc[DF["Forecast"] == DF["Forecast"][y2.first_valid_index()]]
    twopoints=[x1["Adj. Close"],x2["Forecast"]]

    titleforplot= "Accuracy is " + str(accuracy)
    p = figure(plot_width=800, plot_height=800,x_axis_type='datetime',title= titleforplot, toolbar_location="above")
    p.line(pd.to_datetime(DF.index), DF["Adj. Close"], line_width=2, line_color="blue" , legend=stockname1)
    p.line(pd.to_datetime(DF.index), DF["Forecast"], line_width=2, line_color="red" , legend="Forecast")
    p.line(yearsforpoints, twopoints, line_width=2, line_color="red",legend="Forecast")

    p.legend.location = "top_left"
    p.legend.click_policy="hide"

    return p

    
