# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:52:58 2022

@author: Esra
"""


!pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns",None)
# pd.set_option("display.max_rows",None)
pd.set_option("display.float_format",lambda x:"%.5f" %x)


df_=pd.read_csv("flo_data_20k.csv")


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit.round(), up_limit.round()


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df=df_.copy()
df.head()


replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")
replace_with_thresholds(df, "customer_value_total_ever_offline")


df["order_num_total"]=df["order_num_total_ever_offline"]+df["order_num_total_ever_online"]
df["customer_value_total"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]


df.info()
df.first_order_date=df.first_order_date.apply(pd.to_datetime)
df.last_order_date=df.last_order_date.apply(pd.to_datetime)
df.last_order_date_online=df.last_order_date_online.apply(pd.to_datetime)
df.last_order_date_offline=df.last_order_date_offline.apply(pd.to_datetime)



df.last_order_date.max()
analysis_date=dt.datetime(2021,6,1)


cltv_df=pd.DataFrame()
cltv_df["Customer_id"]=df["master_id"]
cltv_df["recency_cltv_weekly"]=((df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["T_weekly"]=((analysis_date-df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"]=df["order_num_total"]
cltv_df["monetary_cltv_avg"]=df["customer_value_total"]/df["order_num_total"]
cltv_df.head()



#BG/NBD (Expected Number of Transaction)
bgf=BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

cltv_df["exp_sales_3_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"])

cltv_df["exp_sales_6_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"])
cltv_df.head()



# Gamma Gamma Submodel (Expected Average Profit)
ggf=GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"]=ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary_cltv_avg"])



#Cltv hesaplanması
cltv=ggf.customer_lifetime_value(bgf,
                                 cltv_df["frequency"],
                                 cltv_df["recency_cltv_weekly"],
                                 cltv_df["T_weekly"],
                                 cltv_df["monetary_cltv_avg"],
                                 time=6,
                                 freq="W",
                                 discount_rate=0.01)



cltv=ggf.customer_lifetime_value(bgf,
                                 cltv_df["frequency"],
                                 cltv_df["recency_cltv_weekly"],
                                 cltv_df["T_weekly"],
                                 cltv_df["monetary_cltv_avg"],
                                 time=6,
                                 freq="W",
                                 discount_rate=0.01).sort_values(ascending=False).head(20)

cltv.head()
cltv_df["cltv"] = cltv

#CLTV'ye göre segmentlerin oluşturulması
cltv_df["segment"]=pd.qcut(cltv_df["cltv"],4,labels=["D","C","B","A"])
cltv.head()


cltv.groupby("segment").agg({"count","mean","sum"})


#Tüm sürecin fonksiyonlaştırılması

def create_cltv_df(df):
    
    replace_with_thresholds(df, "order_num_total_ever_online")
    replace_with_thresholds(df, "order_num_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_online")
    replace_with_thresholds(df, "customer_value_total_ever_offline")
    
    df["order_num_total"]=df["order_num_total_ever_offline"]+df["order_num_total_ever_online"]
    df["customer_value_total"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]
    
    df.first_order_date=df.first_order_date.apply(pd.to_datetime)
    df.last_order_date=df.last_order_date.apply(pd.to_datetime)
    df.last_order_date_online=df.last_order_date_online.apply(pd.to_datetime)
    df.last_order_date_offline=df.last_order_date_offline.apply(pd.to_datetime)
    
    df.last_order_date.max()
    analysis_date=dt.datetime(2021,6,1)
    
    cltv_df=pd.DataFrame()
    cltv_df["Customer_id"]=df["master_id"]
    cltv_df["recency_cltv_weekly"]=((df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]'))/7
    cltv_df["T_weekly"]=((analysis_date-df["first_order_date"]).astype('timedelta64[D]'))/7
    cltv_df["frequency"]=df["order_num_total"]
    cltv_df["monetary_cltv_avg"]=df["customer_value_total"]/df["order_num_total"]
    
    bgf=BetaGeoFitter(penalizer_coef=0.001)
    
    bgf.fit(cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"])
    
    cltv_df["exp_sales_3_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                                         cltv_df["frequency"],
                                                                                         cltv_df["recency_cltv_weekly"],
                                                                                         cltv_df["T_weekly"])
    
    cltv_df["exp_sales_6_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                                                         cltv_df["frequency"],
                                                                                         cltv_df["recency_cltv_weekly"],
                                                                                         cltv_df["T_weekly"])
    
    ggf=GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])
    cltv_df["exp_average_value"]=ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                         cltv_df["monetary_cltv_avg"])
    
    cltv=ggf.customer_lifetime_value(bgf,
                                     cltv_df["frequency"],
                                     cltv_df["recency_cltv_weekly"],
                                     cltv_df["T_weekly"],
                                     cltv_df["monetary_cltv_avg"],
                                     time=6,
                                     freq="W",
                                     discount_rate=0.01)
    
    cltv_df["cltv"] = cltv
    cltv_df["segment"]=pd.qcut(cltv_df["cltv"],4,labels=["D","C","B","A"])
        
    return cltv_df
    
    
cltv_df=create_cltv_df(df)
cltv_df.head(10)
















