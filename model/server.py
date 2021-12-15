import streamlit as st
from datetime import datetime 
import pandas as pd
import numpy as np
np.random.seed(71)
import re

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore")

def daysbin(x):
    if x<10:
        return 0
    elif x<20:
        return 1
    else:
        return 2
        
data=pd.read_csv('accalldata.csv')
school=pd.read_csv('schoolzone.csv',encoding='cp949')
breakrule=pd.read_csv('breakrule.csv')
manyaccident=pd.read_csv('manyaccident.csv')

breakrule.columns=['lat','lon']
manyaccident.columns=['lat','lon']

mapdata=pd.concat([breakrule,manyaccident])

school.loc[4,'위도']=37.530017
school.loc[43,'경도']=127.073328

data_copy=data.copy()
data_copy2=data_copy.copy()

def haversine(coord1, coord2):
    import math
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers

    meters = round(meters)
    km = round(km, 3)
    # print(f"Distance: {meters} m")
    # print(f"Distance: {km} km")
    return meters

for idx,(lon1,lat1) in enumerate(zip(data_copy2['경도'],data_copy2['위도'])):
    for lon2,lat2 in zip(school['경도'],school['위도']):
        meters=haversine(coord1=(lon1,lat1),coord2=(lon2,lat2))
        if meters<300:
            data_copy2.loc[idx,'schoolzone']=1
data2=data_copy2
data2=data2[data2['schoolzone']==1]

train=data2.iloc[:,np.r_[1,2,3,4,6,7,8,9,10,-2,-6]]

traindata=train.copy()
## 1 : 차대사람 2 : 차대차 3 : 차량단독
## 1 : 주간 2 : 야간
train['일_bin']=train['일'].apply(lambda x:daysbin(x))

le=LabelEncoder()

target=traindata['사고유형대분류']
traindata.drop('사고유형대분류',axis=1,inplace=True)

traindata['지역'] = le.fit_transform(traindata['지역'])

from imblearn.over_sampling import RandomOverSampler
rand_over_sample = RandomOverSampler(sampling_strategy='auto', random_state=41)
x_over, y_over = rand_over_sample.fit_resample(traindata, target)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_over,y_over, test_size=0.2, random_state=71)
# print(X_train.shape,y_train.shape)
# print("-------")
# print(X_test.shape,y_test.shape)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=10)

from xgboost import XGBClassifier
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
evals = [(X_test, y_test)]
xgb_wrapper.fit(X_train, y_train,early_stopping_rounds=10, eval_metric="merror", 
                eval_set=evals)

# model = RandomForestClassifier(n_estimators=10)
# model.fit(traindata, target)

from sklearn.metrics import plot_confusion_matrix,accuracy_score

def main():
    today=datetime.today()
    todaytime=today.strftime('%Y-%m-%d-%H')
    year=int(todaytime.split('-')[0])
    month=int(todaytime.split('-')[1])
    day=int(todaytime.split('-')[2])
    hour=int(todaytime.split('-')[3])
    weekday=today.weekday()
    m={0:2,1:3,2:4,3:5,4:6,5:7,6:1}
    weekday=m[weekday]

    st.title("스쿨존 교통사고 주의 대쉬보드")
    st.write(f"""
    {year}년 {month}월 {day}일
    """)

    my_page = st.sidebar.radio('Page Navigation', ['데이터 분석 및 시각화', '교통사고 예측 확률 결과 분석'])
    
    if my_page=='데이터 분석 및 시각화':
        kpi1,kpi2=st.columns(2)

        with kpi1:
            st.markdown("<h3 style='text-align:center'>최다사고유형</h3>",unsafe_allow_html=True)
            ## {month}월 어떤 사고가 많이 일어나는지.
            df=train.groupby(['월','사고유형대분류'])['사고유형대분류'].count()
            top_acc=df.loc[month].idxmax()
            count=df.loc[month].loc[top_acc]
            kind={1:'차대사람',2:'차대차',3:'차량단독'}
            st.markdown(f"<h4 style='text-align:center;color:red'>{kind[top_acc]} <span style='color:rgb(49, 51, 63);'>{count}회</span></h4>",unsafe_allow_html=True)
        with kpi2:
            st.markdown("<h3 style='text-align:center'>최다사고시간</h3>",unsafe_allow_html=True)
            ## 11월 주야구분 
            df_juya=train.groupby(['월','주야구분'])['사고유형대분류'].count()
            top_accj=df_juya.loc[month].idxmax()
            kind={1:'주간',2:'야간'}
            countj=df_juya.loc[month][top_acc]
            ## 11월 몇시에 사고가 가장 많이 나는지.
            df=train.groupby(['월','시간'])['사고유형대분류'].count()
            top_acc_time=df.loc[month].idxmax()
            count=df.loc[month][top_acc_time]
            st.markdown(f"<h4 style='text-align:center;color:red'>{kind[top_accj]} <span style='color:rgb(49, 51, 63);'>과 {top_acc_time}시</span><br><span style='font-size:4px;color:black;'>*24시기준</span></h4>",unsafe_allow_html=True)
            # st.markdown(f"<span style='font-size:4px;'>*24시기준</span>",unsafe_allow_html=True)

        kpi3,kpi4,kpi5=st.columns(3)
        with kpi3:
            ## 11월 무슨요일에 사고가 가장 많이 나는지
            ## 1 일 2 월 .. 7 토
            df=train.groupby(['월','요일'])['사고유형대분류'].count()
            top_acc_time=df.loc[month].idxmax()
            days={1:'일',2:'월',3:'화',4:'수',5:'목',6:'금',7:'토'}
            st.markdown("<h3 style='text-align:center'>최다사고요일</h3>",unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center;color:red'>{days[top_acc_time]}요일</h4>",unsafe_allow_html=True)
        with kpi4:
            ## 11월 어떤 지역에서 사고가 가장 많이 나는지.
            df=train.groupby(['월','지역'])['사고유형대분류'].count()
            top_acc=df.loc[month].idxmax()
            count=df.loc[month][top_acc]
            st.markdown("<h3 style='text-align:center'>최다사고지역</h3>",unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center;color:red;'>{top_acc} <span style='color:rgb(49, 51, 63);'>{count}회</span></h4>",unsafe_allow_html=True)
        with kpi5:
            ## 11월 초,중,말에서 사고가 언제 가장 많이 나는지.
            df=train.groupby(['월','일_bin'])['사고유형대분류'].count()
            top_acc=df.loc[month].idxmax()
            count=df.loc[month][top_acc]
            days_bin={0:'초',1:'중',2:'말'}
            st.markdown("<h3 style='text-align:center'>최다사고날짜</h3>",unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center;color:rgb(49, 51, 63);'>{month}월 <span style='color:red;'>{days_bin[top_acc]}</span></h4>",unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:4px;'>*1일~10일은 초<br>11일~20일은 중<br>21일~30일은 말입니다.</span>",unsafe_allow_html=True)

        st.markdown(f"<h3>사고다발구역과 법규위반구역</h3>",unsafe_allow_html=True)
        st.map(mapdata)

        st.write(" \n ")
        st.write(" \n ")
        st.write(" \n ")
        st.write(" \n ")
        st.write(" \n ")
        st.write(" \n ")

        st.markdown(f"<center style='font-size:10px;font-weight:bold;'>사용 데이터</center>",unsafe_allow_html=True)
        st.markdown(f"<center style='font-size:10px;'>도로교통공단_스쿨존어린이사고다발지역정보서비스</center>",unsafe_allow_html=True)
        st.markdown(f"<center style='font-size:10px;'>도로교통공단_사망교통사고정보서비스</center>",unsafe_allow_html=True)
        st.markdown(f"<center style='font-size:10px;'>전국어린이보호구역표준데이터</center>",unsafe_allow_html=True)
    else:
        dong=st.selectbox('지역 선택',[i for i in le.classes_])
        url=f'https://www.google.com/search?q={dong}+날씨'
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'}
        res = requests.get(url,headers=headers)
        res.raise_for_status()
        soup=BeautifulSoup(res.text,'lxml')

        temp=int(soup.find('span',attrs={'id':'wob_tm'}).text)
        rain=int(soup.find('span',attrs={'id':'wob_pp'}).text[:-1])
        hum=int(soup.find('span',attrs={'id':'wob_hm'}).text[:-1])
        wind=soup.find('span',attrs={'id':'wob_ws'}).text
        p=re.compile('[0-9]+')
        wind=int(p.match(wind).group())

        url=f'https://www.google.com/search?q=광진구+해지는시간'
        res = requests.get(url,headers=headers)
        res.raise_for_status()
        soup=BeautifulSoup(res.text,'lxml')
        time=int(soup.find('div',attrs={'class':'MUxGbd t51gnb lyLwlc lEBKkf'}).text.split(' ')[1].split(':')[0])
        if 4<time:
            juya=1
        else:
            juya=2
        conv_dong=le.transform([dong]).item()

        pred_data=pd.DataFrame({'월':[month],'일':[day],'시간':[hour],'기온(°C)':[temp],'풍속(m/s)':[wind],'강수량(mm)':[rain],'습도(%)':[hum],'주야구분':[juya],'요일':[weekday],'지역':[conv_dong]})
        # st.dataframe(pred_data)
        pred=xgb_wrapper.predict_proba(pred_data)
        kpi1,kpi2,kpi3=st.columns(3)
        print(round(pred[0][0],1))
        with kpi1:
            st.markdown(f"<h2 style='color:red;text-align:center'>차대차</h2>",unsafe_allow_html=True)
            st.markdown(f"<h3>사고확률 : {round(pred[0][0]*100,1)}%</h3>",unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"<h2 style='color:red;text-align:center'>차대사람</h2>",unsafe_allow_html=True)
            st.markdown(f"<h3>사고확률 : {round(pred[0][1]*100,1)}%</h3>",unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"<h2 style='color:red;text-align:center'>차량단독</h2>",unsafe_allow_html=True)
            st.markdown(f"<h3>사고확률 : {round(pred[0][2]*100,1)}%</h3>",unsafe_allow_html=True)

        # model_conp = RandomForestClassifier(n_estimators=100)
        # model_conp.fit(X_train,y_train)

        pred=xgb_wrapper.predict(traindata)

        # st.markdown(f"<h3><center>confusion matrix accuracy score : <span style='color:red;'>{round(accuracy_score(target,pred)*100,1)}%</span></center></h3>",unsafe_allow_html=True)

        fig,ax=plt.subplots(1,1,figsize=(8,6),dpi=140)
        plot_confusion_matrix(xgb_wrapper,traindata,target,ax=ax)
        st.pyplot(fig)
        
        
if __name__=='__main__':
    main()