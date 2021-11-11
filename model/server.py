import streamlit as st
from datetime import datetime 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

data=pd.read_csv('accalldata.csv')
school=pd.read_csv('schoolzone.csv',encoding='cp949')

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
## 1 : 차대사람 2 : 차대차 3 : 차량단독
## 1 : 주간 2 : 야간


def main():
    today=datetime.today().strftime('%Y-%m-%d')
    year=today.split('-')[0]
    month=today.split('-')[1]
    day=today.split('-')[2] 
    st.title("어린이 보호구역 교통사고 주의 대쉬보드")
    st.write(f"""
    {year}년 {month}월 {day}일
    """)

    kpi1,kpi2=st.beta_columns(2)

    with kpi1:
        st.markdown("<h2 style='text-align:center'>최다사고유형</h2>",unsafe_allow_html=True)
        ## 11월 어떤 사고가 많이 일어나는지.
        df=train.groupby(['월','사고유형대분류'])['사고유형대분류'].count()
        top_acc=df.loc[11].idxmax()
        count=df.loc[11][top_acc]
        kind={1:'차대사람',2:'차대차',3:'차량단독'}
        st.markdown(f"<h2 style='text-align:center;color:red'>{kind[top_acc]}</h2>",unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>{count}회</h3>",unsafe_allow_html=True)
        # else:
        #     st.markdown(f"<h2 style='text-align:center;color:green'>▼ {newcase}</h2>",unsafe_allow_html=True)
    # with kpi2:
    #     st.markdown("<h2 style='text-align:center'>누적 확진자 수</h2>",unsafe_allow_html=True)
    #     totalCase=int(result[1].replace(',',''))
    #     st.markdown(f"<h2 style='text-align:center;'>{totalCase}</h2>",unsafe_allow_html=True)
    # with kpi3:
    #     st.markdown("<h2 style='text-align:center'>완치자 수</h2>",unsafe_allow_html=True)
    #     recovered=int(result[2].replace(',',''))
    #     if recovered>0:
    #         st.markdown(f"<h2 style='text-align:center;color:red'>{recovered}</h2>",unsafe_allow_html=True)
    #     else:
    #         st.markdown(f"<h2 style='text-align:center;color:green'>{recovered}</h2>",unsafe_allow_html=True)

    # kpi4,kpi5,kpi6,kpi7=st.beta_columns(4)

    # with kpi4:
    #     st.markdown("<h2 style='text-align:center'>사망자 수</h2>",unsafe_allow_html=True)
    #     death=int(result[3].replace(',',''))

    #     if region=='korea':
    #         if newcase>0:
    #             st.markdown(f"<h2 style='text-align:center;color:red'>▲ {death}</h2>",unsafe_allow_html=True)
    #         else:
    #             st.markdown(f"<h2 style='text-align:center;color:green'>▼ {death}</h2>",unsafe_allow_html=True)
    #     else:        
    #         if newcase>0:
    #             st.markdown(f"<h2 style='text-align:center;color:red'>{death}</h2>",unsafe_allow_html=True)
    #         else:
    #             st.markdown(f"<h2 style='text-align:center;color:green'>{death}</h2>",unsafe_allow_html=True)
    # with kpi5:
    #     st.markdown("<h2 style='text-align:center'>발생률</h2>",unsafe_allow_html=True)
    #     percentage=result[4]
    #     st.markdown(f"<h2 style='text-align:center;color:red;'>{percentage}%</h2>",unsafe_allow_html=True)
    # with kpi6:
    #     st.markdown("<h2 style='text-align:center'>지역발생 수</h2>",unsafe_allow_html=True)
    #     newCcase=int(result[5].replace(',',''))
    #     st.markdown(f"<h2 style='text-align:center;color:red'>▲ {newCcase}</h2>",unsafe_allow_html=True)
    # with kpi7:
    #     st.markdown("<h2 style='text-align:center'>해외유입 수</h2>",unsafe_allow_html=True)
    #     newFcase=int(result[6].replace(',',''))
    #     st.markdown(f"<h2 style='text-align:center;color:red'>▲ {newFcase}</h2>",unsafe_allow_html=True)
if __name__=='__main__':
    main()

