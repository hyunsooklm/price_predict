import pandas as pd
import zipfile
import os
from flask import send_file,url_for,redirect
import shutil

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import flask
from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename

import seaborn as sns
import matplotlib.pyplot as plt


MODEL_SAVE_FOLDER_PATH = './model/'
DF_Directory='./data_preprocessing/'
def loading_model():
    global baechu_model
    global muu_model
    global gochu_model
    global defa_model
    global jjokpa_model
    global manl_model
    baechu_model = load_model(MODEL_SAVE_FOLDER_PATH + 'baechu.h5')
    muu_model = load_model(MODEL_SAVE_FOLDER_PATH + 'muu.h5')
    defa_model = load_model(MODEL_SAVE_FOLDER_PATH + 'defa.h5')
    gochu_model = load_model(MODEL_SAVE_FOLDER_PATH + 'gochu.h5')
    jjokpa_model = load_model(MODEL_SAVE_FOLDER_PATH + 'jjokpa.h5')
    manl_model = load_model(MODEL_SAVE_FOLDER_PATH + 'manl.h5')
    print('loading Done')


def create_dataset(scaled_dataset):
    # Creating a data structure with 90 timestamps and 1 output
    Xlist = []
    Ylist = []

    n_past = 21

    for index, i in enumerate(range(0, len(scaled_dataset) - n_past)):
        Xlist.append(scaled_dataset[i: i + n_past, :])
        Ylist.append(scaled_dataset[i + n_past:i + n_past + 1, -1])

    Xlist, Ylist = np.array(Xlist), np.array(Ylist)
    return Xlist, Ylist

def make_datelist(df_copy):
    # DateList 얻기
    timestamps=list(df_copy.index)
    datelist=[]
    for date in timestamps:
        datelist.append(str(date)[:10])
    # print(datelist)
    return datelist

def make_set_Scaler(df):
    dataset_float = df.astype(float)
    df_numpy = dataset_float.values

    df_scaler = StandardScaler()
    df_numpy_scaled = df_scaler.fit_transform(df_numpy)  # DataFrame Scaler

    price_scaler = StandardScaler()
    price_scaler.fit_transform(df_numpy[:, [-1]])  # price_Scaler

    return df_numpy_scaled, price_scaler  # return (scaling_df, price_scaler)
def mdf(df_copy,xls,name):
    # print(name)
    # print(df_copy.info())
    # print(xls.info())
    xls.reset_index(inplace=True)
    xls.head()
    for n in reversed(range(len(xls))):
        date = xls.iloc[n]['day']
        for i in reversed(range(len(df_copy))):
            d2 = df_copy.iloc[i]['day']
            if d2 == date:
                print(date)
                df_copy.iloc[i] = xls.iloc[n]
                break
    # df_copy.to_excel(f'Test/{name}')

def make_df(xls_list):      #df만들기
    global baechu_df,muu_df,manl_df,gochu_df,defa_df,jjokpa_df
    global baechu_df_copy,muu_df_copy,manl_df_copy,gochu_df_copy,defa_df_copy,jjokpa_df_copy
    baechu_df = pd.read_excel(DF_Directory+'baechu.xlsx')
    muu_df = pd.read_excel(DF_Directory+'muu.xlsx')
    manl_df = pd.read_excel(DF_Directory+'manl.xlsx')
    gochu_df = pd.read_excel(DF_Directory+'gochu.xlsx')
    defa_df = pd.read_excel(DF_Directory+'defa.xlsx')
    jjokpa_df = pd.read_excel(DF_Directory+'jjokpa.xlsx')

    baechu_df_copy=baechu_df[:]
    muu_df_copy=muu_df[:]
    manl_df_copy=manl_df[:]
    gochu_df_copy=gochu_df[:]
    defa_df_copy=defa_df[:]
    jjokpa_df_copy=jjokpa_df[:]

    df_copy_list=[baechu_df_copy,muu_df_copy,manl_df_copy,gochu_df_copy,defa_df_copy,jjokpa_df_copy]
    a=['baechu.xlsx','muu.xlsx','manl.xlsx','gochu.xlsx','defa.xlsx','jjokpa.xlsx']
    for df_copy,xls,name in zip(df_copy_list,xls_list,a):
        mdf(df_copy,xls,name)
    #############################################copy와 list들 조합해야겠구만!
    #day => index로 변환하기(원본따로)
    baechu_df_copy.set_index('day',inplace=True)
    muu_df_copy.set_index('day',inplace=True)
    manl_df_copy.set_index('day',inplace=True)
    gochu_df_copy.set_index('day',inplace=True)
    defa_df_copy .set_index('day',inplace=True)
    jjokpa_df_copy.set_index('day',inplace=True)

    # DateList 얻기
    global baechu_datelist,muu_datelist,manl_datelist,gochu_datelist,defa_datelist,jjokpa_datelist
    baechu_datelist=make_datelist(baechu_df_copy)
    muu_datelist=make_datelist(muu_df_copy)
    manl_datelist=make_datelist(manl_df_copy)
    gochu_datelist=make_datelist(gochu_df_copy)
    defa_datelist=make_datelist(defa_df_copy)
    jjokpa_datelist=make_datelist(jjokpa_df_copy)

    global baechu_scaled,baechu_price_Sc,muu_scaled, muu_price_Sc,manl_scaled, manl_price_Sc,gochu_scaled, gochu_price_Sc,defa_scaled, defa_price_Sc,jjokpa_scaled, jjokpa_price_Sc
    baechu_scaled, baechu_price_Sc = make_set_Scaler(baechu_df_copy)
    muu_scaled, muu_price_Sc = make_set_Scaler(muu_df_copy)
    manl_scaled, manl_price_Sc = make_set_Scaler(manl_df_copy)
    gochu_scaled, gochu_price_Sc = make_set_Scaler(gochu_df_copy)
    defa_scaled, defa_price_Sc = make_set_Scaler(defa_df_copy)
    jjokpa_scaled, jjokpa_price_Sc = make_set_Scaler(jjokpa_df_copy)

    global baechu_Xlist, baechu_Ylist,muu_Xlist, muu_Ylist,manl_Xlist, manl_Ylist,gochu_Xlist, gochu_Ylist,defa_Xlist,defa_Ylist,jjokpa_Xlist,jjokpa_Ylist
    baechu_Xlist, baechu_Ylist = create_dataset(baechu_scaled)
    muu_Xlist, muu_Ylist = create_dataset(muu_scaled)
    manl_Xlist, manl_Ylist = create_dataset(manl_scaled)
    gochu_Xlist, gochu_Ylist = create_dataset(gochu_scaled)
    defa_Xlist, defa_Ylist = create_dataset(defa_scaled)
    jjokpa_Xlist, jjokpa_Ylist = create_dataset(jjokpa_scaled)
    print('make_df Done')
def make_X_Predict_list(day,Xlist,Ylist,datelist,predict_day):
    X_Predict_list=None
    n=-1
    for index,date in enumerate(datelist[::-1],start=1):
        if day==date:
            n=index
            break
    if n==-1:
        print('존재하지 않는 날짜')
    else:
        print(f'n번쨰에 존재:{n}')
        index=len(Xlist)-n+1    #하나 앞에서부터 predict_day만큼 가져가기
        if index+predict_day>len(Xlist):
            print('데이터 부족으로 인한 예측불가')
        else:
            length_datelist=len(datelist)-n+1
            datelist_copy=datelist[length_datelist:length_datelist+predict_day]
            # print(datelist)

            X_Predict_list=Xlist[index:index+predict_day]
            Y_Real_price_list=Ylist[index:index+predict_day]
            # print(f'index:index+predict_day:{index},{index+predict_day},len(Xlist):{len(Xlist)}')
    return X_Predict_list,Y_Real_price_list,datelist_copy
    #1.day가 각자의 datelist[::-1]에서 몇번째 있는지 찾는다. =>n추출
    #2.Xlist[::-1]에서

def make_Predict_parameter(last_Day):
    # Last_day + 28일째 예측
    n_future = 28
    global baechu_x_predict_list, baechu_y_predict_list,muu_x_predict_list, muu_y_predict_list,manl_x_predict_list, manl_y_predict_list
    global gochu_x_predict_list, gochu_y_predict_list,defa_x_predict_list, defa_y_predict_list,jjokpa_x_predict_list, jjokpa_y_predict_list
    global predict_date
    try:
        baechu_x_predict_list, baechu_y_predict_list, predict_date = make_X_Predict_list(last_Day, baechu_Xlist,
                                                                                         baechu_Ylist, baechu_datelist,                                                                                        predict_day=28)
        muu_x_predict_list, muu_y_predict_list, predict_date = make_X_Predict_list(last_Day, muu_Xlist, muu_Ylist,
                                                                                   muu_datelist, predict_day=28)
        manl_x_predict_list, manl_y_predict_list, predict_date = make_X_Predict_list(last_Day, manl_Xlist, manl_Ylist,
                                                                                     manl_datelist, predict_day=28)
        gochu_x_predict_list, gochu_y_predict_list, predict_date = make_X_Predict_list(last_Day, gochu_Xlist,
                                                                                       gochu_Ylist, gochu_datelist,
                                                                                       predict_day=28)
        defa_x_predict_list, defa_y_predict_list, predict_date = make_X_Predict_list(last_Day, defa_Xlist, defa_Ylist,
                                                                                     defa_datelist, predict_day=28)
        jjokpa_x_predict_list, jjokpa_y_predict_list, predict_date = make_X_Predict_list(last_Day, jjokpa_Xlist,
                                                                                         jjokpa_Ylist, jjokpa_datelist,
                                                                                         predict_day=28)
        # print(predict_date)
        # print(len(predict_date))
    except:
        print('데이터부족, 예측불가')
    print('make_Predict_parameter Done')
def make_last_Day(sample_df):
    # sample_df = pd.read_excel(DF_Directory + 'sample.xlsx')
    sample_df['day'] = pd.to_datetime(sample_df['day'], format='%Y%m%d')
    sample_df.set_index('day', inplace=True)
    sample_datelist = make_datelist(sample_df)
    last_Day = sample_datelist[-1]  # Last_day
    print('make_last_Day Done')
    return last_Day

def predict():
    global Predict_baechu_price,Predict_muu_price,Predict_defa_price,Predict_gochu_price,Predict_jjokpa_price,Predict_manl_price
    global Real_baechu_price,Real_muu_price,Real_defa_price,Real_gochu_price,Real_jjokpa_price,Real_manl_price
    Predict_baechu_price = baechu_price_Sc.inverse_transform(baechu_model.predict(baechu_x_predict_list)).flatten()
    Predict_muu_price = muu_price_Sc.inverse_transform(muu_model.predict(muu_x_predict_list)).flatten()
    Predict_defa_price = defa_price_Sc.inverse_transform(defa_model.predict(defa_x_predict_list)).flatten()
    Predict_gochu_price = gochu_price_Sc.inverse_transform(gochu_model.predict(gochu_x_predict_list)).flatten()
    Predict_jjokpa_price = jjokpa_price_Sc.inverse_transform(jjokpa_model.predict(jjokpa_x_predict_list)).flatten()
    Predict_manl_price = manl_price_Sc.inverse_transform(manl_model.predict(manl_x_predict_list)).flatten()

    Real_baechu_price = baechu_price_Sc.inverse_transform(baechu_y_predict_list).flatten()
    Real_muu_price = muu_price_Sc.inverse_transform(muu_y_predict_list).flatten()
    Real_defa_price = defa_price_Sc.inverse_transform(defa_y_predict_list).flatten()
    Real_gochu_price = gochu_price_Sc.inverse_transform(gochu_y_predict_list).flatten()
    Real_jjokpa_price = jjokpa_price_Sc.inverse_transform(jjokpa_y_predict_list).flatten()
    Real_manl_price = manl_price_Sc.inverse_transform(manl_y_predict_list).flatten()

    print('predict Done')
    # print('Real')
    # print(Real_baechu_price)
    #
    # print('Predict')
    # print(Predict_baechu_price)

def make_predict_df():
    global PREDICT_baechu, original_baechu
    global PREDICT_muu, original_muu
    global PREDICT_gochu, original_gochu
    global PREDICT_manl, original_manl
    global PREDICT_defa, original_defa
    global PREDICT_jjokpa, original_jjokpa

    PREDICT_baechu = pd.DataFrame({'Date': pd.Series(predict_date), 'price': Predict_baechu_price})
    PREDICT_baechu['Date'] = pd.to_datetime(PREDICT_baechu['Date'])
    original_baechu = baechu_df[['day', 'some']]
    original_baechu['day'] = pd.to_datetime(original_baechu['day'])
    original_baechu = original_baechu.loc[original_baechu['day'] >= predict_date[0]]
    original_baechu = original_baechu.loc[original_baechu['day'] <= predict_date[-1]]
    # print(PREDICT_baechu)
    # print(original_baechu)

    # fig, axs = plt.subplots(figsize=(10, 5), constrained_layout=True)
    # axs.set_title('baechu')
    # sns.lineplot(original_baechu['day'], original_baechu['some'], color='b', label='Real_Price')
    # sns.lineplot(PREDICT_baechu['Date'], PREDICT_baechu['price'], color='r', label='Predict')

    print('-' * 100)

    PREDICT_muu = pd.DataFrame({'Date': pd.Series(predict_date), 'price': Predict_muu_price})
    PREDICT_muu['Date'] = pd.to_datetime(PREDICT_muu['Date'])
    original_muu = muu_df[['day', 'some']]
    original_muu['day'] = pd.to_datetime(original_muu['day'])
    original_muu = original_muu.loc[original_muu['day'] >= predict_date[0]]
    original_muu = original_muu.loc[original_muu['day'] <= predict_date[-1]]

    # fig, axs = plt.subplots(figsize=(10, 5), constrained_layout=True)
    # axs.set_title('muu')
    # sns.lineplot(original_muu['day'], original_muu['some'], color='b', label='Real_Price')
    # sns.lineplot(PREDICT_muu['Date'], PREDICT_muu['price'], color='r', label='Predict')
    # print('-' * 100)

    PREDICT_manl = pd.DataFrame({'Date': pd.Series(predict_date), 'price': Predict_manl_price})
    PREDICT_manl['Date'] = pd.to_datetime(PREDICT_manl['Date'])
    original_manl = manl_df[['day', 'some']]
    original_manl['day'] = pd.to_datetime(original_manl['day'])
    original_manl = original_manl.loc[original_manl['day'] >= predict_date[0]]
    original_manl = original_manl.loc[original_manl['day'] <= predict_date[-1]]

    # fig, axs = plt.subplots(figsize=(10, 5), constrained_layout=True)
    # axs.set_title('manl')
    # sns.lineplot(original_manl['day'], original_manl['some'], color='b', label='Real_Price')
    # sns.lineplot(PREDICT_manl['Date'], PREDICT_manl['price'], color='r', label='Predict')
    # print('-' * 100)

    PREDICT_gochu = pd.DataFrame({'Date': pd.Series(predict_date), 'price': Predict_gochu_price})
    PREDICT_gochu['Date'] = pd.to_datetime(PREDICT_gochu['Date'])
    original_gochu = gochu_df[['day', 'some']]
    original_gochu['day'] = pd.to_datetime(original_gochu['day'])
    original_gochu = original_gochu.loc[original_gochu['day'] >= predict_date[0]]
    original_gochu = original_gochu.loc[original_gochu['day'] <= predict_date[-1]]

    # fig, axs = plt.subplots(figsize=(10, 5), constrained_layout=True)
    # axs.set_title('gochu')
    # sns.lineplot(original_gochu['day'], original_gochu['some'], color='b', label='Real_Price')
    # sns.lineplot(PREDICT_gochu['Date'], PREDICT_gochu['price'], color='r', label='Predict')
    # print('-' * 100)

    PREDICT_defa = pd.DataFrame({'Date': pd.Series(predict_date), 'price': Predict_defa_price})
    PREDICT_defa['Date'] = pd.to_datetime(PREDICT_defa['Date'])
    original_defa = defa_df[['day', 'some']]
    original_defa['day'] = pd.to_datetime(original_defa['day'])
    original_defa = original_defa.loc[original_defa['day'] >= predict_date[0]]
    original_defa = original_defa.loc[original_defa['day'] <= predict_date[-1]]

    # fig, axs = plt.subplots(figsize=(10, 5), constrained_layout=True)
    # axs.set_title('defa')
    # sns.lineplot(original_defa['day'], original_defa['some'], color='b', label='Real_Price')
    # sns.lineplot(PREDICT_defa['Date'], PREDICT_defa['price'], color='r', label='Predict')
    # print('-' * 100)

    PREDICT_jjokpa = pd.DataFrame({'Date': pd.Series(predict_date), 'price': Predict_jjokpa_price})
    PREDICT_jjokpa['Date'] = pd.to_datetime(PREDICT_jjokpa['Date'])
    original_jjokpa = jjokpa_df[['day', 'some']]
    original_jjokpa['day'] = pd.to_datetime(original_jjokpa['day'])
    original_jjokpa = original_jjokpa.loc[original_jjokpa['day'] >= predict_date[0]]
    original_jjokpa = original_jjokpa.loc[original_jjokpa['day'] <= predict_date[-1]]

    # fig, axs = plt.subplots(figsize=(10, 5), constrained_layout=True)
    # axs.set_title('jjokpa')
    # sns.lineplot(original_jjokpa['day'], original_jjokpa['some'], color='b', label='Real_Price')
    # sns.lineplot(PREDICT_jjokpa['Date'], PREDICT_jjokpa['price'], color='r', label='Predict')
    # print('-' * 100)
    print('make_predict_df Done')
def make_nonPredict_price():
    global gat_df,gool_df,minari_df,myulchi_df,senggang_df,seu_df,sogm_df
    sub_price_path = DF_Directory+'sub_price/'
    gat_df = pd.read_excel(sub_price_path + 'gat.xlsx')
    gool_df = pd.read_excel(sub_price_path + 'gool.xlsx')
    minari_df = pd.read_excel(sub_price_path + 'minari.xlsx')
    myulchi_df = pd.read_excel(sub_price_path + 'myulchi.xlsx')
    senggang_df = pd.read_excel(sub_price_path + 'senggang.xlsx')
    seu_df = pd.read_excel(sub_price_path + 'seu.xlsx')
    sogm_df = pd.read_excel(sub_price_path + 'sogm.xlsx')

    # gat_df['gimjang']=gat_df['gimjang'].astype('float32')

    gat_df.rename(columns={'day': 'Date'}, inplace=True)
    gool_df.rename(columns={'day': 'Date'}, inplace=True)
    minari_df.rename(columns={'day': 'Date'}, inplace=True)
    myulchi_df.rename(columns={'day': 'Date'}, inplace=True)
    senggang_df.rename(columns={'day': 'Date'}, inplace=True)
    seu_df.rename(columns={'day': 'Date'}, inplace=True)
    sogm_df.rename(columns={'day': 'Date'}, inplace=True)

    print(gat_df.head())
    gat_df = gat_df.loc[gat_df['Date'] >= predict_date[0]]
    gat_df = gat_df.loc[gat_df['Date'] <= predict_date[-1]]
    gat_price = gat_df['gimjang'].tolist()
    # print(gat_price)
    # print(f'gat')
    # print(gat_df)
    myulchi_df = myulchi_df.loc[myulchi_df['Date'] >= predict_date[0]]
    myulchi_df = myulchi_df.loc[myulchi_df['Date'] <= predict_date[-1]]
    myulchi_price = myulchi_df['gimjang'].tolist()
    # print(f'myulchi')
    # print(myulchi_df)

    gool_df = gool_df.loc[gool_df['Date'] >= predict_date[0]]
    gool_df = gool_df.loc[gool_df['Date'] <= predict_date[-1]]
    gool_price = gool_df['gimjang'].tolist()
    # print(f'gool')
    # print(gool_df)

    minari_df = minari_df.loc[minari_df['Date'] >= predict_date[0]]
    minari_df = minari_df.loc[minari_df['Date'] <= predict_date[-1]]
    minari_price = minari_df['gimjang'].tolist()
    # print(f'minari')
    # print(minari_df)

    senggang_df = senggang_df.loc[senggang_df['Date'] >= predict_date[0]]
    senggang_df = senggang_df.loc[senggang_df['Date'] <= predict_date[-1]]
    senggang_price = senggang_df['gimjang'].tolist()
    # print(f'senggang')
    # print(senggang_df)

    seu_df = seu_df.loc[seu_df['Date'] >= predict_date[0]]
    seu_df = seu_df.loc[seu_df['Date'] <= predict_date[-1]]
    seu_price = seu_df['gimjang'].tolist()
    # print(f'seu')
    # print(seu_df)

    sogm_df = sogm_df.loc[sogm_df['Date'] >= predict_date[0]]
    sogm_df = sogm_df.loc[sogm_df['Date'] <= predict_date[-1]]
    sogm_price = sogm_df['gimjang'].tolist()
    # print(f'sogm')
    print(sogm_df)

    # print(gat_df.info())
    print('make_nonPredict_price Done')
def Predict_Gimjang():
    # ----------------------original---------------------
    original_baechu['gimjang'] = original_baechu['some'] * 20  # 배추 x 20포기
    original_muu['gimjang'] = original_muu['some'] * 10  # 무 x 10개
    original_gochu['gimjang'] = original_gochu['some'] * 3.1  # 고춧가루 x 3.1
    original_manl['gimjang'] = original_manl['some'] * 1.2  # 마늘 x 1.2
    original_defa['gimjang'] = original_defa['some'] * 2  # 대파 x 2
    original_jjokpa['gimjang'] = original_jjokpa['some'] * 2.4  # 쪽파 x 2.4

    # ----------------------Predict---------------------
    PREDICT_baechu['gimjang'] = PREDICT_baechu['price'] * 20  # 배추 x 20포기
    PREDICT_muu['gimjang'] = PREDICT_muu['price'] * 10  # 무 x 10개
    PREDICT_gochu['gimjang'] = PREDICT_gochu['price'] * 3.1  # 고춧가루 x 3.1
    PREDICT_manl['gimjang'] = PREDICT_manl['price'] * 1.2  # 마늘 x 1.2
    PREDICT_defa['gimjang'] = PREDICT_defa['price'] * 2  # 대파 x 2
    PREDICT_jjokpa['gimjang'] = PREDICT_jjokpa['price'] * 2.4  # 쪽파 x 2.4

    # print(original_manl)
    # print(PREDICT_baechu['gimjang'][0])
    predict_price = []
    original_price = []
    for i in range(len(predict_date)):
        original_sum_price = 0
        predict_sum_price = 0

        original_sum_price += original_baechu['gimjang'].tolist()[i]
        original_sum_price += original_muu['gimjang'].tolist()[i]
        original_sum_price += original_gochu['gimjang'].tolist()[i]
        original_sum_price += original_manl['gimjang'].tolist()[i]
        original_sum_price += original_defa['gimjang'].tolist()[i]
        original_sum_price += original_jjokpa['gimjang'].tolist()[i]

        predict_sum_price += PREDICT_baechu['gimjang'][i]
        predict_sum_price += PREDICT_muu['gimjang'][i]
        predict_sum_price += PREDICT_gochu['gimjang'][i]
        predict_sum_price += PREDICT_manl['gimjang'][i]
        predict_sum_price += PREDICT_defa['gimjang'][i]
        predict_sum_price += PREDICT_jjokpa['gimjang'][i]
        # --------------------------------------------------original/predict price에 각각 실제 농산물가격 / 예측 농산물가격 넣기
        original_sum_price += gat_df['gimjang'].tolist()[i]
        original_sum_price += gool_df['gimjang'].tolist()[i]
        original_sum_price += minari_df['gimjang'].tolist()[i]
        original_sum_price += myulchi_df['gimjang'].tolist()[i]
        original_sum_price += senggang_df['gimjang'].tolist()[i]
        original_sum_price += seu_df['gimjang'].tolist()[i]
        original_sum_price += sogm_df['gimjang'].tolist()[i]
        # ----------------------------------------------------original/predict price에 각각 비예측농산물 실제가격 넣기
        predict_sum_price += gat_df['gimjang'].tolist()[i]
        predict_sum_price += gool_df['gimjang'].tolist()[i]
        predict_sum_price += minari_df['gimjang'].tolist()[i]
        predict_sum_price += myulchi_df['gimjang'].tolist()[i]
        predict_sum_price += senggang_df['gimjang'].tolist()[i]
        predict_sum_price += seu_df['gimjang'].tolist()[i]
        predict_sum_price += sogm_df['gimjang'].tolist()[i]
        original_price.append(original_sum_price)
        predict_price.append(predict_sum_price)
    # print(original_price)
    # print(predict_price)

    global PREDICT_GIMJANG
    PREDICT_GIMJANG = pd.DataFrame(
        {'Date': PREDICT_baechu['Date'], 'Origin_Sum': original_price, 'Predict_Sum': predict_price})
    print(PREDICT_GIMJANG)
    print('predict gimjang Done')
    return PREDICT_GIMJANG

def Gimjang_Total():
    gat_price=gat_df['gimjang'].tolist()
    gool_price=gool_df['gimjang'].tolist()
    minari_price=minari_df['gimjang'].tolist()
    myulchi_price=myulchi_df['gimjang'].tolist()
    senggang_price=senggang_df['gimjang'].tolist()
    seu_price=seu_df['gimjang'].tolist()
    sogm_price=sogm_df['gimjang'].tolist()

    PREDICT_GIMJANG_TOTAL = pd.DataFrame({'Date':PREDICT_baechu['Date']})
    PREDICT_GIMJANG_TOTAL
    predict_baechu=PREDICT_baechu['gimjang'].tolist()
    predict_muu=PREDICT_muu['gimjang'].tolist()
    predict_gochu=PREDICT_gochu['gimjang'].tolist()
    predict_manl=PREDICT_manl['gimjang'].tolist()
    predict_defa=PREDICT_defa['gimjang'].tolist()
    predict_jjokpa=PREDICT_jjokpa['gimjang'].tolist()

    # senggang_price=senggang_price,seu_price=seu_price,sogm_price=sogm_price
    PREDICT_GIMJANG_TOTAL=PREDICT_GIMJANG_TOTAL.assign(baechu=predict_baechu,muu=predict_muu,gochu=predict_gochu,manl=predict_manl,defa=predict_defa,jjokpa=predict_jjokpa,gat_price=gat_price,gool_price=gool_price,minari_price=minari_price,myulchi_price=myulchi_price,senggang_price=senggang_price,seu_price=seu_price,sogm_price=sogm_price)
    PREDICT_GIMJANG_TOTAL=PREDICT_GIMJANG_TOTAL.assign(Total=PREDICT_GIMJANG['Predict_Sum'].tolist())
    PREDICT_GIMJANG_TOTAL

    original_GIMJANG_TOTAL = pd.DataFrame({'Date':original_baechu['day']})
    original_GIMJANG_TOTAL
    Ori_baechu=original_baechu['gimjang'].tolist()
    Ori_muu=original_muu['gimjang'].tolist()
    Ori_gochu=original_gochu['gimjang'].tolist()
    Ori_manl=original_manl['gimjang'].tolist()
    Ori_defa=original_defa['gimjang'].tolist()
    Ori_jjokpa=original_jjokpa['gimjang'].tolist()
    original_GIMJANG_TOTAL=original_GIMJANG_TOTAL.assign(baechu=Ori_baechu,muu=Ori_muu,gochu=Ori_gochu,manl=Ori_manl,defa=Ori_defa,jjokpa=Ori_jjokpa,gat_price=gat_price,gool_price=gool_price,minari_price=minari_price,myulchi_price=myulchi_price,senggang_price=senggang_price,seu_price=seu_price,sogm_price=sogm_price)
    original_GIMJANG_TOTAL=original_GIMJANG_TOTAL.assign(Total=PREDICT_GIMJANG['Origin_Sum'].tolist())

    TOTAL_SAVE_PATH='./summary/'
    if not os.path.isdir(TOTAL_SAVE_PATH):
        os.mkdir(TOTAL_SAVE_PATH)
    else:
        shutil.rmtree((TOTAL_SAVE_PATH))
        os.mkdir(TOTAL_SAVE_PATH)
    PREDICT_GIMJANG_TOTAL.to_excel(TOTAL_SAVE_PATH+'PPEDICT_GIMJANG_TOTAL.xlsx',index=False)
    original_GIMJANG_TOTAL.to_excel(TOTAL_SAVE_PATH + 'original_GIMJANG_TOTAL.xlsx',index=False)
    # return PREDICT_GIMJANG_TOTAL,original_GIMJANG_TOTAL


app = Flask(__name__)
# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def home():
    return flask.render_template('index.html')

@app.route('/result',methods=['POST','GET'])
def result():
    if request.method=='POST':
        baechu = request.files['baechu']
        muu=request.files['muu']
        manl=request.files['manl']
        gochu=request.files['gochu']
        defa=request.files['defa']
        jjokpa=request.files['jjokpa']

        print('files ok!\n')
        upload_path='uploads'
        if os.path.exists(upload_path):
            shutil.rmtree(upload_path)
        os.mkdir(upload_path)
        baechu.save(f'./uploads/{secure_filename(baechu.filename)}')
        muu.save(f'./uploads/{secure_filename(muu.filename)}')
        manl.save(f'./uploads/{secure_filename(manl.filename)}')
        gochu.save(f'./uploads/{secure_filename(gochu.filename)}')
        defa.save(f'./uploads/{secure_filename(defa.filename)}')
        jjokpa.save(f'./uploads/{secure_filename(jjokpa.filename)}')

        baechu_xls = pd.read_excel(f'uploads/{secure_filename(baechu.filename)}',index_col=None)
        muu_xls = pd.read_excel(f'uploads/{secure_filename(muu.filename)}',index_col=None)
        manl_xls = pd.read_excel(f'uploads/{secure_filename(manl.filename)}',index_col=None)
        gochu_xls = pd.read_excel(f'uploads/{secure_filename(gochu.filename)}',index_col=None)
        defa_xls = pd.read_excel(f'uploads/{secure_filename(defa.filename)}',index_col=None)
        jjokpa_xls = pd.read_excel(f'uploads/{secure_filename(jjokpa.filename)}',index_col=None)

        print(baechu_xls.info())
        xls_list=[baechu_xls,muu_xls,manl_xls,gochu_xls,defa_xls,jjokpa_xls]

        last_Day=make_last_Day(baechu_xls)
        print(f'last_Day:{last_Day}')
        del baechu
        del muu
        del gochu
        del manl
        del jjokpa
        del defa

        # print(baechu_xls.info())
        # print(baechu_xls.head())
        make_df(xls_list)
        make_Predict_parameter(last_Day)
        predict()
        make_predict_df()
        make_nonPredict_price()
        PREDICT_GIMJANG=Predict_Gimjang()
        Gimjang_Total()

        # PREDICT_GIMJANG
        labels = PREDICT_GIMJANG['Date'].astype('str')
        Origin_price = PREDICT_GIMJANG['Origin_Sum'].astype('int')
        Predict_price = PREDICT_GIMJANG['Predict_Sum'].astype('int')
        labels = labels.tolist()
        Origin_price=Origin_price.tolist()
        Predict_price=Predict_price.tolist()

        return flask.render_template('index.html',labels=labels,Origin_price=Origin_price,Predict_price=Predict_price)

@app.route('/download_all')
def download_all():
    try:
        os.remove('summary.zip')
    except OSError:
        pass
    if os.path.isdir('./summary'):
        zipf = zipfile.ZipFile('summary.zip','w', zipfile.ZIP_DEFLATED)
        for root,dirs, files in os.walk('./summary/'):
            for file in files:
                zipf.write('./summary/'+file)
                #1. summary 채워주고 2. 여기와서 summary에 있는거 zip만들고 리턴, 없는데 오면 return
        zipf.close()
        shutil.rmtree('./summary')
        return send_file('summary.zip',
                mimetype = 'zip',
                attachment_filename= 'summary.zip',
                as_attachment = True)
    else:
        return redirect(url_for('home'))

if __name__=='__main__':
    loading_model()
    app.run(debug=True)
