import numpy as np


time_sim_roadsign =[[38,83,99,139,111,38,83,99,139,111,38,83,99,139,111,38,83,99,139,111],
                    [250,304,250,306,250,319,250,307,250,306,250,303,250,306,275,304,275,312,264,308],
                    [197,308,200,309,196,323,196,311,196,310,197,307,196,309,199,308,199,316,196,312],
                    [394,394,402,394,394,395,395,405,395,394,400,395,394,394,460,465,462,464,464,462]]

time_real_roadsign =[[55.41,79.42,79.34,92.75,61.4,55.41,79.42,79.34,92.75,61.4,55.41,79.42,79.34,92.75,61.4,55.41,79.42,79.34,92.75,61.4],
                    [88.87,113.41,88.88,113.14,88.8,113.39,88.88,113.26,88.66,113.13,88.87,113.41,88.88,113.14,88.8,113.39,88.88,113.26,88.66,113.13],
                    [67.86,92.78,67.87,92.45,92.42,92.77,92.39,92.36,92.3,92.7,67.86,92.78,67.87,92.45,67.88,92.77,67.87,92.67,67.85,92.7],
                    [96.31,128.6,96.22,128.53,96.1,128.59,96.32,128.74,96.0,128.64,96.31,128.6,96.22,128.53,96.1,128.59,96.32,128.74,96.0,128.64]]

time_sim_sentiment =[[321,321,328,321,321,322,322,331,322,321,213,174,231,174,224,212,175,232,174,224],
                    [366,107,366,108,367,321,376,322,366,107,322,107,367,321,366,322,370,324,370,367],
                    [319,456,319,456,319,456,319,456,319,319,319,456,319,456,319,456,319,456,456,456],
                    [307,320,309,322,306,307,309,309,306,317,307,317,309,316,307,311,306,310,306,310]]

time_real_sentiment =[[182.57,190.57,182.57,190.57,182.57,190.57,182.57,190.57,182.57,190.57,182.57,190.57,182.57,190.57,182.57,190.57,182.57,190.57,182.57,190.57],
                    [203.06,194.25,203.06,194.25,203.06,194.25,203.06,194.25,203.06,194.25,192.51,203.06,212.08,218.46,259.86,192.51,203.06,212.08,218.46,259.86],
                    [210.82,178.63,210.82,178.63,210.82,178.63,210.82,178.63,210.82,178.63,210.82,178.63,210.82,178.63,210.82,178.63,210.82,178.63,210.82,178.63],
                    [194.36,194.55,194.36,194.55,194.36,194.55,194.36,194.55,194.36,194.36,194.36,194.55,194.36,194.55,194.36,194.55,194.36,194.55,194.36,194.36]]


#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
#https://www.adamsmith.haus/python/answers/how-to-normalize-an-array-in-numpy-in-python


for i in range(np.shape(time_sim_roadsign)[0]):
    for j in range(np.shape(time_sim_roadsign)[1]):
        if(i == 0):
            print(np.around(((time_sim_roadsign[i][j]-np.min(time_sim_roadsign[i]))/(np.max(time_sim_roadsign[i])-np.min(time_sim_roadsign[i]))),4),",C3-Match,Road sign inspection")#np.linalg.norm(time_sim_roadsign[i]
        elif(i == 1):
            print(np.around(((time_sim_roadsign[i][j]-np.min(time_sim_roadsign[i]))/(np.max(time_sim_roadsign[i])-np.min(time_sim_roadsign[i]))),4),",NAN,Road sign inspection")
        elif(i == 2):
            print(np.around(((time_sim_roadsign[i][j]-np.min(time_sim_roadsign[i]))/(np.max(time_sim_roadsign[i])-np.min(time_sim_roadsign[i]))),4),",SEA-LEAP,Road sign inspection")
        elif(i == 3):
            print(np.around(((time_sim_roadsign[i][j]-np.min(time_sim_roadsign[i]))/(np.max(time_sim_roadsign[i])-np.min(time_sim_roadsign[i]))),4),",KCSS,Road sign inspection")
for i in range(np.shape(time_sim_sentiment)[0]):
    for j in range(np.shape(time_sim_sentiment)[1]):
        if(i == 0):
            #print(np.linalg.norm(time_sim_sentiment[i]))
            print(np.around(((time_sim_sentiment[i][j]-np.min(time_sim_sentiment[i]))/(np.max(time_sim_sentiment[i])-np.min(time_sim_sentiment[i]))),4),",C3-Match,Sentiment analysis")
        elif(i == 1):
            #print(np.linalg.norm(time_sim_sentiment[i]))
            print(np.around(((time_sim_sentiment[i][j]-np.min(time_sim_sentiment[i]))/(np.max(time_sim_sentiment[i])-np.min(time_sim_sentiment[i]))),4),",NAN,Sentiment analysis")
        elif(i == 2):
            #print(np.linalg.norm(time_sim_sentiment[i]))
            print(np.around(((time_sim_sentiment[i][j]-np.min(time_sim_sentiment[i]))/(np.max(time_sim_sentiment[i])-np.min(time_sim_sentiment[i]))),4),",SEA-LEAP,Sentiment analysis")
        elif(i == 3):
            #print(np.linalg.norm(time_sim_sentiment[i]))
            print(np.around(((time_sim_sentiment[i][j]-np.min(time_sim_sentiment[i]))/(np.max(time_sim_sentiment[i])-np.min(time_sim_sentiment[i]))),4),",KCSS,Sentiment analysis")
''''''