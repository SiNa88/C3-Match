import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import csv
#x_array = np.array(housing['total_bedrooms'])
#normalized_arr = preprocessing.normalize([x_array])
#print(normalized_arr)
#tips = pd.read_csv('seaborn-data-master/tips.csv')
colors_list = [ '#BF3028',  '#AFF820',  '#5FF0F0',  '#FF8030', '#BF3028',  '#AFF820', '#5FF0F0', '#FF8030','#FFD030', '#EFC068', '#7FC850',  '#98D8D8', '#F85888']

sim_data_to_plot = pd.read_csv("D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\UnChangedOne\\pplot-simulation-5-5-22.csv")
#df1 = pd.DataFrame(sim_data_to_plot, columns=["Completion time [s]", "method"])

'''max_c3match_rsi =0
max_c3match_sa =0    
max_nan_rsi =0
max_nan_sa =0
max_sea_rsi =0
max_sea_sa =0
max_kcss_rsi =0
max_kcss_sa =0
with open('D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\UnChangedOne\\pplot-simulation-5-5-22.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
        #print(f'Column names are {", ".join(row)}')
        if((row["method"]=="C3-Match") and row["Workflow"]=="Road sign inspection"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(int(row["Completion time [s]"])>= max_c3match_rsi):
                max_c3match_rsi=int(row["Completion time [s]"])
        elif((row["method"]=="NAN") and row["Workflow"]=="Road sign inspection"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(int(row["Completion time [s]"])>= max_nan_rsi):
                max_nan_rsi=int(row["Completion time [s]"])
        elif((row["method"]=="SEA-LEAP") and row["Workflow"]=="Road sign inspection"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(int(row["Completion time [s]"])>= max_sea_rsi):
                max_sea_rsi=int(row["Completion time [s]"])
        elif((row["method"]=="KCSS") and row["Workflow"]=="Road sign inspection"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(int(row["Completion time [s]"])>= max_kcss_rsi):
                max_kcss_rsi=int(row["Completion time [s]"])
        
        if((row["method"]=="C3-Match") and row["Workflow"]=="Sentiment analysis"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(int(row["Completion time [s]"])>= max_c3match_sa):
                max_c3match_sa=int(row["Completion time [s]"])
        elif((row["method"]=="NAN") and row["Workflow"]=="Sentiment analysis"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(int(row["Completion time [s]"])>= max_nan_sa):
                max_nan_sa=int(row["Completion time [s]"])
        elif((row["method"]=="SEA-LEAP") and row["Workflow"]=="Sentiment analysis"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(int(row["Completion time [s]"])>= max_sea_sa):
                max_sea_sa=int(row["Completion time [s]"])
        elif((row["method"]=="KCSS") and row["Workflow"]=="Sentiment analysis"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(int(row["Completion time [s]"])>= max_kcss_sa):
                max_kcss_sa=int(row["Completion time [s]"])

#print((df1["method"]=="C3-Match"))
print(max_c3match_rsi," " ,max_nan_rsi," " ,max_sea_rsi," " ,max_kcss_rsi)
print(max_c3match_sa," " ,max_nan_sa," " ,max_sea_sa," " ,max_kcss_sa)
#print((sim_data_to_plot["method"].str.contains("C3-Match")))
if((sim_data_to_plot["method"].str.contains("C3-Match"))[1] and (sim_data_to_plot["Workflow"].str.contains("Road sign inspection")[1])):
   sim_data_to_plot["Completion time [s]"]/=max_c3match_rsi
elif((sim_data_to_plot["method"].str.contains("NAN"))[1] and (sim_data_to_plot["Workflow"].str.contains("Road sign inspection")[1])):
    sim_data_to_plot["Completion time [s]"]/=max_nan_rsi
elif((sim_data_to_plot["method"].str.contains("SEA-LEAP"))[1]  and (sim_data_to_plot["Workflow"].str.contains("Road sign inspection")[1])):
    sim_data_to_plot["Completion time [s]"]/=max_sea_rsi
elif((sim_data_to_plot["method"].str.contains("KCSS"))[1]  and (sim_data_to_plot["Workflow"].str.contains("Road sign inspection")[1])):
    sim_data_to_plot["Completion time [s]"]/=max_kcss_rsi
elif((sim_data_to_plot["method"].str.contains("C3-Match"))[1] and (sim_data_to_plot["Workflow"].str.contains("Sentiment analysis")[1])):
   sim_data_to_plot["Completion time [s]"]/=max_c3match_sa
elif((sim_data_to_plot["method"].str.contains("NAN"))[1]  and (sim_data_to_plot["Workflow"].str.contains("Sentiment analysis")[1])):
    sim_data_to_plot["Completion time [s]"]/=max_nan_sa
elif((sim_data_to_plot["method"].str.contains("SEA-LEAP"))[1]  and (sim_data_to_plot["Workflow"].str.contains("Sentiment analysis")[1])):
    sim_data_to_plot["Completion time [s]"]/=max_sea_sa
elif((sim_data_to_plot["method"].str.contains("KCSS"))[1]  and (sim_data_to_plot["Workflow"].str.contains("Sentiment analysis")[1])):
    sim_data_to_plot["Completion time [s]"]/=max_kcss_sa
#sim_data_to_plot.to_csv("D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\UnChangedOne\\pplot-simulation-5-5-22-w.csv")
#sim_data_to_plot["Completion time [s]"]/=np.max(sim_data_to_plot["Completion time [s]"])
#print(sim_data_to_plot)'''

sns.set(font_scale = 2)
sns.set_style(style='white')
ax1 = sns.boxplot(x="Workflow", y="Completion time [s]", hue="method", data=sim_data_to_plot, width=0.75,palette=colors_list,fliersize=0) 
handles, labels = ax1.get_legend_handles_labels()
#plt.yticks([0,0.15,0.25,0.35])
plt.legend(handles[0:4], labels[0:4])
plt.show()


real_data_to_plot = pd.read_csv("D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\UnChangedOne\\pplot-real-5-5-22.csv")
#print(preprocessing.normalize([np.array(real_data_to_plot["Completion time [s]"])]))
#real_data_to_plot["Completion time [s]"]/=np.max(real_data_to_plot["Completion time [s]"])
#real_data_to_plot.to_csv("D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\UnChangedOne\\pplot-real-5-5-22-w.csv")

'''max_c3match_rsi =0
max_c3match_sa =0
max_nan_rsi =0
max_nan_sa =0
max_sea_rsi =0
max_sea_sa =0
max_kcss_rsi =0
max_kcss_sa =0

with open('D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\UnChangedOne\\pplot-real-5-5-22.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        print(row["Completion time [s]"][0:20])
        #print(f'Column names are {", ".join(row)}')
        if((row["method"]=="C3-Match") and row["Workflow"]=="Road sign inspection"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(float(row["Completion time [s]"])>= max_c3match_rsi):
                max_c3match_rsi=float(row["Completion time [s]"])
        elif((row["method"]=="NAN") and row["Workflow"]=="Road sign inspection"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(float(row["Completion time [s]"])>= max_nan_rsi):
                max_nan_rsi=float(row["Completion time [s]"])
        elif((row["method"]=="SEA-LEAP") and row["Workflow"]=="Road sign inspection"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(float(row["Completion time [s]"])>= max_sea_rsi):
                max_sea_rsi=float(row["Completion time [s]"])
        elif((row["method"]=="KCSS") and row["Workflow"]=="Road sign inspection"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(float(row["Completion time [s]"])>= max_kcss_rsi):
                max_kcss_rsi=float(row["Completion time [s]"])
        
        if((row["method"]=="C3-Match") and row["Workflow"]=="Sentiment analysis"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(float(row["Completion time [s]"])>= max_c3match_sa):
                max_c3match_sa=float(row["Completion time [s]"])
        elif((row["method"]=="NAN") and row["Workflow"]=="Sentiment analysis"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(float(row["Completion time [s]"])>= max_nan_sa):
                max_nan_sa=float(row["Completion time [s]"])
        elif((row["method"]=="SEA-LEAP") and row["Workflow"]=="Sentiment analysis"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(float(row["Completion time [s]"])>= max_sea_sa):
                max_sea_sa=float(row["Completion time [s]"])
        elif((row["method"]=="KCSS") and row["Workflow"]=="Sentiment analysis"):
            #row["Completion time [s]"]=((row["Completion time [s]"])-)/(np.max(row["Completion time [s]"]))
            if(float(row["Completion time [s]"])>= max_kcss_sa):
                max_kcss_sa=float(row["Completion time [s]"])

#print((df1["method"]=="C3-Match"))
print(max_c3match_rsi," " ,max_nan_rsi," " ,max_sea_rsi," " ,max_kcss_rsi)
print(max_c3match_sa," " ,max_nan_sa," " ,max_sea_sa," " ,max_kcss_sa)
#print((real_data_to_plot["method"].str.contains("C3-Match")))
#print(real_data_to_plot["Completion time [s]"])
if((real_data_to_plot["method"].str.contains("C3-Match"))[1] and (real_data_to_plot["Workflow"].str.contains("Road sign inspection")[1])):
   real_data_to_plot["Completion time [s]"]/=max_c3match_rsi#[0:20]
elif((real_data_to_plot["method"].str.contains("NAN"))[1] and (real_data_to_plot["Workflow"].str.contains("Road sign inspection")[1])):
    real_data_to_plot["Completion time [s]"]/=max_nan_rsi#[20:40]
elif((real_data_to_plot["method"].str.contains("SEA-LEAP"))[1]  and (real_data_to_plot["Workflow"].str.contains("Road sign inspection")[1])):
    real_data_to_plot["Completion time [s]"]/=max_sea_rsi#[40:60]
elif((real_data_to_plot["method"].str.contains("KCSS"))[1]  and (real_data_to_plot["Workflow"].str.contains("Road sign inspection")[1])):
    real_data_to_plot["Completion time [s]"]/=max_kcss_rsi#[60:80]
elif((real_data_to_plot["method"].str.contains("C3-Match"))[1] and (real_data_to_plot["Workflow"].str.contains("Sentiment analysis")[1])):
   real_data_to_plot["Completion time [s]"]/=max_c3match_sa#[80:100]
elif((real_data_to_plot["method"].str.contains("NAN"))[1]  and (real_data_to_plot["Workflow"].str.contains("Sentiment analysis")[1])):
    real_data_to_plot["Completion time [s]"]/=max_nan_sa#[100:120]
elif((real_data_to_plot["method"].str.contains("SEA-LEAP"))[1]  and (real_data_to_plot["Workflow"].str.contains("Sentiment analysis")[1])):
    real_data_to_plot["Completion time [s]"]/=max_sea_sa#[120:140]
elif((real_data_to_plot["method"].str.contains("KCSS"))[1]  and (real_data_to_plot["Workflow"].str.contains("Sentiment analysis")[1])):
    real_data_to_plot["Completion time [s]"]/=max_kcss_sa#[140:160]
#print(real_data_to_plot["Completion time [s]"][140:160])'''

sns.set(font_scale = 2)
sns.set_style(style='white')
ax0 = sns.boxplot(x="Workflow", y="Completion time [s]", hue="method", data=real_data_to_plot, width=0.75,palette=colors_list,fliersize=0)
handles, labels = ax0.get_legend_handles_labels()
#plt.yticks([0,0.15,0.25,0.35])
plt.legend(handles[0:4], labels[0:4], loc='lower right')
plt.show()