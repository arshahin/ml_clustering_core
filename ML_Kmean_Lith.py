
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from sklearn.cluster import KMeans
#from sklearn import svm
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator

#import the dataset

#df = pd.read_csv (r'D:\PHD\Thesis\Python_Code\Thesis_Code\Match_Data_eclid3.CSV')
df = pd.read_csv (r'.\DATA.CSV')



print(df.head(2))
#scaler = preprocessing.StandardScaler().fit(df)
#print(scaler)
#X_scaled = scaler.transform(df)
#print(X_scaled)
#x = X_scaled[0:, [1,2,3,4,5,6,7,8,9]]
#y = X_scaled[0:, [0]]
#x = df.iloc[:, [11,12,13,14,15,17,18]].values
x = df.iloc[:, [12,13,14,15,17,18]].values
#scaler = preprocessing.StandardScaler().fit(x)
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(x)
print(scaled)
x=scaled
y = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,16,19,20,21,22,23,24]].values
print(x.shape)
print(x[:,:])
print(y)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ FISRT STEP CLUSTERING  & LABELING BY KMEANS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#******************begin of Elbow for determine K ***************************
Error =[]

for i in range(1, 20):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
plt.style.use("fivethirtyeight")       
plt.plot(range(1, 20), Error)
plt.title('Elbow method for Determine K')
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show()

kl = KneeLocator(range(1, 20), Error, curve="convex", direction="decreasing")
k=kl.elbow
print(kl.elbow)
print('k=',k)
#****************************************************************************
kl = KneeLocator(range(1, 20), Error, curve="convex", direction="decreasing")
k=kl.elbow
print(kl.elbow)
print('k=',k)
#************************end of Elbow************************************
k=4
#km = KMeans(n_clusters=k, init='k-means++', max_iter=3000, n_init=2, verbose=2, random_state=339)
km = KMeans(n_clusters=k, max_iter=300, n_init=50, verbose=2, random_state=3245,algorithm='elkan')

y_kmeans5 = km.fit_predict(x)
print(y_kmeans5)

result = np.column_stack((y,x, y_kmeans5))
print(result[0,:])
print(result)

cl=[]
end_of_test=[]
for cl_no in range(0,k):
    cl.append(result[np.where(result[:,24] == cl_no)])
    print(cl[cl_no].shape)
   # print(cl[cl_no])
    end_of_test.append(round(0.8*len(cl[cl_no])))
   # print(end_of_test[cl_no])
n=k
m=5
r=[[0]*m]*n
for i in range(0,k):
            ANYTRID=0
            Calcite=0
            Dolomoite=0 
            SAND=0
            SHEEL=0
            for j in range(0,len(result)):
                #print(result[j,24])
                #print(result[j,23])
                if result[j,17]=="ANYTRID" and result[j,24]==i :
                    ANYTRID+=1
                if result[j,17]=='Calcite' and result[j,24]==i:
                    Calcite+=1
                if result[j,17]=='Dolomoite' and result[j,24]==i :
                    Dolomoite+=1
                if result[j,17]=='SAND' and result[j,24]==i :
                     SAND+=1
                if result[j,17]=='SHEEL' and result[j,24]==i :
                    SHEEL+=1
            r[i][0]=ANYTRID
            r[i][1]=Calcite
            r[i][2]=Dolomoite
            r[i][3]=SAND
            r[i][4]=SHEEL
            print(r[0])



#import matplotlib.pylab as plb
#print(cl[3])
plt1.scatter(result[:,22],result[:,23],c=y_kmeans5,cmap='rainbow')
#z=np.polyfit(np.nan_to_num(result[:,23]),np.nan_to_num(result[:,24]),1)
#p=np.poly1d(z)
#plb.plot(result[:,23],p(result[:,23]),'m-')
plt1.title("kmeans after depth match,lithology,por,perm")
plt1.xlabel("permeability")
plt1.ylabel("porosity")
plt1.show() 