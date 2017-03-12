import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

u_cols = ['user_id','sex','age','occupation', 'zip_code']
u_file=pd.read_csv(r'./users.dat',sep='::',names=u_cols,engine='python')
u_file.index=u_file['user_id']
r_cols= ['user_id','movie_id','rating','timestamp']
r_file=pd.read_csv(r'./ratings.dat',sep='::',names=r_cols,engine='python')
g_cols=["Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance", r"Sci-Fi","Thriller","War","Western"]
items = pd.read_csv(r'./movies.dat', sep='::', names=['movie_id','title','genre'],engine='python')
items= pd.concat([items,pd.DataFrame(columns=list(g_cols))])

for i in range(3883):
    t=items['genre'][i]
    t=t.split('|')
    for j in g_cols:
         for k in t:
            if k=="Children's":
                k='Children'
            if j==k:
                items[k][i]=1
items = items.fillna(0)
items.index = items['movie_id']
del items['genre']
u_file['sex']=u_file['sex']=='M'
occ=["other","academic/educator","artist","clerical/admin","college/grad student","customer service","doctor/health care","executive/managerial","farmer","homemaker","K-12 student","lawyer","programmer","retired","sales/marketing","scientist","self-employed","technician/engineer","tradesman/craftsman","unemployed","writer"]
age=["Under 18","18-24","25-34","35-44","45-49","50-55","56+"]
u_file = pd.concat([u_file,pd.DataFrame(columns=list(age))])
u_file = pd.concat([u_file,pd.DataFrame(columns=list(occ))])
u_file['other']=u_file['occupation']==0
u_file['academic/educator'] =u_file['occupation']==1
u_file['artist'] =u_file['occupation']==2
u_file['clerical/admin'] =u_file['occupation']==3
u_file['college/grad student'] =u_file['occupation']==4
u_file['customer service'] =u_file['occupation']==5
u_file['doctor/health care'] =u_file['occupation']==6
u_file['executive/managerial'] =u_file['occupation']==7
u_file['farmer'] =u_file['occupation']==8
u_file['homemaker'] =u_file['occupation']==9
u_file['K-12 student'] =u_file['occupation']==10
u_file['lawyer'] =u_file['occupation']==11
u_file['programmer'] =u_file['occupation']==12
u_file['retired'] =u_file['occupation']==13
u_file['sales/marketing'] =u_file['occupation']==14
u_file['scientist'] =u_file['occupation']==15
u_file['self-employed'] =u_file['occupation']==16
u_file['technician/engineer'] =u_file['occupation']==17
u_file['tradesman/craftsman'] =u_file['occupation']==18
u_file['unemployed'] =u_file['occupation']==19
u_file['writer'] =u_file['occupation']==20
u_file['Under 18'] =u_file['age']==1
u_file['18-24'] =u_file['age']==18
u_file['25-34'] =u_file['age']==25
u_file['35-44'] =u_file['age']==35
u_file['45-49'] =u_file['age']==45
u_file['50-55'] =u_file['age']==50
u_file['56+'] =u_file['age']==56


y = items['movie_id']
corr = cosine_similarity(items[g_cols])#movies correlation
u_file= pd.concat([u_file,pd.DataFrame(columns=list(y))])

for i in range(1000):
    
    user = r_file.loc[i,['user_id']]
    movie = r_file.loc[i,['movie_id']]
    rating = r_file.loc[i,['rating']]
    u_file.loc[user,movie] = rating
    if i%1000 == 0:
         print(i)
x = u_file.index
user_corr=pd.DataFrame(index=x, columns=x)
cij = pd.DataFrame(index=x, columns=x)
def avg(pre):#function to find average
     p =0
     q=0
     av =0
     for i in pre:
         if i!=0:
             p+=1
             q+=i
     if p!=0:
         av = q/p  
     return av
for i in x:#calculation for cij
    t = u_file.loc[i,occ+age+['sex']]
    list_movie_i = u_file.loc[i,y]
    iavg = avg(list_movie_i)
    list_movie_i = list_movie_i.fillna(iavg)
    for j in x:
        list_movie_j = u_file.loc[j,y]
        javg = avg(list_movie_j)
        k = u_file.loc[j,occ+age+['sex']]
        user_corr.loc[i,j]=cosine(t,k)
        o=0
        m=0
        n=0
        for p in y:
            ri = u_file.loc[i,p]
            for q in y:
                rj=u_file.loc[j,q]
                o += (ri-iavg)*(rj-javg)*corr[p,q]
                m+=(ri-iavg)*(ri-iavg)*corr[p,q]
                cij.loc[i,j]=o/(m*n)

net_corr = cij*w + user_corr*(1-w)
  
def predict(u,m):
    x = u_file['user_id']
    y = items['movie_id']
    pre= u_file.loc[u,y]
    av = avg(pre)
    sum=0
    corr = 0
    for i in x:
        iavg = u_file.loc[i,y]
        irating = u_file.loc[i,m]
        iavg  = mean(iavg)
        corr += abs(net_corr.loc[u,i])
        sum = net_corr.loc[u,i]*(irating - iavg)
    net_rating = av + sum/corr
    return net_rating
print(net_corr)






