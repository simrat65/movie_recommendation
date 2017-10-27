import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# Movie Recommendation without genres
u_cols = ['user_id','sex','age','occupation', 'zip_code']
u_file=pd.read_csv(r'./users.dat',sep='::',names=u_cols,engine='python')
u_file.index=u_file['user_id']
r_cols= ['user_id','movie_id','rating','timestamp']
r_file=pd.read_csv(r'./ratings.dat',sep='::',names=r_cols,engine='python')
g_cols=["Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance", r"Sci-Fi","Thriller","War","Western"]
x_cols=["Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance", r"Sci-Fi","Thriller","War","Western","genre","movie_id","title"]
items = pd.read_csv(r'./ml-1m/movie.csv',engine='python')

items = items.fillna(0)
y = items['movie_id']
int_index = [int(i) for i in y]
y = int_index
items.index = y
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


corr = cosine_similarity(items[g_cols])
u_file= pd.concat([u_file,pd.DataFrame(columns=list(y))])

for i in range(1000209):
    user = int(r_file.loc[i,['user_id']])
    movie = int(r_file.loc[i,['movie_id']])
    rating = int(r_file.loc[i,['rating']])
    u_file.loc[user,movie] = rating
    if i%10000 ==0:
         print(i)
user_corr = cosine_similarity(u_file[occ+age+['sex']])
x = u_file['user_id']
cij = np.corrcoef(u_file[y])
w=0.1
net_corr = w*cij+(1-w)*user_corr
def avg(pre):
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
def predict(u,m):
    x = u_file['user_id']
    y = items['movie_id']
    pre= u_file.loc[u,y]
    av  = avg(pre)
            
    sum=0
    corr = 0
    for i in x:
        iavg = u_file.loc[i,y]
        irating = u_file.loc[i,m]
        iavg  = avg(iavg)
        corr += abs(net_corr[u,i-1])
        sum = net_corr[u,i-1]*(irating - iavg)
    net_rating = av + sum/corr
    return net_rating
