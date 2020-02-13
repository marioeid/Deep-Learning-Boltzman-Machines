# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')

# preparing the training set and the test set
training_set=pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set=np.array(training_set,dtype='int')
test_set=pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set=np.array(test_set,dtype='int')

# getting the number of users and movies 
nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])));
nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))

# converting the data to users in lines and movies in column
def convert(data):
    # we wil create a list of lists cause we are gonna use torch afterwards 
    new_data=[]
    for id_users in range(1,nb_users+1): # the last user is excluded so we add one 
        id_movies=data[:,1][data[:,0]==id_users] # based on condition
        id_ratings=data[:,2][data[:,0]==id_users] 
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data

training_set=convert(training_set)
test_set=convert(test_set)   

# converting the data into torch tensors 
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

# converting the ratings into binary ratings 1 (liked) 0 (NOt liked)
training_set[training_set==0]=-1  # all zeros are unrated movies
training_set[training_set==1]=0    # or operation is different for torch tensors
training_set[training_set==2]=0
training_set[training_set>=3]=1
test_set[test_set==0]=-1  # all zeros are unrated movies
test_set[test_set==1]=0    # or operation is different for torch tensors
test_set[test_set==2]=0
test_set[test_set>=3]=1

# creating the architecture of the neural network
class RBM():
     def __init__(self,nv,nh,):# number of visible,number hidden nodes
         self.W = torch.randn(nh,nv)      # to intialize a variable belongs to the object
         # the weights of szie nh , nv
         self.a = torch.randn(1,nh) # one bias for each hidden node additional dim for the batch
                           # batch,bias
         self.b = torch.randn(1,nv) # the bias for the visible nodes
     
     def sample_h(self,x): # the visible neurenos x 
         wx=torch.mm(x,self.W.t()) # product of two tensors x and w transpose
         activation=wx+self.a.expand_as(wx)
         p_h_given_v=torch.sigmoid(activation)
         return p_h_given_v,torch.bernoulli(p_h_given_v)
         
     def sample_v(self,y): # the hidden neurenos y 
         wy=torch.mm(y,self.W) # product of two tensors x and w transpose
         activation=wy+self.b.expand_as(wy) # bias of the visible nodes
         p_v_given_h=torch.sigmoid(activation)
         return p_v_given_h,torch.bernoulli(p_v_given_h)
                   
     def train(self,v0,vk,ph0,phk):
         self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
         self.b+=torch.sum((v0-vk),0)
         self.a+=torch.sum((ph0-phk),0)
         

nv=len(training_set[0])
nh=100 
batch_size=100
rbm=RBM(nv,nh)         
        
# training the RBM
nb_epoch=10
for epoch in range(1,nb_epoch+1):
    train_loss=0
    s=0. # float type cause of the dot
    for id_user in range(0,nb_users-batch_size,batch_size): # go from 0 to 99 from 100 to 199 the last parameter is the looping step
        vk=training_set[id_user:id_user+batch_size]
        v0=training_set[id_user:id_user+batch_size]
        ph0,_=rbm.sample_h(v0)  # ph0,_ is to get the first return only not two elments               
        
        for k in range(10): # gibbs sampling 
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]      # the unrated movies (-1) should remain like this 
        
        phk,_=rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss+=torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s+=1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))    
               
# testing the RBM

test_loss=0
s=0. # float type cause of the dot
for id_user in range(nb_users): # go from 0 to 99 from 100 to 199 the last parameter is the looping step
    v=training_set[id_user:id_user+1]
    vt=test_set[id_user:id_user+1]
   
    if len(vt[vt>=0])>0:
        _,h=rbm.sample_h(v)
        _,v=rbm.sample_v(h)
        test_loss+=torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s+=1.
print('test loss: '+str(test_loss/s))    
           
    