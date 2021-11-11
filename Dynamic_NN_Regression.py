import tensorflow as tf
import numpy as np
class NN:
  def __init__(self,layers=[9,3,5,5,1],activations=None):
    self.W=[]
    self.B=[]
    self.a=[]
    if activations:
      self.a=activations
    else:
      for i in  range(len(layers)-1):
        self.a.append(tf.nn.relu)
    for i in range(len(layers)-1):
      self.W.append(tf.Variable(tf.random.uniform([layers[i],layers[i+1]],minval=-1,maxval=1),dtype=tf.float32))
      self.B.append(tf.Variable(tf.random.uniform([layers[i+1]],minval=-1,maxval=1),dtype=tf.float32))
  def predict(self,X):
    Z=tf.Variable(np.transpose(X),dtype=tf.float32)
    for w,b,a in zip(self.W,self.B,self.a):
      Z=tf.add(tf.matmul(Z,w),b)
      Z=a(Z)
    return Z
  def loss(self,Y_pred,Y_target):
    return tf.reduce_mean(tf.square(Y_pred-Y_target))
  def fit(self,X,Y,epoch,lr=0.001):
    for i in range(epoch):
      with tf.GradientTape() as tape:
        cost=self.loss(self.predict(X),Y)
        gradients=tape.gradient(cost,[self.W,self.B])
        for dw,db,w,b in zip(gradients[0],gradients[1],self.W,self.B):
          w.assign_sub(dw*lr)
          b.assign_sub(db*lr)
      if (i%100)==0:
        print(f"epoch:{i}  loss:",cost.numpy())

nn=NN(layers=[9,3,5,5,1])
x=np.array([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]])
y=np.array([[12],[22]])
nn.fit(X=x,Y=y,epoch=3000)

#use out neural net to predict unseen data
x=np.array([3,3,3,3,3,3,3,3,3])
x=x[:,np.newaxis]
print(neuralNet.predict(x).numpy())
