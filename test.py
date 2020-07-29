'''

Testing module to test the presence of Anomalous Events in a Video

The module computes reconstruction loss between input bunch and

the reconstructed batch from the model, and flagges the batch as anomalous
if loss value is greater than a given threshold.








Author: Somandra Singh Rathore

'''


from tensorflow.keras.models import load_model
import numpy as np 
from tabulate import tabulate



def mean_squared_loss(x1,x2):


	''' Compute Euclidean Distance Loss  between 
	input frame and the reconstructed frame'''




	diff=x1-x2
	a,b,c,d,e=diff.shape
	n_samples=a*b*c*d*e
	sq_diff=diff**2
	Sum=sq_diff.sum()
	dist=np.sqrt(Sum)
	mean_dist=dist/n_samples

	return mean_dist



'''Define threshold for Sensitivity
Lower the Threshhold,higher the chances that a bunch of frames will be flagged as Anomalous.

'''

threshold=0.0004



model=load_model('AnomalyDetector.h5')

X_test=np.load('test.npy')
frames=X_test.shape[2]
#Need to make number of frames divisible by 10


flag=0 #Overall video flagq

frames=frames-frames%10

X_test=X_test[:,:,:frames]
X_test=X_test.reshape(-1,227,227,10)
X_test=np.expand_dims(X_test,axis=4)
result = []
anomalous_count = 0
for number,bunch in enumerate(X_test):
	t = [number]
	n_bunch=np.expand_dims(bunch,axis=0)
	reconstructed_bunch=model.predict(n_bunch)

	loss=mean_squared_loss(n_bunch,reconstructed_bunch)
	#print("loss:"+str(loss))
	t.append(loss)

	if loss>threshold:
		#print("Anomalous bunch of frames at bunch number {}".format(number))
		anomalous_count+=1
		flag=1
		t.append("Anomalous")


	else:
		#print('Bunch Normal')
		t.append("Normal")
	result.append(t)


print(tabulate(result,headers=["Bunch No.","Loss","Type"],tablefmt="fancy_grid"))

print("Anomalous Events detected::"+str(anomalous_count))















