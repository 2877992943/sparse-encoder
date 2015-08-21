#!/usr/bin/env python
# encoding=utf-8

__docformat__='restructedtext en'
 

import cPickle,gzip,os,sys,time,theano
import theano.tensor as T
parapath='/home/yr/theanoExercise/autoEncoder/para3'
import numpy as np
from PIL import Image


 
 
def show_params():
 
	f=open(parapath,'rb')
	w=cPickle.load(f)
	print np.shape(w)#[784,500]
	f.close()
	n=10#3x3 total 9 kernels
	pic=np.zeros((28*n+2*n,28*n+2*n))
	for i in range(n):
		for j in range(n):
			index=i*n+j
			sqr0=np.zeros((30,30))
			sqr=w[:,index].reshape(28,28)
			sqr0[2:30,2:30]=sqr
			pic[i*30:i*30+30,j*30:j*30+30]=sqr0
	#########
	return pic

		
	

def array_pic(arr):
	import pylab
	pylab.figure()
	pylab.gray()
	pylab.imshow(arr)
	pylab.show()
	im=Image.fromarray(np.uint8(arr))
	im.save('/home/yr/theanoExercise/autoEncoder/pic1.jpg')
	



if __name__=='__main__':
	picarr=show_params()
	array_pic(picarr)
     
		    

		
	
	








