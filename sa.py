#import packages
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import collections
import scipy 
import scipy.interpolate 
import glob
import csv
from numpy import loadtxt
from scipy import optimize
from scipy import stats	
from scipy import interpolate
from tkinter import *
import xlwt
from tkinter import messagebox
from matplotlib.backends.backend_pdf import PdfPages
import csv

def training():
	global array
	global arrayc
	global arraye
	global arraya
	global maint1c
	global maint2co
	global extint
	global prgc
	global nlist
	global priming_threshold
	global dose1
	global dose2
	global maint1
	global fig1
	global fig2
	global loadint
	global resultsm1
	global resultsm2
	global lengthm1
	global loadintlist
	global m1int
	global m2int
	global lengthprg
	global countsprg
	global prg
	global avglist
	global avglistlabel
	global semlist
	global resultsload
	global extint
	global avglistlabel
	global semlist
	global resultsload
	global lintlist
	global fig1
	global fig2
	global maint2counts
	global extc
	global age
	data_file = open(e1.get())
	blockO = ''
	found = False
	for line in data_file:
			if found:
				blockO += line
				if line.strip() == "R:": break
			else:
				if line.strip() == "O:":
					found = True
	sbto=blockO.split(':')
	newlisto=[item.split() for item in ' '.join(sbto).split("'") if item]
	with open('someo.csv', 'w') as f:
		writer=csv.writer(f)
		writer.writerows(newlisto)
	arrayo=np.genfromtxt('someo.csv', delimiter=',')

	blockU = ''
	found = False
	for line in data_file:
			if found:
				blockU += line
				if line.strip() == "V:": break
			else:
				if line.strip() == "U:":
					found = True
	sbtu=blockU.split(':')
	newlistu=[item.split() for item in ' '.join(sbtu).split("'") if item]
	with open('someu.csv', 'w') as f:
		writer=csv.writer(f)
		writer.writerows(newlistu)
	arrayu=np.genfromtxt('someu.csv', delimiter=',')

	data_file.close()

	dose1=arrayo[1]
	dose2=arrayo[2]

	try:
		data_file = open(e1.get()) #import data
	except:
		tkMessageBox.showwarning('File Error', "Enter a valid file name. \n File must be located in the same folder as the script.")

	blockA = "" #create empty list
	found = False
	for line in data_file: #read through the file, finding letters and breaking at them
	    if found:
	        blockA += line
	        if line.strip() == "B:": break #ending letter
	    else:
	        if line.strip() == "A:": #starting letter
	            found = True
	            blockA = "A:"
	sbta=blockA.split(':') #split the string
	newlista=[item.split() for item in ' '.join(sbta).split("'") if item] #write string into a list
	with open('somea.csv', 'w') as f:
	    writer=csv.writer(f)
	    writer.writerows(newlista) #write into a dummy CSV file for formatting
	arraya=np.genfromtxt('somea.csv', delimiter=',') #create numpy array of data from the dummy .csv
	arraya = arraya[np.logical_not(np.isnan(arraya))]            
	blockC = ""
	found = False
	for line in data_file:
	    if found:
	        blockC += line
	        if line.strip() == "D:": break
	    else:
	        if line.strip() == "C:":
	            found = True
	sbtc=blockC.split(':')
	newlistc=[item.split() for item in ' '.join(sbtc).split("'") if item]
	with open('somec.csv', 'w') as f:
	    writer=csv.writer(f)
	    writer.writerows(newlistc)
	arrayc=np.genfromtxt('somec.csv', delimiter=',')

	blockE = ""
	found = False
	for line in data_file:
	    if found:
	        blockE += line
	        if line.strip() == "G:": break
	    else:
	        if line.strip() == "E:":
	            found = True
	sbte=blockE.split(':')
	newliste=[item.split() for item in ' '.join(sbte).split("'") if item]
	with open('somee.csv', 'w') as f:
	    writer=csv.writer(f)
	    writer.writerows(newliste)
	arraye=np.genfromtxt('somee.csv', delimiter=',')

	blockT = ''
	found = False
	for line in data_file:
			if found:
				blockT += line
				if line.strip() == "U:": break
			else:
				if line.strip() == "T:":
					found = True
	sbt=blockT.split(':')
	newlist=[item.split() for item in ' '.join(sbt).split("'") if item]
	with open('some.csv', 'w') as f:
		writer=csv.writer(f)
		writer.writerows(newlist)
	array=np.genfromtxt('some.csv', delimiter=',')

	data_file.close()
	
	var1output=var1.get()

	array = array[np.logical_not(np.isnan(array))]
	arrayc = arrayc[np.logical_not(np.isnan(arrayc))]
	arraye = arraye[np.logical_not(np.isnan(arraye))]
	#deletes the line labels
	array = np.delete(array, np.arange(0, array.size, 11))
	arrayc = np.delete(arrayc, np.arange(0, arrayc.size, 11))
	arraye=np.delete(arraye, np.arange(0, arraye.size, 11))
	var1output=var2.get()
	if var1output==0:
		age=arraya[6]
	else:
		age=e2.get()

	#reads the event codes to only take the inejctions and relevant events

	maint1indices=[np.where(np.logical_or(arraye==64400111, arraye==62100111))]
	maint1c=np.take(arrayc, maint1indices)
	maint1=np.take(array,maint1indices)
	maint1=np.array(maint1)
	lengthm1=maint1.size
	countsm1=list(range(lengthm1))
	countsm1=np.array(countsm1)
	maint1=maint1.reshape(lengthm1,)
	maint1c=maint1c.reshape(lengthm1,)

	fig1=plt.figure()
	fig1.suptitle(arraya[4], fontsize=14, fontweight='bold')
	ax=fig1.add_subplot(111)
	ax.set_title(age, fontsize=14)
	fig1.subplots_adjust(top=0.85)
	ax.scatter(maint1, countsm1)

	fig2=plt.figure()
	ax1=fig2.add_subplot(111)
	ax1.scatter(maint1, maint1c)
	plt.show()

	m1list=maint1.tolist()
	m1clist=maint1c.tolist()
	geninfolabel=['rat#', 'day', 'training']
	geninfo=[arraya[4], arraya[6], ['training session!!']]
	header=['maint1 intervals', 'maint1 conc', ' ', ' ']

	filename='rat%sd%s.csv'%(arraya[4], age)
	from itertools import zip_longest
	d=[m1list,m1clist, geninfolabel,geninfo]
	export_data=zip_longest(*d, fillvalue=' ')
	with open(filename, 'w',encoding="ISO-8859-1", newline='') as f:
		writer=csv.writer(f)
		writer.writerow(header)
		writer.writerows(export_data)

	pltfilename='rat%sd%s.pdf'%(arraya[4], age)
	with PdfPages(pltfilename) as pdf:
		pdf.savefig(fig1)
		pdf.savefig(fig2)	

	plt.close()

def analysis():
	#all global declarations are to make the variable available outside of the analysi funciton
	global turning_point
	global array
	global arrayc
	global arraye
	global arraya
	global maint1c
	global maint2counts
	global extint
	global prgc
	global nlist
	global priming_threshold
	global dose1
	global dose2
	global maint1
	global fig1
	global fig2
	global loadint
	global resultsm1
	global resultsm2
	global lengthm1
	global loadintlist
	global m1int
	global m2int
	global lengthprg
	global countsprg
	global prg
	global avglist
	global avglistlabel
	global semlist
	global resultsload
	global extint
	global avglistlabel
	global semlist
	global resultsload
	global lintlist
	global fig1
	global fig2
	global maint2co
	global extc
	global age
	data_file = open(e1.get())
	blockO = ''
	found = False
	for line in data_file:
			if found:
				blockO += line
				if line.strip() == "R:": break
			else:
				if line.strip() == "O:":
					found = True
	sbto=blockO.split(':')
	newlisto=[item.split() for item in ' '.join(sbto).split("'") if item]
	with open('someo.csv', 'w') as f:
		writer=csv.writer(f)
		writer.writerows(newlisto)
	arrayo=np.genfromtxt('someo.csv', delimiter=',')

	blockU = ''
	found = False
	for line in data_file:
			if found:
				blockU += line
				if line.strip() == "V:": break
			else:
				if line.strip() == "U:":
					found = True
	sbtu=blockU.split(':')
	newlistu=[item.split() for item in ' '.join(sbtu).split("'") if item]
	with open('someu.csv', 'w') as f:
		writer=csv.writer(f)
		writer.writerows(newlistu)
	arrayu=np.genfromtxt('someu.csv', delimiter=',')

	data_file.close()

	penultimate_priming_dose=arrayu[4]
	ultimate_priming_dose=arrayu[5]
	priming_threshold=(penultimate_priming_dose+ultimate_priming_dose)/2
	dose1=arrayo[1]
	dose2=arrayo[2]

	try:
		data_file = open(e1.get()) #import data
	except:
		tkMessageBox.showwarning('File Error', "Enter a valid file name. \n File must be located in the same folder as the script.")

	blockA = "" #create empty list
	found = False
	for line in data_file: #read through the file, finding letters and breaking at them
	    if found:
	        blockA += line
	        if line.strip() == "B:": break #ending letter
	    else:
	        if line.strip() == "A:": #starting letter
	            found = True
	            blockA = "A:"
	sbta=blockA.split(':') #split the string
	newlista=[item.split() for item in ' '.join(sbta).split("'") if item] #write string into a list
	with open('somea.csv', 'w') as f:
	    writer=csv.writer(f)
	    writer.writerows(newlista) #write into a dummy CSV file for formatting
	arraya=np.genfromtxt('somea.csv', delimiter=',') #create numpy array of data from the dummy .csv
	arraya = arraya[np.logical_not(np.isnan(arraya))]            
	blockC = ""
	found = False
	for line in data_file:
	    if found:
	        blockC += line
	        if line.strip() == "D:": break
	    else:
	        if line.strip() == "C:":
	            found = True
	sbtc=blockC.split(':')
	newlistc=[item.split() for item in ' '.join(sbtc).split("'") if item]
	with open('somec.csv', 'w') as f:
	    writer=csv.writer(f)
	    writer.writerows(newlistc)
	arrayc=np.genfromtxt('somec.csv', delimiter=',')

	blockE = ""
	found = False
	for line in data_file:
	    if found:
	        blockE += line
	        if line.strip() == "G:": break
	    else:
	        if line.strip() == "E:":
	            found = True
	sbte=blockE.split(':')
	newliste=[item.split() for item in ' '.join(sbte).split("'") if item]
	with open('somee.csv', 'w') as f:
	    writer=csv.writer(f)
	    writer.writerows(newliste)
	arraye=np.genfromtxt('somee.csv', delimiter=',')

	blockT = ''
	found = False
	for line in data_file:
			if found:
				blockT += line
				if line.strip() == "U:": break
			else:
				if line.strip() == "T:":
					found = True
	sbt=blockT.split(':')
	newlist=[item.split() for item in ' '.join(sbt).split("'") if item]
	with open('some.csv', 'w') as f:
		writer=csv.writer(f)
		writer.writerows(newlist)
	array=np.genfromtxt('some.csv', delimiter=',')

	data_file.close()
	
	var1output=var1.get()

	array = array[np.logical_not(np.isnan(array))]
	arrayc = arrayc[np.logical_not(np.isnan(arrayc))]
	arraye = arraye[np.logical_not(np.isnan(arraye))]
	#deletes the line labels
	array = np.delete(array, np.arange(0, array.size, 11))
	arrayc = np.delete(arrayc, np.arange(0, arrayc.size, 11))
	arraye=np.delete(arraye, np.arange(0, arraye.size, 11))
	var1output=var2.get()
	if var1output==0:
		age=arraya[6]
	else:
		age=e2.get()

	#reads the event codes to only take the inejctions and relevant events
	k1indices=np.where(arraye==64400115)
	k1=np.take(array,k1indices)
	k1=np.array(k1)
	lengthk1=k1.size
	countsk1=list(range(lengthk1))
	countsk1=np.array(countsk1)
	k1=k1.reshape(lengthk1,)
	k1c=np.take(arrayc, k1indices)
	k1c=k1c.reshape(lengthk1,)

	k9indices=np.where(arraye==63400009)
	k9=np.take(array,k9indices)
	k9=np.array(k9)
	lengthk9=k9.size
	countsk9=list(range(lengthk9))
	countsk9=np.array(countsk9)
	k9=k9.reshape(lengthk9,)
	k9c=np.take(arrayc, k9indices)
	k9c=k9c.reshape(lengthk9,)

	ldindices=np.where(arraye==42100111)
	ld=np.take(array,ldindices)
	ld=np.array(ld)
	lengthld=ld.size
	countsld=list(range(lengthld))
	countsld=np.array(countsld)
	ld=ld.reshape(lengthld,)
	ldc=np.take(arrayc, ldindices)
	ldc=ldc.reshape(lengthld,)

	prgmindices=np.where(arraye==31100200)
	prg=np.take(array,prgmindices)
	prg=np.array(prg)
	lengthprg=prg.size
	countsprg=list(range(lengthprg))
	countsprg=np.array(countsprg)
	prg=prg.reshape(lengthprg,)
	prgc=np.take(arrayc, prgmindices)
	prgc=prgc.reshape(lengthprg,)

	maint1indices=[np.where(np.logical_or(arraye==64400111, arraye==62100111))]
	maint1c=np.take(arrayc, maint1indices)
	maint1=np.take(array,maint1indices)
	maint1=np.array(maint1)
	lengthm1=maint1.size
	countsm1=list(range(lengthm1))
	countsm1=np.array(countsm1)
	maint1=maint1.reshape(lengthm1,)
	maint1c=maint1c.reshape(lengthm1,)

	maint2indices=np.where(arraye==74400111)
	maint2=np.take(array,maint2indices)
	maint2=np.array(maint2)
	lengthm2=maint2.size
	countsm2=list(range(lengthm2))
	countsm2=np.array(countsm2)
	maint2=maint2.reshape(lengthm2,)
	maint2co=np.take(arrayc, maint2indices)
	maint2co=maint2co.reshape(lengthm2,)


	extindices=[np.where((arraye>89999999) &(arraye<98000000))]
	extinction=np.take(array,extindices)
	extinction=np.array(extinction)
	lengthe=extinction.size
	countse=list(range(lengthe))
	countse=np.array(countse)
	extinction=extinction.reshape(lengthe, )
	extc=np.take(arrayc, extindices)
	extc=extc.reshape(lengthe, )
	global totaltime
	global totalconc
	totaltime=np.concatenate((ld, prg, maint1, maint2, extinction), axis=0)
	totalconc=np.concatenate((ldc, prgc, maint1c, maint2co, extc), axis=0)

	if len(k9>0):
		k9idx = (np.abs(maint1 - k9[-1])).argmin()
		m1take= list(range(0,k9idx))
		m2take=list(range(k9idx,len(maint1)))
		maint1s=maint1
		maint1cs=maint1c
		maint1=np.take(maint1s,m1take)
		maint2=np.take(maint1s,m2take)
		maint1c=np.take(maint1cs, m1take)
		maint2co=np.take(maint1cs, m2take)

	lengthm1=maint1.size
	countsm1=list(range(lengthm1))
	countsm1=np.array(countsm1)
	maint1=maint1.reshape(lengthm1,)
	maint1c=maint1c.reshape(lengthm1,)

	lengthm2=maint2.size
	countsm2=list(range(lengthm2))
	countsm2=np.array(countsm2)
	maint2=maint2.reshape(lengthm2,)
	maint2co=maint2co.reshape(lengthm2,)

	xm1=maint1
	ym1=countsm1
	

	#takes the second deriviative, and splits the list at that point
	outputm11 = []
	outputm12=[]
	each_part = []

	if len(xm1)>1:
		
		tck = interpolate.splrep(xm1, ym1, k=2, s=0)
		xnew = np.linspace(0, lengthm1)
		dev_2 = interpolate.splev(xm1, tck, der=2)
		turning_point_mask = dev_2 == np.amax(dev_2)
		turning_point= [i for i, xm1 in enumerate(turning_point_mask) if xm1]
		# turning_point = str(turning_point[0:])

		for i in range(len(maint1)):
			if i not in turning_point:
				each_part.append(maint1[i]) #put each element before each index to each_part
			else:
				each_part.append(maint1[i]) #put element that on the index to each_part
				outputm11.append(each_part) #put the each_part to the output
				each_part = [] #clear each_part

		if turning_point[-1] < len(maint1):
			last_part =  maint1[turning_point[-1]+1:] #put the element after last index to last part
			outputm12.append(last_part) #put the element after last index to output 

		turning_point=np.array(turning_point)
		turning_point=np.asscalar(turning_point)


		outputm11=np.asarray(outputm11)
		outputm12=np.asarray(outputm12)
		outputm11=outputm11.reshape(turning_point+1, )
		outputm12=outputm12.reshape(lengthm1-(turning_point+1), )
	#the counts are what is used in the cumulative event plots
		countsm11=list(range(0,turning_point+1))
		countsm12=list(range(1,((lengthm1-turning_point))))
		maint2counts=maint1c[len(outputm11):len(maint1c)]
		maint1c=maint1c[0:len(outputm11)]

		#linear regressions
		fit1=scipy.stats.linregress(outputm11,countsm11)
		fit2=scipy.stats.linregress(outputm12, countsm12)
		fit1c=scipy.stats.linregress(outputm11, maint1c)
		fit2c=scipy.stats.linregress(outputm12, maint2counts)

		fit3=stats.linregress(maint2+lengthm1,countsm2)
		fit3c=stats.linregress(maint2+lengthm1,maint2co)


#calculating the means, sems, etc
		loadint=np.diff(outputm11)
		avgldint=np.mean(loadint)
		loadsem=scipy.stats.sem(loadint)
		m1int=np.diff(outputm12)
		avgm1int=np.mean(m1int)
		m2int=np.diff(maint2)
		avgm2int=np.mean(m2int)
		extint=np.diff(extinction)
		avgextint=np.mean(extint)
		sdm1=scipy.stats.tstd(maint1)
		sdm2=scipy.stats.tstd(maint2)
		sdext=scipy.stats.tstd(extinction)
		m1sem=scipy.stats.sem(m1int)
		m2sem=scipy.stats.sem(m2int)
		extsem=scipy.stats.sem(extinction)


		avglist=[avgldint, avgm1int, avgm2int, avgextint]
		semlist=[loadsem, m1sem, m2sem, extsem]
		nlist=[turning_point, (lengthm1-turning_point), lengthm2, lengthe]
		loadintlist=loadint.tostring


		#these are all used to print out the results later
		resultsload=[avgldint, loadsem, len(outputm11)]
		resultsm1=[avgm1int, m1sem, len(outputm12)]
		resultsm2=[avgm2int, m2sem, len(maint2)]

		countsld=list(range(len(ld)))
	# the rest is making the graph. 
		fig1=plt.figure()
		fig1.suptitle(arraya[4], fontsize=14, fontweight='bold')
		ax=fig1.add_subplot(111)
		ax.set_title(age, fontsize=14)
		fig1.subplots_adjust(top=0.85)
		ax2=fig1.add_subplot(111)
		ax3=fig1.add_subplot(111)
		ax4=fig1.add_subplot(111)
		ax5=fig1.add_subplot(111)
		ax6=fig1.add_subplot(111)
		ax7=fig1.add_subplot(111)
		ax81=fig1.add_subplot(111)
		ax91=fig1.add_subplot(111)
		ax.set_xlabel('Time (hours)')
		ax.set_ylabel('Lever Presses')
		ax.plot(extinction/60/60, countse+lengthm1+lengthm2+lengthprg, 'ko') #extinction
		ax2.plot(maint1/60/60, countsm1+lengthprg,'yo', markersize=5) #maint1
		ax3.plot(maint2/60/60, countsm2+lengthm1+lengthprg, 'bo') #maint2
		ax4.plot(outputm11/60/60, fit1[0]*outputm11+lengthprg+fit1[1], 'r', linewidth=2.5)
		ax5.plot(outputm12/60/60, fit2[0]*outputm12+lengthprg+(fit2[1]+turning_point), 'b', linewidth=2.5)

		ax7.plot(prg/60/60, countsprg, 'wo', markersize=4) #priming programmed
		ax81.plot(ld/60/60, countsld, 'ko', markersize=4) #1500 loading injections
		if k1:
			ax91.axvline(x=k1/60/60, linestyle='--')
		fig1.subplots_adjust(left=0.25, bottom=0.25)
		min0x=0
		max0x=8
		axcolor = 'lightgoldenrodyellow'
		
		fig2=plt.figure(111)

		ax8=fig2.add_subplot(111)
		ax9=fig2.add_subplot(111)
		ax10=fig2.add_subplot(111)
		ax11=fig2.add_subplot(111)
		ax12=fig2.add_subplot(111)
		ax13=fig2.add_subplot(111)
		ax14=fig2.add_subplot(111)
		ax15=fig2.add_subplot(111)
		ax16=fig2.add_subplot(111)
		ax17=fig2.add_subplot(111)
		ax18=fig2.add_subplot(111)
		ax8.set_xlabel('Time (hours)')
		ax8.set_ylabel('Concentration (nmol/kg)')
		ax8.plot(outputm11/60/60,maint1c, 'ro')
		ax9.plot(outputm12/60/60, maint2counts, 'yo')
		ax10.plot(extinction/60/60, extc, 'ko')
		ax11.plot(prg/60/60, prgc, 'wo')
		ax12.plot(outputm11/60/60, fit1c[0]*outputm11+lengthprg+fit1c[1], 'r', linewidth=2.5)
		ax13.plot(outputm12/60/60, fit2c[0]*outputm12+lengthprg+(fit2c[1]+turning_point), 'b', linewidth=2.5)
		ax16.plot(maint2/60/60, maint2co, 'bo')
		try:
			ax6.plot(maint2/60/60,fit3[0]*maint2+lengthm1+lengthprg+fit3[1], linewidth=2.5)
			ax15.plot(maint2/60/60,fit3c[0]*maint2+lengthm1+lengthprg+fit3c[1], linewidth=2.5)
		except(UnboundLocalError):
			print ('no second maintenence')
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		try:
			ax.text(0.05, 0.95, 'loading= %i+/- %i \n m1 interval= %i +/- %i \n m2 interval= %i +/- %i \n PT=%i'%(avgldint, loadsem, avgm1int, m1sem, avgm2int, m2sem, priming_threshold),
				transform=ax.transAxes, fontsize=12,
			    verticalalignment='top', bbox=props) 
		except(TypeError):
			print ('error')
		ax17.plot(ld/60/60, ldc, 'ko')
		ax18.plot(k1/60/60, k1, marker="x")

		plt.show()

	else:
		loadint=np.array([111])
		m1int=np.array([0])
		m2int=np.array([0])
		extint=np.array([0])
		maint1c=np.array([0])
		maint2co=np.array([0])
		maint2counts=np.array([0])
		extc=np.array([0])
		avglist=[0]
		semlist=[0]
		nlist=[0]

		fig1=plt.figure()
		ax1=fig1.add_subplot(111)
		ax1.plot(maint1, maint1c, 'bo')
		fig2=plt.figure()
		ax2=fig2.add_subplot(111)
		ax2.plot(maint1, countsm1, 'bo')
		ax3=fig2.add_subplot(111)

		plt.show()
def reanalysis():
	#all global declarations are to make the variable available outside of the analysi funciton
	global turning_point
	global array
	global arrayc
	global arraye
	global arraya
	global maint1c
	global maint2counts
	global extint
	global prgc
	global nlist
	global priming_threshold
	global dose1
	global dose2
	global maint1
	global fig1
	global fig2
	global loadint
	global resultsm1
	global resultsm2
	global lengthm1
	global loadintlist
	global m1int
	global m2int
	global lengthprg
	global countsprg
	global prg
	global avglist
	global avglistlabel
	global semlist
	global resultsload
	global extint
	global avglistlabel
	global semlist
	global resultsload
	global lintlist
	global fig1
	global fig2
	global maint2co
	global extc
	global age
	data_file = open(e1.get())
	blockO = ''
	found = False
	for line in data_file:
			if found:
				blockO += line
				if line.strip() == "R:": break
			else:
				if line.strip() == "O:":
					found = True
	sbto=blockO.split(':')
	newlisto=[item.split() for item in ' '.join(sbto).split("'") if item]
	with open('someo.csv', 'w') as f:
		writer=csv.writer(f)
		writer.writerows(newlisto)
	arrayo=np.genfromtxt('someo.csv', delimiter=',')

	blockU = ''
	found = False
	for line in data_file:
			if found:
				blockU += line
				if line.strip() == "V:": break
			else:
				if line.strip() == "U:":
					found = True
	sbtu=blockU.split(':')
	newlistu=[item.split() for item in ' '.join(sbtu).split("'") if item]
	with open('someu.csv', 'w') as f:
		writer=csv.writer(f)
		writer.writerows(newlistu)
	arrayu=np.genfromtxt('someu.csv', delimiter=',')

	data_file.close()

	penultimate_priming_dose=arrayu[4]
	ultimate_priming_dose=arrayu[5]
	priming_threshold=(penultimate_priming_dose+ultimate_priming_dose)/2
	dose1=arrayo[1]
	dose2=arrayo[2]

	try:
		data_file = open(e1.get()) #import data
	except:
		tkMessageBox.showwarning('File Error', "Enter a valid file name. \n File must be located in the same folder as the script.")

	blockA = "" #create empty list
	found = False
	for line in data_file: #read through the file, finding letters and breaking at them
	    if found:
	        blockA += line
	        if line.strip() == "B:": break #ending letter
	    else:
	        if line.strip() == "A:": #starting letter
	            found = True
	            blockA = "A:"
	sbta=blockA.split(':') #split the string
	newlista=[item.split() for item in ' '.join(sbta).split("'") if item] #write string into a list
	with open('somea.csv', 'w') as f:
	    writer=csv.writer(f)
	    writer.writerows(newlista) #write into a dummy CSV file for formatting
	arraya=np.genfromtxt('somea.csv', delimiter=',') #create numpy array of data from the dummy .csv
	arraya = arraya[np.logical_not(np.isnan(arraya))]            
	blockC = ""
	found = False
	for line in data_file:
	    if found:
	        blockC += line
	        if line.strip() == "D:": break
	    else:
	        if line.strip() == "C:":
	            found = True
	sbtc=blockC.split(':')
	newlistc=[item.split() for item in ' '.join(sbtc).split("'") if item]
	with open('somec.csv', 'w') as f:
	    writer=csv.writer(f)
	    writer.writerows(newlistc)
	arrayc=np.genfromtxt('somec.csv', delimiter=',')

	blockE = ""
	found = False
	for line in data_file:
	    if found:
	        blockE += line
	        if line.strip() == "G:": break
	    else:
	        if line.strip() == "E:":
	            found = True
	sbte=blockE.split(':')
	newliste=[item.split() for item in ' '.join(sbte).split("'") if item]
	with open('somee.csv', 'w') as f:
	    writer=csv.writer(f)
	    writer.writerows(newliste)
	arraye=np.genfromtxt('somee.csv', delimiter=',')

	blockT = ''
	found = False
	for line in data_file:
			if found:
				blockT += line
				if line.strip() == "U:": break
			else:
				if line.strip() == "T:":
					found = True
	sbt=blockT.split(':')
	newlist=[item.split() for item in ' '.join(sbt).split("'") if item]
	with open('some.csv', 'w') as f:
		writer=csv.writer(f)
		writer.writerows(newlist)
	array=np.genfromtxt('some.csv', delimiter=',')

	data_file.close()
	
	var1output=var1.get()

	array = array[np.logical_not(np.isnan(array))]
	arrayc = arrayc[np.logical_not(np.isnan(arrayc))]
	arraye = arraye[np.logical_not(np.isnan(arraye))]
	#deletes the line labels
	array = np.delete(array, np.arange(0, array.size, 11))
	arrayc = np.delete(arrayc, np.arange(0, arrayc.size, 11))
	arraye=np.delete(arraye, np.arange(0, arraye.size, 11))
	var1output=var2.get()
	if var1output==0:
		age=arraya[6]
	else:
		age=e2.get()

	#reads the event codes to only take the inejctions and relevant events
	k1indices=np.where(arraye==64400115)
	k1=np.take(array,k1indices)
	k1=np.array(k1)
	lengthk1=k1.size
	countsk1=list(range(lengthk1))
	countsk1=np.array(countsk1)
	k1=k1.reshape(lengthk1,)
	k1c=np.take(arrayc, k1indices)
	k1c=k1c.reshape(lengthk1,)

	k9indices=np.where(arraye==63400009)
	k9=np.take(array,k9indices)
	k9=np.array(k9)
	lengthk9=k9.size
	countsk9=list(range(lengthk9))
	countsk9=np.array(countsk9)
	k9=k9.reshape(lengthk9,)
	k9c=np.take(arrayc, k9indices)
	k9c=k9c.reshape(lengthk9,)

	ldindices=np.where(arraye==42100111)
	ld=np.take(array,ldindices)
	ld=np.array(ld)
	lengthld=ld.size
	countsld=list(range(lengthld))
	countsld=np.array(countsld)
	ld=ld.reshape(lengthld,)
	ldc=np.take(arrayc, ldindices)
	ldc=ldc.reshape(lengthld,)

	prgmindices=np.where(arraye==31100200)
	prg=np.take(array,prgmindices)
	prg=np.array(prg)
	lengthprg=prg.size
	countsprg=list(range(lengthprg))
	countsprg=np.array(countsprg)
	prg=prg.reshape(lengthprg,)
	prgc=np.take(arrayc, prgmindices)
	prgc=prgc.reshape(lengthprg,)

	maint1indices=[np.where(np.logical_or(arraye==64400111, arraye==62100111))]
	maint1c=np.take(arrayc, maint1indices)
	maint1=np.take(array,maint1indices)
	maint1=np.array(maint1)
	lengthm1=maint1.size
	countsm1=list(range(lengthm1))
	countsm1=np.array(countsm1)
	maint1=maint1.reshape(lengthm1,)
	maint1c=maint1c.reshape(lengthm1,)

	maint2indices=np.where(arraye==74400111)
	maint2=np.take(array,maint2indices)
	maint2=np.array(maint2)
	lengthm2=maint2.size
	countsm2=list(range(lengthm2))
	countsm2=np.array(countsm2)
	maint2=maint2.reshape(lengthm2,)
	maint2co=np.take(arrayc, maint2indices)
	maint2co=maint2co.reshape(lengthm2,)


	extindices=[np.where((arraye>89999999) &(arraye<98000000))]
	extinction=np.take(array,extindices)
	extinction=np.array(extinction)
	lengthe=extinction.size
	countse=list(range(lengthe))
	countse=np.array(countse)
	extinction=extinction.reshape(lengthe, )
	extc=np.take(arrayc, extindices)
	extc=extc.reshape(lengthe, )
	global totaltime
	global totalconc
	totaltime=np.concatenate((ld, prg, maint1, maint2, extinction), axis=0)
	totalconc=np.concatenate((ldc, prgc, maint1c, maint2co, extc), axis=0)




	if len(k9>0):
		k9idx = (np.abs(maint1 - k9[-1])).argmin()
		m1take= list(range(0,k9idx))
		m2take=list(range(k9idx,len(maint1)))
		maint1s=maint1
		maint1cs=maint1c
		maint1=np.take(maint1s,m1take)
		maint2=np.take(maint1s,m2take)
		maint1c=np.take(maint1cs, m1take)
		maint2co=np.take(maint1cs, m2take)

	lengthm1=maint1.size
	countsm1=list(range(lengthm1))
	countsm1=np.array(countsm1)
	maint1=maint1.reshape(lengthm1,)
	maint1c=maint1c.reshape(lengthm1,)

	lengthm2=maint2.size
	countsm2=list(range(lengthm2))
	countsm2=np.array(countsm2)
	maint2=maint2.reshape(lengthm2,)
	maint2co=maint2co.reshape(lengthm2,)

	xm1=maint1
	ym1=countsm1
	

	#takes the second deriviative, and splits the list at that point
	breakpoint=v1.get()
	breakpoint=breakpoint-len(prgc)
	outputm11 = []
	outputm12=[]
	each_part = []
	breakpoint=[breakpoint]

	if len(xm1)>1:
		
		for i in range(len(maint1)):
			if i not in breakpoint:
				each_part.append(maint1[i]) #put each element before each index to each_part
			else:
				each_part.append(maint1[i]) #put element that on the index to each_part
				outputm11.append(each_part) #put the each_part to the output
				each_part = [] #clear each_part

		if breakpoint[-1] < len(maint1):
			last_part =  maint1[breakpoint[-1]+1:] #put the element after last index to last part
			outputm12.append(last_part) #put the element after last index to output 

		breakpoint=np.array(breakpoint)
		breakpoint=np.asscalar(breakpoint)

		outputm11=np.asarray(outputm11)
		outputm12=np.asarray(outputm12)
		outputm11=outputm11.reshape(breakpoint+1, )
		outputm12=outputm12.reshape(lengthm1-(breakpoint+1), )

		countsm11=list(range(0,breakpoint+1))
		countsm12=list(range(1,((lengthm1-breakpoint))))
		maint2counts=maint1c[len(outputm11):len(maint1c)]
		maint1c=maint1c[0:len(outputm11)]

		#linear regressions
		fit1=scipy.stats.linregress(outputm11,countsm11)
		fit2=scipy.stats.linregress(outputm12, countsm12)
		fit1c=scipy.stats.linregress(outputm11, maint1c)
		fit2c=scipy.stats.linregress(outputm12, maint2counts)

		fit3=stats.linregress(maint2+lengthm1,countsm2)
		fit3c=stats.linregress(maint2+lengthm1,maint2co)


#calculating the means, sems, etc
		loadint=np.diff(outputm11)
		avgldint=np.mean(loadint)
		loadsem=scipy.stats.sem(loadint)
		m1int=np.diff(outputm12)
		avgm1int=np.mean(m1int)
		m2int=np.diff(maint2)
		avgm2int=np.mean(m2int)
		extint=np.diff(extinction)
		avgextint=np.mean(extint)
		sdm1=scipy.stats.tstd(maint1)
		sdm2=scipy.stats.tstd(maint2)
		sdext=scipy.stats.tstd(extinction)
		m1sem=scipy.stats.sem(m1int)
		m2sem=scipy.stats.sem(m2int)
		extsem=scipy.stats.sem(extinction)


		avglist=[avgldint, avgm1int, avgm2int, avgextint]
		semlist=[loadsem, m1sem, m2sem, extsem]
		nlist=[breakpoint, (lengthm1-breakpoint), lengthm2, lengthe]
		loadintlist=loadint.tostring


		#these are all used to print out the results later
		resultsload=[avgldint, loadsem, len(outputm11)]
		resultsm1=[avgm1int, m1sem, len(outputm12)]
		resultsm2=[avgm2int, m2sem, len(maint2)]

	
	# the rest is making the graph. 
		fig1=plt.figure()
		fig1.suptitle(arraya[4], fontsize=14, fontweight='bold')
		ax=fig1.add_subplot(111)
		ax.set_title(age, fontsize=14)
		fig1.subplots_adjust(top=0.85)
		ax2=fig1.add_subplot(111)
		ax3=fig1.add_subplot(111)
		ax4=fig1.add_subplot(111)
		ax5=fig1.add_subplot(111)
		ax6=fig1.add_subplot(111)
		ax7=fig1.add_subplot(111)
		ax81=fig1.add_subplot(111)
		ax91=fig1.add_subplot(111)
		ax.set_xlabel('Time (hours)')
		ax.set_ylabel('Lever Presses')
		ax.plot(extinction/60/60, countse+lengthm1+lengthm2+lengthprg, 'ko') #extinction
		ax2.plot(maint1/60/60, countsm1+lengthprg,'yo', markersize=5) #maint1
		ax3.plot(maint2/60/60, countsm2+lengthm1+lengthprg, 'bo') #maint2
		ax4.plot(outputm11/60/60, fit1[0]*outputm11+lengthprg+fit1[1], 'r', linewidth=2.5)
		ax5.plot(outputm12/60/60, fit2[0]*outputm12+lengthprg+(fit2[1]+breakpoint), 'b', linewidth=2.5)

		ax7.plot(prg/60/60, countsprg, 'wo', markersize=4) #priming programmed
		ax81.plot(ld/60/60, countsld, 'ko', markersize=4) #1500 loading injections
		if k1:
			ax91.axvline(x=k1/60/60, linestyle='--')
		fig1.subplots_adjust(left=0.25, bottom=0.25)
		min0x=0
		max0x=8
		axcolor = 'lightgoldenrodyellow'
		
		fig2=plt.figure(111)

		ax8=fig2.add_subplot(111)
		ax9=fig2.add_subplot(111)
		ax10=fig2.add_subplot(111)
		ax11=fig2.add_subplot(111)
		ax12=fig2.add_subplot(111)
		ax13=fig2.add_subplot(111)
		ax14=fig2.add_subplot(111)
		ax15=fig2.add_subplot(111)
		ax16=fig2.add_subplot(111)
		ax17=fig2.add_subplot(111)
		ax18=fig2.add_subplot(111)
		ax8.set_xlabel('Time (hours)')
		ax8.set_ylabel('Concentration (nmol/kg)')
		ax8.plot(outputm11/60/60,maint1c, 'ro')
		ax9.plot(outputm12/60/60, maint2counts, 'yo')
		ax10.plot(extinction/60/60, extc, 'ko')
		ax11.plot(prg/60/60, prgc, 'wo')
		ax12.plot(outputm11/60/60, fit1c[0]*outputm11+lengthprg+fit1c[1], 'r', linewidth=2.5)
		ax13.plot(outputm12/60/60, fit2c[0]*outputm12+lengthprg+(fit2c[1]+breakpoint), 'b', linewidth=2.5)
		ax16.plot(maint2/60/60, maint2co, 'bo')
		try:
			ax6.plot(maint2/60/60,fit3[0]*maint2+lengthm1+lengthprg+fit3[1], linewidth=2.5)
			ax15.plot(maint2/60/60,fit3c[0]*maint2+lengthm1+lengthprg+fit3c[1], linewidth=2.5)
		except(UnboundLocalError):
			print ('no second maintenence')
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		try:
			ax.text(0.05, 0.95, 'loading= %i+/- %i \n m1 interval= %i +/- %i \n m2 interval= %i +/- %i \n PT=%i'%(avgldint, loadsem, avgm1int, m1sem, avgm2int, m2sem, priming_threshold),
				transform=ax.transAxes, fontsize=12,
			    verticalalignment='top', bbox=props) 
		except(TypeError):
			print ('error')
		ax17.plot(ld/60/60, ldc, 'ko')
		ax18.plot(k1/60/60, k1, marker="x")

		plt.show()

	else:
		loadint=np.array([111])
		m1int=np.array([0])
		m2int=np.array([0])
		extint=np.array([0])
		maint1c=np.array([0])
		maint2co=np.array([0])
		maint2counts=np.array([0])
		extc=np.array([0])
		avglist=[0]
		semlist=[0]
		nlist=[0]

		fig1=plt.figure()
		ax1=fig1.add_subplot(111)
		ax1.plot(maint1, maint1c, 'bo')
		fig2=plt.figure()
		ax2=fig2.add_subplot(111)
		ax2.plot(maint1, countsm1, 'bo')
		ax3=fig2.add_subplot(111)

		plt.show()
def importU():
	global priming_threshold
	global dose1
	global dose2
	data_file = open(e1.get())
	blockO = ''
	found = False
	for line in data_file:
			if found:
				blockO += line
				if line.strip() == "R:": break
			else:
				if line.strip() == "O:":
					found = True
	sbto=blockO.split(':')
	newlisto=[item.split() for item in ' '.join(sbto).split("'") if item]
	with open('someo.csv', 'wb') as f:
		writer=csv.writer(f)
		writer.writerows(newlisto)
	arrayo=np.genfromtxt('someo.csv', delimiter=',')

	blockU = ''
	found = False
	for line in data_file:
			if found:
				blockU += line
				if line.strip() == "V:": break
			else:
				if line.strip() == "U:":
					found = True
	sbtu=blockU.split(':')
	newlistu=[item.split() for item in ' '.join(sbtu).split("'") if item]
	with open('someu.csv', 'wb') as f:
		writer=csv.writer(f)
		writer.writerows(newlistu)
	arrayu=np.genfromtxt('someu.csv', delimiter=',')
	

	
	data_file.close()
	
	penultimate_priming_dose=arrayu[4]
	ultimate_priming_dose=arrayu[5]
	priming_threshold=(penultimate_priming_dose+ultimate_priming_dose)/2
	dose1=arrayo[1]
	dose2=arrayo[2]

	print (arrayo)
	print (dose1)
	print (dose2)

master = Tk()
master.configure(background='pink')
master.configure(highlightbackground='magenta')

Label(master, text="file name").grid(row=0)
Label(master, text="age").grid(row=2)
Label(master, text="experiment").grid(row=4)
Label(master, text ="test drug").grid(row=6)
Label(master, text = "keyword").grid(row=7)

master.wm_title('Self-administration Analysis')
e1 = Entry(master)
e2= Entry(master)
e3= Entry(master)
e4=Entry(master)
e5=Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=2, column=1)
e3.grid(row=4, column=1)
e4.grid(row=6, column=1)
e5.grid(row=7, column=1)




def write():
	lintlist=loadint.tolist()
	m1list=m1int.tolist()
	m2list=m2int.tolist()
	extintlist=extint.tolist()
	m1clist=maint1c.tolist()
	m2clist=maint2co.tolist()
	# m3clist=maint2c.tolist()
	extclist=extc.tolist()

	ptexc=var3.get()
	m1exc=var4.get()
	m2exc=var5.get()
	extexc=var6.get()
	if ptexc==0:
		ptexclude='no'
	else:
		ptexclude='yes'
	
	if m1exc==0:
		m1exclude='no'
	else:
		m1exclude='yes'
	
	if m2exc==0:
		m2exclude='no'
	else:
		m2exclude='yes'
	
	if extexc==0:
		extexclude='no'
	else:
		extexclude='yes'


	expcode = e3.get()
	testdrug=e4.get()
	keywords=e5.get()
	geninfolabel=['rat#', 'day', 'priming threshold', 'experiment', 'test drug', 'keyword', 'exclude priming', 'exclude m1', 
	'exclude m2', 'exclude extinction']
	geninfo=[arraya[4], arraya[6], priming_threshold, expcode, testdrug, keywords, ptexclude,
	m1exclude, m2exclude, extexclude]
	avglistlabel=["loading ints", 'm1 ints', 'm2 ints', 'extinction ints']
	doselist=[dose1, dose2]
	header=['loading int','maint1 int', 'maint2 int', 'ext int', 'loading conc', 'maint1 conc', 
		'maint2 conc', 'ext conc',' ', ' ', ' ', 'interval mean', 'sem', 'n', 'dose', 'total time', 'concentration']

	filename='rat%sd%s.csv'%(arraya[4], age)
	from itertools import zip_longest
	d=[lintlist, m1list, m2list, extintlist, m1clist, m2clist, 
		m2clist, extclist, geninfolabel,geninfo,avglistlabel, avglist,
		semlist,nlist, doselist, totaltime, totalconc]
	export_data=zip_longest(*d, fillvalue=' ')
	with open(filename, 'w',encoding="ISO-8859-1", newline='') as f:
		writer=csv.writer(f)
		writer.writerow(header)
		writer.writerows(export_data)
		
	pltfilename='rat%sd%s.pdf'%(arraya[4], age)
	with PdfPages(pltfilename) as pdf:
		pdf.savefig(fig1)
		pdf.savefig(fig2)

	var2.set(0)
	var3.set(0)
	var4.set(0)
	var5.set(0)
	var6.set(0)	
# the buttons...
var1=IntVar()
var2=IntVar()
var3=IntVar()
var4=IntVar()
var5=IntVar()
var6=IntVar()


c2=Checkbutton(master, text='Age needed', variable=var2).grid(row=10, column= 4,sticky=W)
c3=Checkbutton(master, text='exclude priming', variable=var3).grid(row=10, sticky=W)
c4=Checkbutton(master, text='exclude m1', variable=var4).grid(row=11, sticky=W)
c5=Checkbutton(master, text='exclude m2', variable=var5).grid(row=12, sticky=W)
c6=Checkbutton(master, text='exclude extinction', variable=var6).grid(row=13, sticky=W)

Button(master, text='Run', command=analysis).grid(row=5, column=1, sticky=W, pady=4)
Button(master, text='Quit', command=master.quit).grid(row=5, column=2, sticky=W, pady=4)
Button(master, text='Save', command=write).grid(row=7, column=3, sticky=W, pady=4)
Button(master, text='Training', command=training).grid(row=13, column=4, sticky=W, pady=4)
e1.grid(row=0, column=1)

v1=IntVar()
e2= Entry(master, textvariable=v1)
e2.grid(row=1, column=1)
# e2.insert(END,turning_point)
Label(master, text='Where does loading end?').grid(row=1)
Button(master, text='Rerun', command=reanalysis).grid(row=5, column=4, sticky=W, pady=4)

mainloop( )
