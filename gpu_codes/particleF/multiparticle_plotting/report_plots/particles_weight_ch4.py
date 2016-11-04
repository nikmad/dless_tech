import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
 
data = np.loadtxt('particleCord.txt')
data1 = np.loadtxt('myrobotCord.txt')

f1 = plt.figure(1)
#f1.subplots_adjust(left=0.05, right=0.95)
f1.subplots_adjust(left=0.08, right=1.0, hspace=0.3, wspace=0.2, top=0.95, bottom=0.05)
f1.patch.set_facecolor('white')
#f1.set_figheight(30)
f1.set_figheight(20)
f1.set_figwidth(20)

ALL_FONT_SIZES = 35

#st = f1.suptitle("Workspace = 100x100; NUM_PARTICLES = 10000", fontsize=ALL_FONT_SIZES,color = 'black')
#st = f1.suptitle("Robot in a workspace of size 100x100", fontsize=55,color = 'black')
#st.set_verticalalignment('center')
#st.set_y(0.95)

#for i in range(6):
i=3
ax = plt.subplot(1,1,1, axisbg='white')
#ax = plt.subplot(3,2,i+1, axisbg='white')

#plt.plot(data[ i*10000 : (i+1)*10000-1 , 0 ], data[  i*10000 : (i+1)*10000-1 ,3], 'cD', markersize=5)
#plt.plot(data1[i,0], 0,'rs', markersize=50)

vect = data[ i*10000 : (i+1)*10000-1 , 3 ]

plt.scatter(data[ i*10000 : (i+1)*10000-1 , 0 ], data[  i*10000 : (i+1)*10000-1 ,3], s=150, c=vect*10, marker= 'o', linewidths=1,rasterized=True)
cbar = plt.colorbar()
cbar.set_label('Particle Weight', rotation=270, fontsize=ALL_FONT_SIZES)
plt.plot(data1[i,0], 0.0000005, marker ='s', color = ('#7cfc00'), markersize=30)


#plt.xcorr(data[ i*10000 : (i+1)*10000-1 , 0 ], data[  i*10000 : (i+1)*10000-1 ,1], usevlines=True, maxlags=100, lw=2)
#plt.grid(True)
#plt.axhline(0, color='black', lw=2)

plt.title("Iteration: %i" % (i+1), color = 'black', fontsize=ALL_FONT_SIZES, fontweight='bold')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')

formatter = ScalarFormatter(useMathText=True) 
formatter.set_scientific(True) 
formatter.set_powerlimits((-2,4)) 
ax.tick_params(axis='both', colors='black', labelsize=ALL_FONT_SIZES)
ax.xaxis.set_major_formatter(formatter) 
ax.yaxis.set_major_formatter(formatter) 

ax.xaxis.grid(color='gray', linestyle='dashed', linewidth=1.0)
ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=1.0)

plt.xlabel('X-location', fontsize=ALL_FONT_SIZES, verticalalignment='top')
plt.ylabel('Particle Weight', fontsize=ALL_FONT_SIZES)

plt.xlim(0.0, 100.0)
plt.ylim(0.0, 0.0005)

#plt.savefig("particles_weight.pdf", facecolor='white', format='pdf', dpi=50)	
plt.savefig("particles_weight.eps", facecolor='white', format='eps', dpi=50)	
#plt.show(1)

#f2 = plt.figure(2)
#f2.subplots_adjust(left=0.05, right=0.95)
#f2.patch.set_facecolor('black')
#f2.set_figheight(17)
#f2.set_figwidth(30)
#
#st2 = f2.suptitle("World Size = 100; N = 10000; Motion = (0.1rad, 1unit)", fontsize=14,fontweight='bold',color = 'white')
#st2.set_verticalalignment('center')
#st2.set_y(0.95)
#
#for i in range(20):
#
#	ax = plt.subplot(4,5,i+1, axisbg='white')
#	
#	plt.plot(data[ (i+20)*10000 : (i+21)*10000-1 , 0 ], data[  (i+20)*10000 : (i+21)*10000-1 ,1], 'cD', markersize=4)
#	plt.plot(data1[i+20,0], data1[i+20,1],'rs', markersize=8)
#
#	plt.title("Iteration: %i" % (i+21), color = 'white', fontsize=11, fontweight='bold')
#
#	ax.spines['bottom'].set_color('black')
#	ax.spines['top'].set_color('black')
#	ax.spines['left'].set_color('black')
#	ax.spines['right'].set_color('black')
#	
#	ax.xaxis.label.set_color('white')
#	ax.yaxis.label.set_color('white')
#	
#	ax.tick_params(axis='both', colors='white', labelsize='large')
#
#	ax.xaxis.grid(color='gray', linestyle='dashed', linewidth=1.0)
#	ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=1.0)
#
#	plt.xlabel('X-axis', fontsize=11, fontweight='bold')
#	plt.ylabel('Y-axis', fontsize=11, fontweight='bold')
#
#	plt.xlim(0.0, 100.0)
#	plt.ylim(0.0, 100.0)
#
#plt.savefig("out2.png", facecolor='black', dpi=100)	
#plt.show(2)
#
