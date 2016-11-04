import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

data = np.loadtxt('particleCord.txt')
data1 = np.loadtxt('myrobotCord.txt')

f1 = plt.figure(1)
#f1.subplots_adjust(left=0.05, right=0.95)
f1.subplots_adjust(left=0.12, right=0.93, top=0.92, bottom=0.07, hspace=0.3, wspace=0.2)
f1.patch.set_facecolor('white')
#f1.set_figheight(30)
f1.set_figheight(20)
f1.set_figwidth(20)

ALL_FONT_SIZES = 55

st = f1.suptitle("Workspace = 100x100; NUM_PARTICLES = 10000", fontsize=ALL_FONT_SIZES,color = 'black')
#st = f1.suptitle("Robot in a workspace of size 100x100", fontsize=ALL_FONT_SIZES,color = 'black')
#st.set_verticalalignment('center')
#st.set_y(0.95)

#for i in range(6):
i=0
ax = plt.subplot(1,1,1, axisbg='white')
#ax = plt.subplot(3,2,i+1, axisbg='white')

#plt.plot(data[ i*10000 : (i+1)*10000-1 , 0 ], data[  i*10000 : (i+1)*10000-1 ,1], 'cD', markersize=5)
#plt.plot(data1[i,0], data1[i,1],'rs', markersize=20)

vect = data[ i*10000 : (i+1)*10000-1 , 3 ]

plt.scatter(data[ i*10000 : (i+1)*10000-1 , 0 ], data[  i*10000 : (i+1)*10000-1 ,1], s=150, c=vect*10, marker= 'o', linewidths=1,rasterized=True)
#plt.colorbar()
plt.plot(data1[i,0], data1[i,1], marker ='D', color = ('#7cfc00'), markersize=50)

#plt.title("Iteration: %i" % (i+1), color = 'black', fontsize=ALL_FONT_SIZES, fontweight='bold')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')

ax.tick_params(axis='both', colors='black', labelsize=ALL_FONT_SIZES)

ax.xaxis.grid(color='gray', linestyle='dashed', linewidth=1.0)
ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=1.0)

plt.xlabel('X-location', fontsize=ALL_FONT_SIZES, verticalalignment='top')
plt.ylabel('Y-location', fontsize=ALL_FONT_SIZES)

plt.xlim(0.0, 100.0)
plt.ylim(0.0, 100.0)

plt.savefig("robot_with_particles.eps", facecolor='white', format='eps', dpi=50)	
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
