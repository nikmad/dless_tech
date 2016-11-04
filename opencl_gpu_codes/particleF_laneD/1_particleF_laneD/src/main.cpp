#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "RNG.h"
#include "robot.h"

using namespace Eigen;
using namespace std;

#define CV_LOAD_IMAGE_GRAYSCALE 0
#define CV_LOAD_IMAGE_COLOR 1
#define CV_WINDOW_NORMAL 0

float forwardNoise = 0.05;
float turnNoise = 0.05;
float senseNoise = 0.5;

float TURNMAX = 0.2;
float DISTMAX = 2.0;


typedef unsigned long int ULONG;

int main( int argc, char** argv )
{
	system("rm goodCord.txt");
	system("rm bestCord.txt");
	system("rm lines.txt");
	
   int BEST_LINE_DIST = atoi(argv[1]);
	int NUM_BEST_LINES = atoi(argv[2]);
	cv::Mat img_org = cv::imread( argv[3], 
				CV_LOAD_IMAGE_COLOR);

	cv::Mat img_gray;
	img_gray = cv::imread( argv[3], 
				CV_LOAD_IMAGE_GRAYSCALE);
	 
	 cv::Mat img_grad(img_gray.rows, img_gray.cols-1, CV_8UC1); //gradient matrix
	 
	int NUM_LINES = 30000;
	int NUM_GOOD_LINES = 2000;
	float GOOD_LINE_DIST = img_grad.cols/(2*NUM_GOOD_LINES);

	
	for(int row=0; row < img_grad.rows; row++) {
	for(int col=0; col < img_grad.cols; col++) {
	img_grad.at<uchar>(row, col) =  (uchar)abs(
					(int)img_gray.at<uchar>(row, col+1) - 
			 	     	(int)img_gray.at<uchar>(row, col));
	}
	}
	
	double maxVal=0; 
	double minVal=0;
	cv::minMaxLoc(img_grad, &minVal, &maxVal, 0, 0);

	cout << "MaxValue: " << maxVal << "\n" << endl;
	
	//any gradient less than 30% of maxvalue is discarded. 
	for(int row=0; row < img_grad.rows; row++) {
	for(int col=0; col < img_grad.cols; col++) {
	if((int)img_grad.at<uchar>(row, col) < 0.3*maxVal) 
	img_grad.at<uchar>(row, col) = 0;
	else
	img_grad.at<uchar>(row, col) = 1;
	}
	}

	cout << img_grad.rows << endl;
	cout << img_grad.cols << endl;
	
linestruct *line;
line = new linestruct [NUM_LINES];
float r;
int NEIGHBORHOOD = 5;
int c2 = 0;
float xadd = 0.0;
float bgauss = 0.0;

for(int ln=0; ln<NUM_LINES; ln++)
{
	line[ln].x = new float [img_grad.rows]; // Intuition says it is .cols but we use .rows cuz we need as many x as we have y for the loop below.
	line[ln].y = new float [img_grad.rows];
	
	line[ln].forwardNoise = forwardNoise;
	line[ln].turnNoise = turnNoise;
	line[ln].senseNoise = senseNoise;
	
	line[ln].theta = ln * (PI)/NUM_LINES;
	
	line[ln].dist = 0.0;
		
	srand(ln*10000+20*ln*ln*ln);
	c2 = rand()%10000;
	xadd = (float)(((float)c2/10000.0000) * img_grad.cols);

	for(int l=0; l<img_grad.rows; l++)
	{
		line[ln].y[l] = l; 
		r = line[ln].y[l]/sin(line[ln].theta);	
		
		line[ln].x[l] = abs((int)(xadd + r * (cos(line[ln].theta))));

			RNG rng(time(NULL));
			bgauss = (rng.gaussian(line[ln].senseNoise));	
	       		if(line[ln].x[l]>=NEIGHBORHOOD && line[ln].x[l] <= img_grad.cols-NEIGHBORHOOD){  
	       			for(int m=line[ln].x[l]-NEIGHBORHOOD; m<line[ln].x[l]+NEIGHBORHOOD; m++)
	       			{
	       				line[ln].dist += img_grad.at<uchar>(l,m);
				}
				//line[ln].dist += bgauss;
			}
	       		else if(line[ln].x[l] > img_grad.cols-NEIGHBORHOOD){  
				for(int m=line[ln].x[l]-NEIGHBORHOOD; m<img_grad.cols; m++)
				{
					line[ln].dist += img_grad.at<uchar>(l,m);
				}
				//line[ln].dist += bgauss;
			}
			else if(line[ln].x[l]<NEIGHBORHOOD){
				for(int m=0; m<line[ln].x[l]+NEIGHBORHOOD; m++)
				{
					line[ln].dist += img_grad.at<uchar>(l,m);
				}
				//line[ln].dist += bgauss;
			}
	}
}


FILE* file = fopen("lines.txt", "a"); 
char output[255]; 
for (int i=0; i<NUM_LINES; i++)
{
 	sprintf(output, "%7.3f   %10.9f\n", line[i].dist, line[i].theta); // fill the buffer with some information
   fputs(output, file); 
}
fclose(file); 


int best_line_idx[NUM_BEST_LINES];
int good_line_idx[NUM_GOOD_LINES];
int count =0;

//*************GOOD LINES*********************
float dist_vect[NUM_LINES];
for(int ln=0; ln<NUM_LINES; ln++)
{
dist_vect[ln] = line[ln].dist;
}


std::vector<float> myvector(dist_vect, dist_vect+NUM_LINES-1);
std::sort (myvector.begin(), myvector.end()); 

for( std::vector<float>::const_iterator i = myvector.end(); i != myvector.begin(); i--)
{
	if(*i != (float)0 ) 
	for(int ln=0; ln<NUM_LINES; ln++)
	{
		if(dist_vect[ln] == *i && count<NUM_GOOD_LINES)
		{
			if(count == 0){
      			good_line_idx[count] = ln;
      			count++;
			}
			else 
				if (count > 0)
				{
					for(int kl=0; kl<count; kl++){
					if(abs( line[ln].x[img_grad.rows-1] -
						line[good_line_idx[kl]].x[img_grad.rows-1] ) > 
						((float)GOOD_LINE_DIST/100.0)*img_grad.cols)
						//0.3*img_grad.cols)
						{
						if( kl<count-1 ) continue;
						else{	good_line_idx[count] = ln;
							count++;}
						}
					if(abs( line[ln].x[img_grad.rows-1] -
					line[good_line_idx[kl]].x[img_grad.rows-1] ) < 
					((float)GOOD_LINE_DIST/100.0)*img_grad.cols) break;
					//0.3*img_grad.cols) break;
					}
				}
		}
	}//END ln
}

linestruct *good_line;
good_line = new linestruct [NUM_GOOD_LINES];

for(int gl=0; gl<NUM_GOOD_LINES; gl++)
{
good_line[gl] = line[good_line_idx[gl]];
}

//************* BEST LINES*********************
float good_dist_vect[NUM_GOOD_LINES];
for(int ln=0; ln<NUM_GOOD_LINES; ln++)
{
good_dist_vect[ln] = good_line[ln].dist;
}

std::vector<float> mygoodvector(good_dist_vect, good_dist_vect+NUM_GOOD_LINES-1);
std::sort (mygoodvector.begin(), mygoodvector.end()); 

count = 0;
cout << "check 1" << endl;

for( std::vector<float>::const_iterator i = mygoodvector.end(); i != mygoodvector.begin(); i--)
{
	if(*i != (float)0 ) 
	for(int ln=0; ln<NUM_GOOD_LINES; ln++)
	{
		if(good_dist_vect[ln] == *i && count<NUM_BEST_LINES)
		{
			if(count == 0){
      			best_line_idx[count] = ln;
      			count++;
			}
			else 
				if (count > 0)
				{
					for(int kl=0; kl<count; kl++){
					if(abs( good_line[ln].x[img_grad.rows-1] -
						good_line[best_line_idx[kl]].x[img_grad.rows-1] ) > 
						((float)BEST_LINE_DIST/100.0)*img_grad.cols)
						//0.3*img_grad.cols)
						{
						if( kl<count-1 ) continue;
						else{	best_line_idx[count] = ln;
							count++;}
						}
					if(abs( good_line[ln].x[img_grad.rows-1] -
					good_line[best_line_idx[kl]].x[img_grad.rows-1] ) < 
					((float)BEST_LINE_DIST/100.0)*img_grad.cols) break;
					//0.3*img_grad.cols) break;
					}
				}
		}
	}//END ln
}

//Best line weights
cout << "Best line weights" << endl;
for(int i=0; i<NUM_BEST_LINES; i++)
{
cout << good_line[best_line_idx[i]].dist << endl; 
}

linestruct *best_line;
best_line = new linestruct [NUM_BEST_LINES];

for(int gl=0; gl<NUM_BEST_LINES; gl++)
{
best_line[gl] = good_line[best_line_idx[gl]];
}

////**************** PLOTTING ****************
////------GOOD LINES-------
//
//std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > goodlines[NUM_GOOD_LINES];
//int x1good=0, y1good=0, x2good=0, y2good=0;  
//
//for (int i=0; i<NUM_GOOD_LINES; i++)
//{
//r = sqrt(pow(img_gray.rows,2) + pow(img_gray.cols,2));
//x1good = (int) (good_line[i].x[0]);
//y1good = (int) (good_line[i].y[0]);
//x2good = (int) (r * cos(good_line[i].theta) + good_line[i].x[0]);
//y2good = (int) (r * sin(good_line[i].theta) + good_line[i].y[0]);
//
//goodlines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1good,y1good), std::pair<int, int>(x2good,y2good)));  
//
//std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it_good;  
//for(it_good = goodlines[i].begin();it_good!=goodlines[i].end();it_good++)  
//{  
//     cv::line(img_org, cv::Point(it_good->first.first, it_good->first.second), 
//	cv::Point(it_good->second.first, it_good->second.second), cv::Scalar(0,255,0), 2,8,0);  
//}
//
//} 
//
////------BEST LINES-------
//
std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > bestlines[NUM_BEST_LINES];
int x1best=0, y1best=0, x2best=0, y2best=0;  

for (int i=0; i<NUM_BEST_LINES; i++)
{
r = sqrt(pow(img_gray.rows,2) + pow(img_gray.cols,2));
x1best = (int) (best_line[i].x[0]);
y1best = (int) (best_line[i].y[0]);
x2best = (int) (r * cos(best_line[i].theta) + best_line[i].x[0]);
y2best = (int) (r * sin(best_line[i].theta) + best_line[i].y[0]);

bestlines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1best,y1best), std::pair<int, int>(x2best,y2best)));  

std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it_best;  
for(it_best=bestlines[i].begin();it_best!=bestlines[i].end();it_best++)  
{  
     cv::line(img_org, cv::Point(it_best->first.first, it_best->first.second), 
	cv::Point(it_best->second.first, it_best->second.second), cv::Scalar(0,0,255), 2, 8,0);  
}

}
	cv::namedWindow("Test Window",CV_WINDOW_NORMAL);	
	cv::imshow("Test Window", img_org);
	cv::waitKey( 0 );
//___________________________________________________________________
//
//                          PARTICLE FILTER
//___________________________________________________________________
	int iterations = 40;
	linestruct *bestlinepf;
	bestlinepf = new linestruct [NUM_BEST_LINES];
	linestruct *goodlinepf;
	goodlinepf = new linestruct [NUM_GOOD_LINES];
	//float srandSeed[NUM_GOOD_LINES];
	MatrixXf bestlineparams(NUM_BEST_LINES,3);
	MatrixXf goodlineparams(NUM_GOOD_LINES,3);
	
	//Initializing myrobot:	
	float turn1 =0.0, dist1 = 0.0;
	int c1 = 0;
	for(int bl=0; bl<NUM_BEST_LINES; bl++) //'gl' for goodline
	{	
		bestlinepf[bl] = best_line[bl]; 
		
		for(int blord=0;blord<img_grad.rows;blord++)//glord for goodline coordinate
			{
			RNG rng(time(NULL));
			if(bestlinepf[bl].x[blord]>=NEIGHBORHOOD && bestlinepf[bl].x[blord] <= img_grad.cols-NEIGHBORHOOD){  
			 	for(int m=bestlinepf[bl].x[blord]-NEIGHBORHOOD; m<bestlinepf[bl].x[blord]+NEIGHBORHOOD; m++)
			 	{
			 		bestlinepf[bl].dist += img_grad.at<uchar>(blord,m);
				//	bgauss = (rng.gaussian(goodlinepf[gl].senseNoise));	
				//	goodlinepf[gl].dist += bgauss;
			 	}
			 }
			 else if(bestlinepf[bl].x[blord] > img_grad.cols-NEIGHBORHOOD){  
			 	for(int m=bestlinepf[bl].x[blord]-NEIGHBORHOOD; m<img_grad.cols; m++)
			 	{
			 		bestlinepf[bl].dist += img_grad.at<uchar>(blord,m);
				//	bgauss = (rng.gaussian(goodlinepf[gl].senseNoise));	
				//	goodlinepf[gl].dist += bgauss;
			 	}
			 }
			 else if(bestlinepf[bl].x[blord]<NEIGHBORHOOD){
			 	for(int m=0; m<bestlinepf[bl].x[blord]+NEIGHBORHOOD; m++)
			 	{
			 		bestlinepf[bl].dist += img_grad.at<uchar>(blord,m);
				//	bgauss = (rng.gaussian(goodlinepf[gl].senseNoise));	
				//	goodlinepf[gl].dist += bgauss;
			 	}
			 }
			//measurement << sense_rob(&myrobot);
			} //end blord
	} //end bl

	for(int ln=0; ln<NUM_GOOD_LINES; ln++)
	{
		goodlinepf[ln] = good_line[ln]; 
	}

	float w_pf[NUM_GOOD_LINES];
	for(int topIndex=0; topIndex<iterations; topIndex++) 
	{
		for(int bl=0; bl<NUM_BEST_LINES; bl++) //'gl' for goodline
		{	
			move_line(&bestlinepf[bl], 0.1, 0.5, img_grad.rows, img_grad.cols);
			//SENSING: 
			for(int blord=0;blord<img_grad.rows;blord++)//glord for goodline coordinate
			{
			RNG rng(time(NULL));
			if(bestlinepf[bl].x[blord]>=NEIGHBORHOOD && bestlinepf[bl].x[blord] <= img_grad.cols-NEIGHBORHOOD){  
			 	for(int m=bestlinepf[bl].x[blord]-NEIGHBORHOOD; m<bestlinepf[bl].x[blord]+NEIGHBORHOOD; m++)
			 	{
			 		bestlinepf[bl].dist += img_grad.at<uchar>(blord,m);
				//	bgauss = (rng.gaussian(goodlinepf[gl].senseNoise));	
				//	goodlinepf[gl].dist += bgauss;
			 	}
			 }
			 else if(bestlinepf[bl].x[blord] > img_grad.cols-NEIGHBORHOOD){  
			 	for(int m=bestlinepf[bl].x[blord]-NEIGHBORHOOD; m<img_grad.cols; m++)
			 	{
			 		bestlinepf[bl].dist += img_grad.at<uchar>(blord,m);
				//	bgauss = (rng.gaussian(goodlinepf[gl].senseNoise));	
				//	goodlinepf[gl].dist += bgauss;
			 	}
			 }
			 else if(bestlinepf[bl].x[blord]<NEIGHBORHOOD){
			 	for(int m=0; m<bestlinepf[bl].x[blord]+NEIGHBORHOOD; m++)
			 	{
			 		bestlinepf[bl].dist += img_grad.at<uchar>(blord,m);
				//	bgauss = (rng.gaussian(goodlinepf[gl].senseNoise));	
				//	goodlinepf[gl].dist += bgauss;
			 	}
			 }
			} //end blord
		}//end bl
	
		float weight_pf = 0.0;
		float prob_dist = 0.0;
		int bl_points[NUM_BEST_LINES];
		for(int gl=0; gl<NUM_BEST_LINES; gl++) bl_points[gl] = 0;	
      
      for(int i=0; i<NUM_GOOD_LINES; i++)
      {
          c1 = rand()%10000;
          turn1 = ((float)c1/10000.0000)*TURNMAX;
          c2 = rand()%10000;
          dist1 = ((float)c2/10000.0000)*DISTMAX;
		    move_line(&goodlinepf[i], turn1, dist1, img_grad.rows, img_grad.cols);
		    
		    goodlinepf[i].dist = 0.0;
		    for(int l=0; l<img_grad.rows; l++)
		    {	
		    	RNG rng(time(NULL));
		    	if(goodlinepf[i].x[l]>=NEIGHBORHOOD && goodlinepf[i].x[l] <= img_grad.cols-NEIGHBORHOOD){  
		        		for(int m=goodlinepf[i].x[l]-NEIGHBORHOOD; m<goodlinepf[i].x[l]+NEIGHBORHOOD; m++)
		        		{
		        			goodlinepf[i].dist += img_grad.at<uchar>(l,m);
		    	//	bgauss = (rng.gaussian(linepf[i].senseNoise));	
		    	//	linepf[i].dist += bgauss;
		    	}
		    	}
	        			else if(goodlinepf[i].x[l] > img_grad.cols-NEIGHBORHOOD){  
		    		for(int m=goodlinepf[i].x[l]-NEIGHBORHOOD; m<img_grad.cols; m++)
		    		{
		    			goodlinepf[i].dist += img_grad.at<uchar>(l,m);
		    	//		bgauss = (rng.gaussian(linepf[i].senseNoise));	
		    	//		linepf[i].dist += bgauss;
		    		}
		    	}
		    	else if(goodlinepf[i].x[l]<NEIGHBORHOOD){
		    		for(int m=0; m<goodlinepf[i].x[l]+NEIGHBORHOOD; m++)
		    		{
		    			goodlinepf[i].dist += img_grad.at<uchar>(l,m);
		    	//		bgauss = (rng.gaussian(linepf[i].senseNoise));	
		    	//		linepf[i].dist += bgauss;
		    		}
		    	}
		    }//end l
		    
		    w_pf[i] = 0.0;	
		    //Computing the weight of each particle
		    for(int gl=0; gl<NUM_BEST_LINES; gl++){	
		    	prob_dist = gaussian(goodlinepf[i].theta, goodlinepf[i].turnNoise, bestlinepf[gl].theta);	
		      prob_dist *= gaussian(goodlinepf[i].x[0], goodlinepf[i].senseNoise, bestlinepf[gl].x[0]);
		      prob_dist *= gaussian(goodlinepf[i].x[img_grad.rows-1], goodlinepf[i].senseNoise, bestlinepf[gl].x[img_grad.rows-1]);
		    	//cout << prob_dist << endl;
		    	//if( prob_dist > w_pf[i] && bl_points[gl] < (int)NUM_GOOD_LINES/NUM_BEST_LINES)
		    	if( prob_dist > w_pf[i])
		    	{
		    	w_pf[i] = prob_dist;
		    	bl_points[gl]++;
		    	}
		    } //CLOSING gl-LOOP.	
		} //CLOSING i-LOOP.	
			
		//Normalizing the weights.
		float tWeight = 0.0;
		for(int i=0; i<NUM_GOOD_LINES; i++)
		{
			tWeight = tWeight + w_pf[i];
		}
		
		for(int i=0; i<NUM_GOOD_LINES; i++)
		{
			w_pf[i] = w_pf[i]/tWeight;
		}

		srand(time(NULL));
		int index = rand() % NUM_GOOD_LINES;
		float beta1 = 0.0; 
		float beta = 0.0;
		float mw = 0.0; //max. value out of w[1...N]	
		linestruct *goodlinepf1;
		goodlinepf1 = new linestruct [NUM_GOOD_LINES];
		
		//Evaluating the maximum weight
		for (int i=0; i<NUM_GOOD_LINES; i++)
		{
			if(w_pf[i] > mw)
			{
				mw = w_pf[i];
			}	
		}
			cout << "mw = " << mw << endl;
		
		//RESAMPLING STEP
		for (int i=0; i<NUM_GOOD_LINES; i++)
		{
			beta1 = rand()%NUM_GOOD_LINES;
			beta += ((float)beta1/NUM_GOOD_LINES) * ( 2 * mw );
			while (beta > w_pf[index])
			{
				beta -= w_pf[index];
				index = (index + 1) % NUM_GOOD_LINES;
			}
			goodlinepf1[i] = goodlinepf[index];
		}
		
		//Setting the noise and landmarks for the new particles
		for (int i=0; i<NUM_GOOD_LINES; i++)
		{
			goodlinepf[i] = goodlinepf1[i];
		}
	
		FILE* file1 = fopen("bestCord.txt", "a"); 
	 	char output1[255]; 
	 	for (int i=0; i<NUM_BEST_LINES; i++)
		{
			bestlineparams(i,0) = bestlinepf[i].x[0];
			bestlineparams(i,1) = bestlinepf[i].y[0];
			bestlineparams(i,2) = bestlinepf[i].theta;
	 	 	sprintf(output1, "Iteration=1:\n"); // fill the buffer with some information
	 	 	sprintf(output1, "%7.3f   %10.9f\n", bestlineparams(i,0), bestlineparams(i,2)); // fill the buffer with some information
	 		
		fputs(output1, file1); 
		}
		fclose(file1); 
         	
		FILE* file2 = fopen("goodCord.txt", "a"); 
	 	char output2[255]; 
	 	for (int i=0; i<NUM_GOOD_LINES; i++)
		{
			goodlineparams(i,0) = goodlinepf[i].x[0];
			goodlineparams(i,1) = goodlinepf[i].y[0];
			goodlineparams(i,2) = goodlinepf[i].theta;
	 	 	sprintf(output2, "Iteration=1:\n"); // fill the buffer with some information
	 	 	sprintf(output2, "%7.3f  %10.9f\n", goodlineparams(i,0), goodlineparams(i,2)); // fill the buffer with some information
	 		
		fputs(output2, file2); 
		}
		fclose(file2); 

//*****************************************
//if(topIndex==10){
//////std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines2[N];
//////int x1=0, y1=0, x2=0, y2=0;  
//////
////////cv::Mat img_temp[iterations]; 
////////img_temp[topIndex] = img_org;
//////for (int i=NUM_GOOD_LINES; i<N; i++)
//////{
//////r = sqrt(pow(img_gray.rows,2) + pow(img_gray.cols,2));
//////x1 = (int) (goodlinepf[i].x[0]);
//////y1 = (int) (goodlinepf[i].y[0]);
//////x2 = (int) (r * cos(goodlinepf[i].theta) + goodlinepf[i].x[0]);
//////y2 = (int) (r * sin(goodlinepf[i].theta) + goodlinepf[i].y[0]);
//////lines2[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
//////std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it2;  
//////for(it2=lines2[i].begin();it2!=lines2[i].end();it2++)  
//////{  
//////     cv::line(img_org, cv::Point(it2->first.first, it2->first.second), cv::Point(it2->second.first, it2->second.second), cv::Scalar(0,255,0), 1, 8,0);  
//////}
//////
//////}
////////*****************************************
//////std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines1[NUM_GOOD_LINES];
//////x1=0, y1=0, x2=0, y2=0;  
//////
//////for (int i=0; i<NUM_GOOD_LINES; i++)
//////{
//////r = sqrt(pow(img_gray.rows,2) + pow(img_gray.cols,2));
//////x1 = (int) (goodlinepf[i].x[0]);
//////y1 = (int) (goodlinepf[i].y[0]);
//////x2 = (int) (r * cos(goodlinepf[i].theta) + goodlinepf[i].x[0]);
//////y2 = (int) (r * sin(goodlinepf[i].theta) + goodlinepf[i].y[0]);
//////lines1[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
//////std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it1; 
//////for(it1=lines1[i].begin();it1!=lines1[i].end();it1++)  
//////{  
//////     cv::line(img_org, cv::Point(it1->first.first, it1->first.second), cv::Point(it1->second.first, it1->second.second), cv::Scalar(0,0,255), 2, 8,0);  
//////}
//////}
//////
////////*****************************************
//////	cv::namedWindow("Test Window",CV_WINDOW_NORMAL);	
//////	cv::imshow("Test Window", img_org);
//////	cv::waitKey( 10 );

} //CLOSING topIndex-LOOP.	

return 0;
}
