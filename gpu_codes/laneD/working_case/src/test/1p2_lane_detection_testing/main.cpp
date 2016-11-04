#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>

using namespace std;

#define CV_LOAD_IMAGE_GRAYSCALE 0
#define CV_LOAD_IMAGE_COLOR 1
#define CV_WINDOW_NORMAL 0
#define PI 3.14159265358979323846 

typedef struct{
float theta1; //radian
float theta2; //radian
float *x;
float *y;
float dist;
}linestruct;

int main( int argc, char** argv )
{

	int NUM_LINES = 30000;
	int DIST_CRITERIA = atoi(argv[1]);
	int NUM_GOOD_LINES = atoi(argv[2]);
	cv::Mat img_org = cv::imread( argv[3], 
				CV_LOAD_IMAGE_COLOR);

	cv::Mat img_gray;
	img_gray = cv::imread( argv[3], 
				CV_LOAD_IMAGE_GRAYSCALE);
	 
	 cv::Mat img_grad(img_gray.rows, img_gray.cols-1, CV_8UC1); //gradient matrix
	 
	
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
	if((int)img_grad.at<uchar>(row, col) < 0.1*maxVal) 
	img_grad.at<uchar>(row, col) = 0;
	else
	img_grad.at<uchar>(row, col) = 1;
	}
	}

	//cout << img_grad << endl;
	cout << img_grad.rows << endl;
	cout << img_grad.cols << endl;
	
linestruct line[NUM_LINES];
float r;

//cout << "Testing 1" << endl;

for(int ln=0; ln<NUM_LINES; ln++)
{
	line[ln].x = new float [img_grad.rows]; // Intuition says it is .cols but we use .rows cuz we need as many x as we have y for the loop below.
	line[ln].y = new float [img_grad.rows];
	
	
	line[ln].dist = 0.0;
		
	int c2 = 0, c3=0;
	float xadd = 0.0;
   float angVar = 0.0;
	srand(ln*10000+20*ln*ln*ln);

	c2 = rand()%10000;
	xadd = (float)(((float)c2/10000.0000) * img_grad.cols);

	c3 = rand()%10000;
	angVar = (float)(((float)c3/10000.0000) * (PI/6));
	
	line[ln].theta1 = ln * (PI)/NUM_LINES;
   line[ln].theta2 = ln * (PI)/NUM_LINES - angVar;
   
   for(int l=0; l<img_grad.rows; l++)
	{
		line[ln].y[l] = l; 
      
      if(l <= (int)(img_grad.rows/4))
        {
           r = line[ln].y[l]/sin(line[ln].theta1);	
	     	  line[ln].x[l] = abs((int)(xadd + r * (cos(line[ln].theta1))));
           //cout << "Testing from 1-4th, l = " << l << " & ln = " << ln << endl;
        }
      else
        {
           //cout << "Testing from 3-4th, l = " << l << " & ln = " << ln << endl;
           r = (line[ln].y[l]-line[ln].y[(int)(img_grad.rows/4)])/sin(line[ln].theta2);
           line[ln].x[l] = line[ln].x[(int)(img_grad.rows/4)] + r * (cos(line[ln].theta2));
        }
     
           int NEIGHBORHOOD = 3;
	       		if(line[ln].x[l]>=NEIGHBORHOOD && line[ln].x[l] <= img_grad.cols-NEIGHBORHOOD){  
	       			for(int m=line[ln].x[l]-NEIGHBORHOOD; m<line[ln].x[l]+NEIGHBORHOOD; m++)
	       			{
	       				line[ln].dist += img_grad.at<uchar>(l,m);
				}
			}
	       		else if(line[ln].x[l] > img_grad.cols-NEIGHBORHOOD){  
				for(int m=line[ln].x[l]-NEIGHBORHOOD; m<img_grad.cols; m++)
				{
					line[ln].dist += img_grad.at<uchar>(l,m);
				}
			}
			else if(line[ln].x[l]<NEIGHBORHOOD){
				for(int m=0; m<line[ln].x[l]+NEIGHBORHOOD; m++)
				{
					line[ln].dist += img_grad.at<uchar>(l,m);
				}
			}
		
			
	}
}


int best_line_idx[NUM_GOOD_LINES];
int count =0;
float dist_vect[NUM_LINES];

for(int ln=0; ln<NUM_LINES; ln++)
{
dist_vect[ln] = line[ln].dist;
}


std::vector<float> myvector(dist_vect, dist_vect+NUM_LINES-1);
std::sort (myvector.begin(), myvector.end()); 

//for( std::vector<float>::const_iterator i = myvector.begin(); i != myvector.end(); ++i)
for( std::vector<float>::const_iterator i = myvector.end(); i != myvector.begin(); i--)
{
	if(*i != (float)0 ) 
	for(int ln=0; ln<NUM_LINES; ln++)
	{
		if(dist_vect[ln] == *i && count<NUM_GOOD_LINES)
		{
			if(count == 0){
      			best_line_idx[count] = ln;
      			count++;
			}
			else 
				if (count > 0)
				{
		//CONDITION BELOW IS TO CHECK IF THE "GOOD" LINES ARE NOT VERY CLOSE TO EACH OTHER SO 
		//AS TO AVOID ANY SINGLE ROAD LINE BEING REPRESENTED BY TOO MANY "GOOD" LINES
					for(int kl=0; kl<count; kl++){
					if(abs( line[ln].x[img_grad.rows-1] -
						line[best_line_idx[kl]].x[img_grad.rows-1] ) > 
						((float)DIST_CRITERIA/100.0)*img_grad.cols)
						//0.3*img_grad.cols)
						{
						if( kl<count-1 ) continue;
						else{	best_line_idx[count] = ln;
							count++;}
						}
					if(abs( line[ln].x[img_grad.rows-1] -
					line[best_line_idx[kl]].x[img_grad.rows-1] ) < 
					((float)DIST_CRITERIA/100.0)*img_grad.cols) break;
					//0.3*img_grad.cols) break;
					}
				}
		}
}
}

//Best line weights
cout << "Best line weights" << endl;
for(int i=0; i<NUM_GOOD_LINES; i++)
{
cout << line[best_line_idx[i]].dist << endl; 
}

linestruct *best_line;
best_line = new linestruct [NUM_GOOD_LINES];

for(int gl=0; gl<NUM_GOOD_LINES; gl++)
{
best_line[gl] = line[best_line_idx[gl]];
}

int x1=0, y1=0, x2=0, y2=0;  

//std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines[NUM_LINES];
//for (int i=0; i<NUM_LINES; i++)
//{
//   x1 = (int) (line[i].x[0]);
//   y1 = (int) (line[i].y[0]);
//   x2 = (int) (line[i].x[(int)(img_grad.rows/4)]);
//   y2 = (int) (line[i].y[(int)(img_grad.rows/4)]);
//   
//   lines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
//   
//   std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;  
//   for(it=lines[i].begin();it!=lines[i].end();it++)  
//   {  
//        cv::line(img_org, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar(0,255,0), 5, 8,0);  
//   }
// 
//   //glines[i].clear();
//    
//   x1 = (int) (line[i].x[(int)(img_grad.rows/4)]);
//   y1 = (int) (line[i].y[(int)(img_grad.rows/4)]);
//   x2 = (int) (line[i].x[img_grad.rows-1]);
//   y2 = (int) (line[i].y[img_grad.rows-1]);
//   
//   lines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
//   
//   //std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;  
//   for(it=lines[i].begin();it!=lines[i].end();it++)  
//   {  
//        cv::line(img_org, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar(0,255,0), 5, 8,0);  
//   } 
//}

std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > glines[NUM_GOOD_LINES];

for (int i=0; i<NUM_GOOD_LINES; i++)
{
   //r = sqrt(pow(img_gray.rows,2) + pow(img_gray.cols,2));
   x1 = (int) (best_line[i].x[0]);
   y1 = (int) (best_line[i].y[0]);
   // x2 = (int) (r * cos(best_line[i].theta) + best_line[i].x[0]);
   // y2 = (int) (r * sin(best_line[i].theta) + best_line[i].y[0]);
   x2 = (int) (best_line[i].x[(int)(img_grad.rows/4)]);
   y2 = (int) (best_line[i].y[(int)(img_grad.rows/4)]);
   
   glines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
   
   std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;  
   for(it=glines[i].begin();it!=glines[i].end();it++)  
   {  
        cv::line(img_org, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar(0,0,255), 5, 8,0);  
   }
 
   //glines[i].clear();
    
   x1 = (int) (best_line[i].x[(int)(img_grad.rows/4)]);
   y1 = (int) (best_line[i].y[(int)(img_grad.rows/4)]);
   x2 = (int) (best_line[i].x[img_grad.rows-1]);
   y2 = (int) (best_line[i].y[img_grad.rows-1]);
   
   glines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
   
   //std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;  
   for(it=glines[i].begin();it!=glines[i].end();it++)  
   {  
        cv::line(img_org, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar(0,0,255), 5, 8,0);  
   } 
} 
	cv::namedWindow("Test Window",CV_WINDOW_NORMAL);	
	cv::imshow("Test Window", img_org);
	cv::waitKey( 0 );
return 0;
}
