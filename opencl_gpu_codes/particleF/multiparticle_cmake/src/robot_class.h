#define PI 3.141592653589793  

using namespace Eigen;
using namespace std;

double cyclicWorld(double, double);
//The Robot Class
class Robot{
private:
	double xCord;
	double yCord;
	double orientation;	//This is angle in radians.
	double forwardNoise;
	double turnNoise;
	double senseNoise;
	MatrixXd landmarks;
public: 
	//Constructors
	Robot(){};
	~Robot(){};	

	//Accessor Methods
	void set(double, double, double, double);
	Vector3d get() const;
	void setNoise(double, double, double);
	void setLandmarks(MatrixXd);
	
	//Other Methods
	void init(double, double);
	void move(double, double, double);
	VectorXd sense() const;
	double gaussian(double, double, double) const;
	double measurement_prob(VectorXd) const;
};

//Robot Member Functions
void Robot::init(double world_size, double a)
{
	int c1, c2, c3;

	srand(a);
	c1 = rand()%10000;
	xCord = ((double)c1/10000.0000) * world_size;

	srand(a+1.324);
	c2 = rand()%10000;
	yCord = ((double)c2/10000.0000) * world_size;

	srand(a+23343.33323);
	c3 = rand()%10000;
	orientation = ((double)c3/10000.0000) * 2 * PI;
	
	forwardNoise = 0.0;
	turnNoise = 0.0;
	senseNoise = 0.0;
}

void Robot::set(double new_x, double new_y, double new_orientation, double world_size)
{
	if ( new_x<0 || new_x>=world_size)
	printf("X coordinate is out of bounds\n");
	if ( new_y<0 || new_y>=world_size)
	printf("Y coordinate is out of bounds\n");
	if ( new_orientation<0 || new_orientation>=2.0*PI)
	printf("Orientation is out of bounds\n");
	xCord = new_x;
	yCord = new_y;
	orientation = new_orientation;
}

//void Robot::get() const
Vector3d Robot::get() const
{
	Vector3d coordinates(xCord, yCord, orientation);
	return coordinates;
}

void Robot::setNoise(double new_f_noise, double new_t_noise, double new_s_noise)
{
	//makes it possible to change the noise parameters.
	//this is often useful in particle filters.
	forwardNoise = new_f_noise;
	turnNoise = new_t_noise;
	senseNoise = new_s_noise;
}

VectorXd Robot::sense() const
{
	double b;
	RNG rng(time(NULL));
	VectorXd dist(landmarks.rows());
	for (int i=0; i<landmarks.rows(); i++)
	{
	dist[i] = sqrt(pow((xCord-landmarks(i,0)),2)+pow((yCord-landmarks(i,1)),2));
	b = rng.gaussian(senseNoise);
	dist[i] = dist[i]+b;
	}
	return dist;
}

void Robot::move(double turnAngle, double moveDistance, double world_size) 
{
	RNG rng(time(NULL));

	if(moveDistance < 0){printf("Robot can't move backwards!!!\n");}

	orientation = orientation + turnAngle + rng.gaussian(turnNoise);
	orientation = cyclicWorld(orientation, 2*PI);

	moveDistance = moveDistance + rng.gaussian(forwardNoise);

	xCord = xCord + (moveDistance * cos(orientation));
	xCord = cyclicWorld(xCord, world_size); //cyclic truncation.

	yCord = yCord + (moveDistance * sin(orientation));
	yCord = cyclicWorld(yCord, world_size);
}

double Robot::gaussian(double mu, double sigma, double x) const
{
	return (1/sqrt(2.*PI*pow(sigma,2)))*exp(-0.5*pow((x-mu),2)/pow(sigma,2));
	//Observe the expression (x-mu)^2 above. Greater this difference, lesser 
	//the value of ¨return¨. So when we cal this 'gaussian' function from 
	//'measurement_prob' function, this difference is nothing but the difference
	//between the distances measured to each of the landmarks from robot and 
	//each particle. So, farther a particle from robot, smaller the value of 'prob'
	//which represents the weight of that particle.  
}

double Robot::measurement_prob(VectorXd measurement) const
{
	//Calculates how likely a measurement should be
	//which is an essential step
	double prob = 1.0;
	double dist;
	for (int i=0; i<landmarks.rows(); i++)
	{
	dist = sqrt(pow((xCord-landmarks(i,0)),2)+pow((yCord-landmarks(i,1)),2));
	prob *= gaussian(dist, senseNoise, measurement[i]);
	}
	return prob;
}

void Robot::setLandmarks(MatrixXd new_landmarks)
{
	landmarks = new_landmarks;
}

//Subfunction below calculates a%b where a and b are doubles. 
//But modulus operator "%" in C++ only works for integers.
//Hence the implementation had to be indirect in case of doubles. 
double cyclicWorld(double a, double b)
{
	if (a>=0)
	{
		return a-b*(int)(a/b);
	}
	else
	{
		return a+b*(1+(int)(abs(a/b)));
	}	
}
