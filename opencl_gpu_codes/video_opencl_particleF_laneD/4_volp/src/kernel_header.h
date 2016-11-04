#ifndef __KERNEL_HEADER_H_INCLUDED__
#define __KERNEL_HEADER_H_INCLUDED__
//-----------------------
typedef struct{
   float forwardNoise;
   float turnNoise;
   float senseNoise;
} noiseStruct;
//----------------------------
typedef struct{
   int NEIGHBORHOOD;
   int NUM_GOOD_LINES;   
   int NUM_BEST_LINES;   
   int imgrows;
   int imgcols;
   ulong framme_seed; 
} wtKernParams;
//-----------------------
typedef struct{
   int imgrows;
   int imgcols;  
   int NUM_LINES;
   int NEIGHBORHOOD;
} createLineParams;
//-----------------------
float cyclicWorld_kernel(float a, float b)
{
	if (a>=0)
		return a-b*(int)(a/b);
	else
		return a+b*(1+(int)(fabs(a/b)));
}
//----------------------------
float gaussian_kernel(float mu, float sigma, float x)
{
	return (1/sqrt(2.0f*M_PI_F*pow(sigma,2)))*exp(-0.5f*pow((x-mu),2)/pow(sigma,2));
}
//----------------------------
int findn(uint num)
{
    return (int) (log10((float)num)) + 1;
}	
//------------------RNG---------------------
/*
Part of MWC64X by David Thomas, dt10@imperial.ac.uk
This is provided under BSD, full license is with the main package.
See http://www.doc.ic.ac.uk/~dt10/research
*/
#ifndef dt10_mwc64x_rng_cl
#define dt10_mwc64x_rng_cl

// Pre: a<M, b<M
// Post: r=(a+b) mod M
ulong MWC_AddMod64(ulong a, ulong b, ulong M)
{
	ulong v=a+b;
	if( (v>=M) || (v<a) )
		v=v-M;
	return v;
}

// Pre: a<M,b<M
// Post: r=(a*b) mod M
// This could be done more efficently, but it is portable, and should
// be easy to understand. It can be replaced with any of the better
// modular multiplication algorithms (for example if you know you have
// double precision available or something).
ulong MWC_MulMod64(ulong a, ulong b, ulong M)
{	
	ulong r=0;
	while(a!=0){
		if(a&1)
			r=MWC_AddMod64(r,b,M);
		b=MWC_AddMod64(b,b,M);
		a=a>>1;
	}
	return r;
}

// Pre: a<M, e>=0
// Post: r=(a^b) mod M
// This takes at most ~64^2 modular additions, so probably about 2^15 or so instructions on
// most architectures
ulong MWC_PowMod64(ulong a, ulong e, ulong M)
{
	ulong sqr=a, acc=1;
	while(e!=0){
		if(e&1)
			acc=MWC_MulMod64(acc,sqr,M);
		sqr=MWC_MulMod64(sqr,sqr,M);
		e=e>>1;
	}
	return acc;
}

uint2 MWC_SkipImpl_Mod64(uint2 curr, ulong A, ulong M, ulong distance)
{
	ulong m=MWC_PowMod64(A, distance, M);
	ulong x=curr.x*(ulong)A+curr.y;
	x=MWC_MulMod64(x, m, M);
	return (uint2)((uint)(x/A), (uint)(x%A));
}

uint2 MWC_SeedImpl_Mod64(ulong A, ulong M, uint vecSize, uint vecOffset, ulong streamBase, ulong streamGap)
{
	// This is an arbitrary constant for starting LCG jumping from. I didn't
	// want to start from 1, as then you end up with the two or three first values
	// being a bit poor in ones - once you've decided that, one constant is as
	// good as any another. There is no deep mathematical reason for it, I just
	// generated a random number.
	enum{ MWC_BASEID = 4077358422479273989UL };
	
	ulong dist=streamBase + (get_global_id(0)*vecSize+vecOffset)*streamGap;
	ulong m=MWC_PowMod64(A, dist, M);
	
	ulong x=MWC_MulMod64(MWC_BASEID, m, M);
	return (uint2)((uint)(x/A), (uint)(x%A));
}

//! Represents the state of a particular generator
typedef struct{ uint x; uint c; } mwc64x_state_t;

enum{ MWC64X_A = 4294883355U };
enum{ MWC64X_M = 18446383549859758079UL };

//void MWC64X_Step(__local mwc64x_state_t *s)
void MWC64X_Step(mwc64x_state_t *s)
{
	uint X=s->x, C=s->c;
	
	uint Xn=MWC64X_A*X+C;
	uint carry=(uint)(Xn<C);				// The (Xn<C) will be zero or one for scalar
	uint Cn=mad_hi(MWC64X_A,X,carry);  
	
	s->x=Xn;
	s->c=Cn;
}

void MWC64X_Skip(mwc64x_state_t *s, ulong distance)
{
	uint2 tmp=MWC_SkipImpl_Mod64((uint2)(s->x,s->c), MWC64X_A, MWC64X_M, distance);
	s->x=tmp.x;
	s->c=tmp.y;
}

//void MWC64X_SeedStreams(__local mwc64x_state_t *s, ulong baseOffset, ulong perStreamOffset)
void MWC64X_SeedStreams(mwc64x_state_t *s, ulong baseOffset, ulong perStreamOffset)
{
	uint2 tmp=MWC_SeedImpl_Mod64(MWC64X_A, MWC64X_M, 1, 0, baseOffset, perStreamOffset);
	s->x=tmp.x;
	s->c=tmp.y;
}

//! Return a 32-bit integer in the range [0..2^32)
//uint MWC64X_NextUint(__local mwc64x_state_t *s)
uint MWC64X_NextUint(mwc64x_state_t *s)
{
	//__local uint res;
	uint res;
	res = s->x ^ s->c;
	MWC64X_Step(s);
	return res;
}

//float MWC64X_NextFloat(__local mwc64x_state_t *s)
float MWC64X_NextFloat(mwc64x_state_t *s)
{
	uint res=s->x ^ s->c;
	MWC64X_Step(s);
	return  2.3283064365386962890625e-10f * ( float ) res;
}

typedef mwc64x_state_t RNG;

#define RNG_init( rng, linpos, numdraws ) do {\
	MWC64X_SeedStreams( &rng, 0, 0 ); \
	MWC64X_Skip( &rng, ( linpos ) * numdraws ); \
} while( 0 )

#define RNG_float( x ) MWC64X_NextFloat( x )

#endif

//--------------------END RNG---------------------
//--------------------START RNG GAUSS---------------------
//float rng_gaussian( float sigma,__local mwc64x_state_t *rng)
float rng_gaussian( float sigma, mwc64x_state_t *rng)
{
	float u, v, x, y, q;
   
  // mwc64x_state_t rng;
  // ulong samplesPerStream = 30;
  // MWC64X_SeedStreams(&rng,0xf00dcafe, samplesPerStream);
   
   do {
		u = MWC64X_NextFloat(rng);
		v = 1.7156f * MWC64X_NextFloat(rng) - 0.5f ;
		x = u - 0.449871f;
		y = fabs( v ) + 0.386595f;
		q = x*x + y * ( 0.19600f * y - 0.25472f * x );
	} while ( q > 0.27597f && ( q > 0.27846f || v*v > -4.0f * log( u ) *u*u ) );

	return (float)(sigma * v / u);
}

//--------------------END RNG GAUSS---------------------
#endif


