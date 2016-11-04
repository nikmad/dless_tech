#ifndef __KERNEL_HEADER_H_INCLUDED__
#define __KERNEL_HEADER_H_INCLUDED__

//----------------------------
typedef struct{
   float forwardNoise;
   float turnNoise;
   float senseNoise;
}noiseStruct;
//----------------------------
//typedef struct{
//float x[298];
//float y[79];
////float x[277];
////float y[101];
////float x[575];
////float y[96];
////float x[600];
////float y[200];
//float dist;
//float theta; 
//}linestruct;
//----------------------------
typedef struct{
   int imgrows;
   int imgcols;  
   int NUM_LINES;
   int NEIGHBORHOOD;
}createLineParams;
//----------------------------
typedef struct{
   int NEIGHBORHOOD;
   int NUM_GOOD_LINES;   
   int NUM_BEST_LINES;   
   int imgrows;
   int imgcols;
}wtKernParams;
//----------------------------
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
//linestruct move_line_kernel(linestruct line, float turnAngle, 
//   float moveDistance, int imgrows, int imgcols, float rngturn, float rngfwd) 
//{
//	line.theta = line.theta + turnAngle + rngturn;
//	line.theta = cyclicWorld_kernel(line.theta, M_PI_F);
//	
//   moveDistance = moveDistance + rngfwd;
//	float xincept = line.x[0]+moveDistance;
//	float r=0.0f;
//	for(int l=0; l<imgrows; l++)
//	{
//		line.y[l] = l; 
//		r = line.y[l]/sin(line.theta);	
//		line.x[l] = abs((int)(xincept + r * (cos(line.theta))));
//		line.x[l] = cyclicWorld_kernel(line.x[l], imgcols);
//	}
//
//   return line;
//}
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

void MWC64X_Step(__local mwc64x_state_t *s)
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

void MWC64X_SeedStreams(__local mwc64x_state_t *s, ulong baseOffset, ulong perStreamOffset)
{
	uint2 tmp=MWC_SeedImpl_Mod64(MWC64X_A, MWC64X_M, 1, 0, baseOffset, perStreamOffset);
	s->x=tmp.x;
	s->c=tmp.y;
}

//! Return a 32-bit integer in the range [0..2^32)
uint MWC64X_NextUint(__local mwc64x_state_t *s)
{
	__local uint res;
	res = s->x ^ s->c;
	MWC64X_Step(s);
	return res;
}

float MWC64X_NextFloat(__local mwc64x_state_t *s)
{
	uint res=s->x ^ s->c;
	MWC64X_Step(s);
	return  2.3283064365386962890625e-10f * ( float ) res;
}

typedef mwc64x_state_t RNG;

#define RNG_init( x, linpos, numdraws ) do {\
	MWC64X_SeedStreams( &rng, 0, 0 ); \
	MWC64X_Skip( &rng, ( linpos ) * numdraws ); \
} while( 0 )

#define RNG_float( x ) MWC64X_NextFloat( x )

#endif

//--------------------END RNG---------------------
//--------------------START RNG GAUSS---------------------

//ulong4 rng_ulong(ulong4 rng_ulong_inp)
//{
//   rng_ulong_inp.x = rng_ulong_inp.x * 2862933555777941757U + 7046029254386353087U;
//	rng_ulong_inp.y ^= rng_ulong_inp.y >> 17;
//	rng_ulong_inp.y ^= rng_ulong_inp.y << 31;
//	rng_ulong_inp.y ^= rng_ulong_inp.y >> 8;
//	rng_ulong_inp.z = 4294957665U * ( rng_ulong_inp.z & 0xffffffff ) + ( rng_ulong_inp.z >> 32 );
//
//	ulong x = rng_ulong_inp.x ^ ( rng_ulong_inp.x << 21 );
//
//	x ^= x >> 35;
//	x ^= x << 4;
//
//   rng_ulong_inp.w = ( x + rng_ulong_inp.y ) ^ rng_ulong_inp.z;
//   
//   ulong4 rng_ulong_ret;
//   rng_ulong_ret = rng_ulong_inp;
//
//   return rng_ulong_ret;	
//}
//
//ulong4 rng_gauss_seed( ulong seed ) 
//{
//	ulong4 rng_ulong_ret;
//   ulong final_num = 0;
//
//   rng_ulong_ret.y = 4101842887655102017U;
//   //rng_ulong_ret.y = 2147483647;
//   rng_ulong_ret.x = seed ^ rng_ulong_ret.y;
//   rng_ulong_ret.z = 1;
//   rng_ulong_ret.w = final_num;
//	
//   rng_ulong_ret = rng_ulong(rng_ulong_ret);
//
//   rng_ulong_ret.y = rng_ulong_ret.x;
//   rng_ulong_ret = rng_ulong(rng_ulong_ret);
//	
//   rng_ulong_ret.z = rng_ulong_ret.y;
//   rng_ulong_ret = rng_ulong(rng_ulong_ret);
//   
//   return rng_ulong_ret;
//}
//
//float db_gauss(ulong4 db_gauss_inp)
//{
//	/* 2^-64 * rng_ulong */
//   db_gauss_inp = rng_ulong(db_gauss_inp);
//	return (float)(5.4210108624275221700372640043497E-20 * db_gauss_inp.w);
//}

//float rng_gaussian( float sigma, ulong4 db_gauss_inp )
float rng_gaussian( float sigma,__local mwc64x_state_t *rng)

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
