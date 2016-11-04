#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <math.h>

class RNG
{
	public:
		RNG( uint64_t seed );

		uint32_t	uint32();
		uint64_t	uint64();
		float		fl();
		double		db();

		/* uniform distribution */
		double		uniform( double min, double max );
		float		uniform( float min, float max );
		int			uniform( int min, int max );

		/* zero mean, with sigma stddev */
		double		gaussian( double sigma );

	private:
		uint64_t _u;
		uint64_t _v;
		uint64_t _w;
};

inline RNG::RNG( uint64_t seed ) :
	_v( 4101842887655102017LL ),
	_w( 1 )
{
	_u = seed ^ _v;
	uint64();
	_v = _u;
	uint64();
	_w = _v;
	uint64();
}

inline uint64_t RNG::uint64()
{
	_u = _u * 2862933555777941757LL + 7046029254386353087LL;
	_v ^= _v >> 17;
	_v ^= _v << 31;
	_v ^= _v >> 8;
	_w = 4294957665U * ( _w & 0xffffffff ) + ( _w >> 32 );

	uint64_t x = _u ^ ( _u << 21 );

	x ^= x >> 35;
	x ^= x << 4;

	return ( x + _v ) ^ _w;
}

/* between 0 and 1 */
inline float RNG::fl()
{
	return ( float )2.3283064365386962890625E-10 * uint32();
}

/* between 0.0 and 1.0 */
inline double RNG::db()
{
	/* 2^-64 * uint64 */
	return 5.4210108624275221700372640043497E-20 * uint64();
}

inline uint32_t RNG::uint32()
{
	return ( uint32_t ) uint64();
}

inline double RNG::gaussian( double sigma )
{
	double u, v, x, y, q;
	do {
		u = db();
		v = 1.7156 * ( db() - 0.5 );
		x = u - 0.449871;
		y = fabs( v ) + 0.386595;
		q = x*x + y * ( 0.19600 * y - 0.25472 * x );
	} while ( q > 0.27597 && ( q > 0.27846 || v*v > -4.0 * log( u ) *u*u ) );

	return sigma * v / u;
}

inline double RNG::uniform( double min, double max )
{
	return min + db() * ( max - min );
}

inline float RNG::uniform( float min, float max )
{
	return min + fl() * ( max - min );
}

inline int RNG::uniform( int min, int max )
{
	return (int) ( min + fl() * ( max - min ) );
}


#endif
