#include <stdio.h>
// Declare some vector types. This should work on most compilers 
// that try to be GCC compatible. Alternatives are provided 
// for those that don't conform to GCC behavior in vector 
// type declaration.
// Here a vFloat is a vector of four floats, and 
// a vInt is a vector of four 32-bit ints. 
#if 1
	// This should work on most compilers that try 
	// to be GCC compatible 
	// cc main.c -Wall -pedantic
	typedef float vFloat __attribute__ ((__vector_size__(16)));
	typedef int vInt __attribute__ ((__vector_size__(16))); 
	#define init_vFloat(a, b, c, d) (const vFloat) {a, b, c, d}
#else
	//Not GCC compatible 
	#if defined( __SSE2__ )
		// depending on compiler you might need to pass 
		// something like -msse2 to turn on SSE2
		#include <emmintrin.h> 
		typedef __m128 vFloat; 
		typedef __m128i vInt;
		static inline vFloat init_vFloat(float a, float b, float c, float d);
		static inline vFloat init_vFloat(float a, float b, float c, float d);
		{ union{ vFloat v; float f[4];}u; u.f[0] = a; u.f[1] = b; u.f[2] = c; u.f[3] = d; return u.v;
		} 
	#elif defined( __VEC__ )
		// depending on compiler you might need to pass 
		// something like -faltivec or -maltivec or 
		// "Enable AltiVec Extensions" to turn this part on
		#include <altivec.h> 
		typedef vector float vFloat; 
		typedef vector int vInt;
		#if 1
			// for compliant compilers 
			#define init_vFloat(a, b, c, d) (const vFloat) (a, b, c, d)
		#else // for FSF GCC
			#define init_vFloat(a, b, c, d) (const vFloat) {a, b, c, d}
		#endif 
	#endif
#endif

void print_vInt(vInt v) {
	union{ vInt v; 
	int i[4]; }
	u;
	u.v = v;
	printf("vInt: 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x\n", u.i[0], u.i[1], u.i[2], u.i[3]);
}

void print_vFloat(vFloat v) {
	union{ 	vFloat v; 
	float i[4]; }
	u; 
	u.v = v;
	printf("vFloat: %f %f %f %f\n", u.i[0], u.i[1], u.i[2], u.i[3]); 
}

int main(void) {
	vFloat f;
	f = init_vFloat(1.0f, 2.0f, 3.0f, 4.0f); 
	vInt i; 
	print_vFloat(f);
	printf("assign with cast: vInt i = (vInt) f;\n" ); 
	i = (vInt) f;
	print_vInt(i); 
	return 0;
}

