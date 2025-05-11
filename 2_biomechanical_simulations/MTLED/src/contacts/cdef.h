/* This file containes type definitions.
It is platform dependent */

#if !defined(_CDEF_H_)
#define _CDEF_H_

// signed integers:
typedef signed char int8;
typedef signed short int16;
typedef signed long int32;

// un-signed integers:
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned long uint32;

// floating types
typedef float float4;
typedef double float8;
typedef long double float10;

// definitions
//#define INLINE __inline
#define INLINE

#define ON 1
#define OFF 0

#define YES 1
#define NO 0

#define E_OK 0
#define E_ERROR 1

#endif
