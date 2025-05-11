/* Vector operations */

#if !defined(_VECT_H_)
#define _VECT_H_

#include "cdef.h"
#include "stdio.h"

#define VECT_Abs(x) ((x)>0?(x):(-(x)))

template <class type>
INLINE void VECT_vInit(type *pVect, uint32 u32Size, type Value)
{ 
	uint32 i; 
	type *pT = pVect; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*pT = Value; 
		pT++; 
	} 
} 

// copy
template <class type>
INLINE void VECT_vCopy(type *pV1, type *pV2, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *p2; 
	p1 = pV1; p2 = pV2; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*p2 = *p1; 
		p1++; p2++; 
	} 
} 

// swap
template <class type>
INLINE void VECT_vSwap(type *pV1, type *pV2, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *p2, f; 
	p1 = pV1; p2 = pV2; 
	for (i = 0; i < u32Size; i++) 
	{ 
		f = *p2;
		*p2 = *p1;
		*p1 = f;
		p1++; p2++; 
	} 
} 

// addition
template <class type>
INLINE void VECT_vAdd(type *pV1, type *pV2, type *pVr, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *p2, *pr; 
	p1 = pV1; p2 = pV2; pr = pVr; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*pr = *p1 + *p2; 
		p1++; p2++; pr++; 
	} 
} 

// scalar addition
template <class type>
INLINE void VECT_vAddScalar(type *pV1, type x, type *pVr, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *pr; 
	p1 = pV1; pr = pVr; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*pr = *p1 + x; 
		p1++; pr++; 
	} 
} 

// incrementation
template <class type>
INLINE void VECT_vIncrement(type *pV1, type x, uint32 u32Size)
{ 
	uint32 i; 
	type *p1; 
	p1 = pV1; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*p1 += x; 
		p1++; 
	} 
} 

// substraction
template <class type>
INLINE void VECT_vSubstract(type *pV1, type *pV2, type *pVr, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *p2, *pr; 
	p1 = pV1; p2 = pV2; pr = pVr; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*pr = *p1 - *p2; 
		p1++; p2++; pr++; 
	} 
} 


// substract from scalar
template <class type>
INLINE void VECT_vSubstractFromScalar(type *pV1, type x, type *pVr, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *pr; 
	p1 = pV1; pr = pVr; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*pr = x - *p1; 
		p1++; pr++; 
	} 
} 

// multiplication
template <class type>
INLINE void VECT_vMultiply(type *pV1, type *pV2, type *pVr, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *p2, *pr; 
	p1 = pV1; p2 = pV2; pr = pVr; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*pr = (*p1)*(*p2); 
		p1++; p2++; pr++; 
	} 
} 

// division
template <class type>
INLINE void VECT_vDivide(type *pV1, type *pV2, type *pVr, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *p2, *pr; 
	p1 = pV1; p2 = pV2; pr = pVr; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*pr = (*p1)/(*p2); 
		p1++; p2++; pr++; 
	} 
} 

// scalar product
template <class type>
INLINE float8 VECT_f8ScalarProduct(type *pV1, type *pV2, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *p2; 
	float8 res; 
	p1 = pV1; p2 = pV2; res = (*p1) * (*p2); 
	for (i = 1; i < u32Size; i++) 
	{ 
		p1++; p2++; 
		res += (*p1) * (*p2); 
	} 
	return res; 
} 

// multiplication with scalar
template <class type>
INLINE void VECT_vMultiplyScalar(type *pV1, type x, type *pVr, uint32 u32Size)
{ 
	uint32 i; 
	type *p1, *pr; 
	p1 = pV1; pr = pVr; 
	for (i = 0; i < u32Size; i++) 
	{ 
		*pr = (*p1) * x; 
		p1++; pr++; 
	} 
} 

// division to scalar
#define VECT_vDivideScalar(pV1, x, pVr, u32Size) \
                  VECT_vMultiplyScalar(pV1, 1./(x), pVr, u32Size)

// norm L1
template <class type>
INLINE float8 VECT_f8Norm1(type *pV, uint32 u32Size)
{ 
	uint32 i; 
	type *p; 
	float8 res; 
	p = pV; res = VECT_Abs(*p); 
	for (i = 1; i < u32Size; i++) 
	{ 
		p++; 
		res += VECT_Abs(*p); 
	} 
	return res; 
} 

// maximum element value
template <class type>
INLINE float8 VECT_f8MaxValue(type *pV, uint32 u32Size)
{ 
	uint32 i; 
	type *p; 
	float8 res, x; 
	p = pV; res = *p; 
	for (i = 1; i < u32Size; i++) 
	{ 
		p++; 
		x = *p; 
		if (x > res) { res = x; }; 
	} 
	return res; 
} 

// minimum element value
template <class type>
INLINE float8 VECT_f8MinValue(type *pV, uint32 u32Size)
{ 
	uint32 i; 
	type *p; 
	float8 res, x; 
	p = pV; res = *p; 
	for (i = 1; i < u32Size; i++) 
	{ 
		p++; 
		x = *p; 
		if (x < res) { res = x; }; 
	} 
	return res; 
} 

// norm Linf
template <class type>
INLINE float8 VECT_f8NormInf(type *pV, uint32 u32Size)
{ 
	uint32 i; 
	type *p; 
	float8 res, x; 
	p = pV; res = VECT_Abs(*p); 
	for (i = 1; i < u32Size; i++) 
	{ 
		p++; 
		x = VECT_Abs(*p); 
		if (x > res) { res = x; }; 
	} 
	return res; 
} 

// relative tolerance using norm Linf
template <class type>
INLINE float8 VECT_f8TolRelNormInf(type *pV, type *pVbase, uint32 u32Size)
{ 
	uint32 i; 
	type *p, *p1; 
	float8 res, res1, x; 
	p = pV;  
	p1 = pVbase; 
	res = VECT_Abs(*p1);
	res1 = VECT_Abs(*p - *p1);
	for (i = 1; i < u32Size; i++) 
	{ 
		p++; p1++;
		x = VECT_Abs(*p1); 
		if (x > res) { res = x; }; 
		x = VECT_Abs(*p - *p1); 
		if (x > res1) { res1 = x; }; 
	} 
	return res1/res; 
} 


// sub-vector extraction
template <class type>
INLINE void VECT_vExtractSubVector(type *pV, uint32 *pu32Idx, 
                                  uint32 u32IdxSize, type *pVr)
{ 
	uint32 i; 
	type *pr; 
	uint32 *pIdx; 
	pIdx = pu32Idx; pr = pVr; 
	for (i = 0; i < u32IdxSize; i++) 
	{ 
		*pr = pV[*pIdx]; 
		pr++; pIdx++; 
	} 
}

// sub-vector attribution
template <class type>
INLINE void VECT_vAttribSubVector(type *pV, uint32 *pu32Idx, 
                                          uint32 u32IdxSize, type *pVr)
{ 
	uint32 i; 
	type *pr; 
	uint32 *pIdx; 
	pIdx = pu32Idx; pr = pVr; 
	for (i = 0; i < u32IdxSize; i++) 
	{ 
		pV[*pIdx] = *pr; 
		pr++; pIdx++; 
	} 
}

// scalar attribution to sub-vector
template <class type>
INLINE void VECT_vAttribScalarToSubVector(type *pV, uint32 *pu32Idx, 
                                                 uint32 u32IdxSize, type x)
{ 
	uint32 i; 
	uint32 *pIdx; 
	pIdx = pu32Idx; 
	for (i = 0; i < u32IdxSize; i++) 
	{ 
		pV[*pIdx] = x; 
		pIdx++; 
	} 
}


// sub-vector addition
template <class type>
INLINE void VECT_vAddToSubVector(type *pV, uint32 *pu32Idx, 
                                                 uint32 u32IdxSize, type *pVr)
{ 
	uint32 i; 
	type *pr;
	uint32 *pIdx; 
	pIdx = pu32Idx; pr = pVr; 
	for (i = 0; i < u32IdxSize; i++) 
	{ 
		pV[*pIdx] += *pr; 
		pr++; pIdx++; 
	} 
}

// scalar addition to subvector
template <class type>
INLINE void VECT_vAddScalarToSubVector(type *pV, uint32 *pu32Idx, 
                                                uint32 u32IdxSize, type x)
{ 
	for (uint32 i = 0; i < u32IdxSize; i++) 
	{ 
		pV[pu32Idx[i]] += x; 
	} 
}

// search element
template <class type>
INLINE int32 VECT_i32FindElement(type *pV, type x, uint32 u32Size)
{ 
	for (uint32 i = 0; i < u32Size; i++) 
	{ 
		if (pV[i] == x) return (int32)i; 
	} 
	return -1;
}

template <class type>
void VECT_vSaveBinary(type *pT, uint32 u32Size, FILE *pf)
{
	// save vector
	fwrite(&u32Size, sizeof(uint32), 1, pf);
	fwrite(pT, sizeof(type), u32Size, pf);
}

template <class type>
void VECT_vReadBinary(type *pT, uint32 &u32Size, FILE *pf)
{
	fread(&u32Size, sizeof(uint32), 1, pf);
	if (pT != NULL) delete[] pT;
	pT = new type[u32Size];
	fread(pT, sizeof(type), u32Size, pf);
}

#endif
