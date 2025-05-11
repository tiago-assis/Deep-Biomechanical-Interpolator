/* Matrix operations for float8 matrix */
#if !defined(_MATRIX_H)
#define _MATRIX_H

#include "cdef.h"
#include "vect.h"

// initialization
#define MATRIX_pAllocMem(u32Lines, u32Cols) (float8 *)MEM_pvAlloc((u32Lines)*(u32Cols)*sizeof(float8))
#define MATRIX_pAllocRAM(u32Lines, u32Cols) (float8 *)MEM_pvAllocRAM((u32Lines)*(u32Cols)*sizeof(float8))


#define MATRIX_vInit(pMat, u32Lines, u32Cols, f8Value) VECT_vInit(pMat, (u32Lines)*(u32Cols), f8Value)

// de-initialization
#define MATRIX_vFreeMem(pMat) MEM_vFree(pMat)

// copy
#define MATRIX_vCopy(pM1, pM2, u32Lines, u32Cols) VECT_vCopy(pM1, pM2, (u32Lines)*(u32Cols))

// addition
#define MATRIX_vAdd(pM1, pM2, pMr, u32Lines, u32Cols) VECT_vAdd(pM1, pM2, pMr, (u32Lines)*(u32Cols))

// scalar addition
#define MATRIX_vAddScalar(pM1, x, pMr, u32Lines, u32Cols) VECT_vAddScalar(pM1, x, pMr, (u32Lines)*(u32Cols))

// incrementation
#define MATRIX_vIncrement(pM1, x, u32Lines, u32Cols) VECT_vIncrement(pM1, x, (u32Lines)*(u32Cols))

// substraction
#define MATRIX_vSubstract(pM1, pM2, pMr, u32Lines, u32Cols) VECT_vSubstract(pM1, pM2, pMr, (u32Lines)*(u32Cols))

// substract from scalar
#define MATRIX_vSubstractFromScalar(pM1, x, pMr, u32Lines, u32Cols) VECT_vSubstractFromScalar(pM1, x, pMr, (u32Lines)*(u32Cols))

// multiplication
#define MATRIX_vMultiply(pM1, pM2, pMr, u32Lines, u32Cols) VECT_vMultiply(pM1, pM2, pMr, (u32Lines)*(u32Cols))

// multiplication with scalar
#define MATRIX_vMultiplyScalar(pM1, x, pMr, u32Lines, u32Cols) VECT_vMultiplyScalar(pM1, x, pMr, (u32Lines)*(u32Cols))

// division to scalar
#define MATRIX_vDivideScalar(pM1, x, pMr, u32Lines, u32Cols) VECT_vDivideScalar(pM1, x, pMr, (u32Lines)*(u32Cols))

// product
INLINE void MATRIX_vProduct(float8 *pM1, float8 *pM2, float8 *pMr, uint32 u32Lines, uint32 u32Cols1, uint32 u32Cols2);
INLINE void MATRIX_vProductMTM(float8 *pM1, float8 *pM2, float8 *pMr, uint32 u32Lines, uint32 u32Cols1, uint32 u32Cols2);
INLINE void MATRIX_vProductMMT(float8 *pM1, float8 *pM2, float8 *pMr, uint32 u32Lines, uint32 u32Cols1, uint32 u32Cols2);
INLINE void MATRIX_vProduct333(float8 *pM1, float8 *pM2, float8 *pMr);
INLINE void MATRIX_vProduct163(float8 *pM1, float8 *pM2, float8 *pMr);
INLINE void MATRIX_vProduct333T(float8 *pM1, float8 *pM2, float8 *pMr);
INLINE void MATRIX_vProduct433T(float8 *pM1, float8 *pM2, float8 *pMr);
INLINE void MATRIX_vProductMMT3(float8 *pM1, float8 *pMr);
INLINE void MATRIX_vProduct83T3(float8 *pM1, float8 *pM2, float8 *pMr);
INLINE void MATRIX_vProduct43T3(float8 *pM1, float8 *pM2, float8 *pMr);
INLINE void MATRIX_vProduct33T3(float8 *pM1, float8 *pM2, float8 *pMr);
INLINE void MATRIX_vProduct83T4(float8 *pM1, float8 *pM2, float8 *pMr);
INLINE void MATRIX_vProduct84T3(float8 *pM1, float8 *pM2, float8 *pMr);

// print
INLINE void MATRIX_vPrint(float8 *pM, uint32 u32Lines, uint32 u32Cols);

// trace
INLINE float8 MATRIX_f8Trace(float8 *pM, uint32 lc);
INLINE float8 MATRIX_f8Trace3(float8 *pM);

// determinant
INLINE float8 MATRIX_f8Determinant3(float8 *pM);

// transpose
INLINE void MATRIX_vTranspose(float8 *pM, float8 *pMr, uint32 u32Lines, uint32 u32Cols);
INLINE void MATRIX_vTranspose33(float8 *pM, float8 *pMr);

// norm
INLINE float8 MATRIX_f8Norm1(float8 *pM, uint32 u32lc);
INLINE float8 MATRIX_f8NormInf(float8 *pM, uint32 u32lc);

// inverse
INLINE int8 MATRIX_iInverse3(float8 *pM, float8 *pr);
INLINE int8 MATRIX_iSymInverse3(float8 *pM, float8 *pr);

// sub-matrix extraction
INLINE void MATRIX_vExtractSubMatrix83(float8 *pM, uint32* pu32LineIdx, float8 *pMr);
INLINE void MATRIX_vExtractSubMatrix43(float8 *pM, uint32* pu32LineIdx, float8 *pMr);

// sub-matrix addition
INLINE void MATRIX_vAddSubMatrix83(float8 *pM, float8 *pMa, uint32* pu32LineIdx);
INLINE void MATRIX_vAddSubMatrix43(float8 *pM, float8 *pMa, uint32* pu32LineIdx);

// sub-matrix attribution
INLINE void MATRIX_vAttribSubMatrix83(float8 *pM, float8 *pMa, uint32* pu32LineIdx);

// add I matrix
INLINE void MATRIX_vAddI33(float8 *pM);
INLINE void MATRIX_vAddScalarToDiag33(float8 *pM, float8 x);

// banded matrix handling
#define MATRIX_vSetElementBanded(pMat, u32Band, u32Line, u32Col, Value) pMat[u32Line*u32Band + u32Col - u32Line] = Value 

// solve system of eq
int8 MATRIX_iSolve2(float8 *f8a, float8 *f8b, float8 *f8Res);
int8 MATRIX_iSolve3(float8 *f8a, float8 *f8b, float8 *f8Res);

#endif
