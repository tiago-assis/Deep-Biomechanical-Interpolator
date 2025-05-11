#include <string.h>
#include "matrix.h"

INLINE void MATRIX_vProduct(float8 *pM1, float8 *pM2, float8 *pMr, uint32 u32Lines, uint32 u32Cols1, uint32 u32Cols2)
{
     float8 *p1, *p2, *pr, *pl1, *pc2;
     float8 res;
     uint32 i,j,k;
     pl1 = pM1; pc2 = pM2; pr = pMr;
     for( i = 0; i < u32Lines; i++) 
     {
          for( j = 0; j < u32Cols2; j++) 
          {
               p2 = pc2;     
               p1 = pl1;     
               res = (*p1)*(*p2);
               for( k = 1; k < u32Cols1; k++) 
               {
                    p1++; p2 += u32Cols2;
                    res += (*p1)*(*p2);
               }
               *pr = res;
               pr++;
               pc2++;
          }
          pc2 = pM2;
          pl1+= u32Cols1;    
     }
}

INLINE void MATRIX_vProductMTM(float8 *pM1, float8 *pM2, float8 *pMr, uint32 u32Lines, uint32 u32Cols1, uint32 u32Cols2)
{
     float8 *p1, *p2, *pr, *pl1, *pc2;
     float8 res;
     uint32 i,j,k;
     pl1 = pM1; pc2 = pM2; pr = pMr;
     for( i = 0; i < u32Cols1; i++) 
     {
          for( j = 0; j < u32Cols2; j++) 
          {
               p2 = pc2;     
               p1 = pl1;     
               res = (*p1)*(*p2);
               for( k = 1; k < u32Lines; k++) 
               {
                    p1+= u32Cols1; p2 += u32Cols2;
                    res += (*p1)*(*p2);
               }
               *pr = res;
               pr++;
               pc2++;
          }
          pc2 = pM2;
          pl1++;    
     }
}

INLINE void MATRIX_vProductMMT(float8 *pM1, float8 *pM2, float8 *pMr, uint32 u32Lines1, uint32 u32Cols1, uint32 u32Lines2)
{
     float8 *p1, *p2, *pr, *pl1;
     float8 res;
     uint32 i,j,k;
     pl1 = pM1; p2 = pM2; pr = pMr;
     for( i = 0; i < u32Lines1; i++) 
     {
		  p2 = pM2;     
          for( j = 0; j < u32Lines2; j++) 
          {
               p1 = pl1;     
               res = (*p1)*(*p2);
               for( k = 1; k < u32Cols1; k++) 
               {
                    p1++; p2++;
                    res += (*p1)*(*p2);
               }
               *pr = res;
               pr++;
			   p2++;
          }
          pl1 = p1+1;    
     }
}



INLINE void MATRIX_vProduct333(float8 *pM1, float8 *pM2, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = (*pM1)*(*pM2)+pM1[1]*pM2[3]+pM1[2]*pM2[6];
     pr++;
     *pr = (*pM1)*(pM2[1])+pM1[1]*pM2[4]+pM1[2]*pM2[7];
     pr++;
     *pr = (*pM1)*(pM2[2])+pM1[1]*pM2[5]+pM1[2]*pM2[8];
     pr++;
     *pr = (pM1[3])*(*pM2)+pM1[4]*pM2[3]+pM1[5]*pM2[6];
     pr++;
     *pr = (pM1[3])*(pM2[1])+pM1[4]*pM2[4]+pM1[5]*pM2[7];
     pr++;
     *pr = (pM1[3])*(pM2[2])+pM1[4]*pM2[5]+pM1[5]*pM2[8];
     pr++;
     *pr = (pM1[6])*(*pM2)+pM1[7]*pM2[3]+pM1[8]*pM2[6];
     pr++;
     *pr = (pM1[6])*(pM2[1])+pM1[7]*pM2[4]+pM1[8]*pM2[7];
     pr++;
     *pr = (pM1[6])*(pM2[2])+pM1[7]*pM2[5]+pM1[8]*pM2[8];
}

INLINE void MATRIX_vProduct163(float8 *pM11, float8 *pM21, float8 *pMr1)
{
       float8 pM1[6];
       float8 pM2[6*3];
       float8 pMr[3];
       memcpy(pM1, pM11, 6*sizeof(float8));
       memcpy(pM2, pM21, 18*sizeof(float8));
       pMr[0] = pM1[0]*pM2[0]+pM1[1]*pM2[3]+pM1[2]*pM2[6]+
                pM1[3]*pM2[9]+pM1[4]*pM2[12]+pM1[5]*pM2[15];
       pMr[1] = pM1[0]*pM2[1]+pM1[1]*pM2[4]+pM1[2]*pM2[7]+
                pM1[3]*pM2[10]+pM1[4]*pM2[13]+pM1[5]*pM2[16];
       pMr[2] = pM1[0]*pM2[2]+pM1[1]*pM2[5]+pM1[2]*pM2[8]+
                pM1[3]*pM2[11]+pM1[4]*pM2[14]+pM1[5]*pM2[17];
       memcpy(pMr1, pMr, 3*sizeof(float8));
}        

INLINE void MATRIX_vProduct163Long(float8 *pM1, float8 *pM2, float8 *pMr)
{
       pMr[0] = pM1[0]*pM2[0]+pM1[1]*pM2[3]+pM1[2]*pM2[6]+
                pM1[3]*pM2[9]+pM1[4]*pM2[12]+pM1[5]*pM2[15];
       pMr[1] = pM1[0]*pM2[1]+pM1[1]*pM2[4]+pM1[2]*pM2[7]+
                pM1[3]*pM2[10]+pM1[4]*pM2[13]+pM1[5]*pM2[16];
       pMr[2] = pM1[0]*pM2[2]+pM1[1]*pM2[5]+pM1[2]*pM2[8]+
                pM1[3]*pM2[11]+pM1[4]*pM2[14]+pM1[5]*pM2[17];
}                

INLINE void MATRIX_vProduct333T(float8 *pM1, float8 *pM2, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = pM1[0]*pM2[0]+pM1[1]*pM2[1]+pM1[2]*pM2[2];
     pr++;
     *pr = pM1[0]*pM2[3]+pM1[1]*pM2[4]+pM1[2]*pM2[5];
     pr++;
     *pr = pM1[0]*pM2[6]+pM1[1]*pM2[7]+pM1[2]*pM2[8];
     pr++;
     *pr = pM1[3]*pM2[0]+pM1[4]*pM2[1]+pM1[5]*pM2[2];
     pr++;
     *pr = pM1[3]*pM2[3]+pM1[4]*pM2[4]+pM1[5]*pM2[5];
     pr++;
     *pr = pM1[3]*pM2[6]+pM1[4]*pM2[7]+pM1[5]*pM2[8];
    pr++;
     *pr = pM1[6]*pM2[0]+pM1[7]*pM2[1]+pM1[8]*pM2[2];
     pr++;
     *pr = pM1[6]*pM2[3]+pM1[7]*pM2[4]+pM1[8]*pM2[5];
     pr++;
     *pr = pM1[6]*pM2[6]+pM1[7]*pM2[7]+pM1[8]*pM2[8];
}

INLINE void MATRIX_vProduct433T(float8 *pM1, float8 *pM2, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = pM1[0]*pM2[0]+pM1[1]*pM2[1]+pM1[2]*pM2[2];
     pr++;
     *pr = pM1[0]*pM2[3]+pM1[1]*pM2[4]+pM1[2]*pM2[5];
     pr++;
     *pr = pM1[0]*pM2[6]+pM1[1]*pM2[7]+pM1[2]*pM2[8];
     pr++;
     *pr = pM1[3]*pM2[0]+pM1[4]*pM2[1]+pM1[5]*pM2[2];
     pr++;
     *pr = pM1[3]*pM2[3]+pM1[4]*pM2[4]+pM1[5]*pM2[5];
     pr++;
     *pr = pM1[3]*pM2[6]+pM1[4]*pM2[7]+pM1[5]*pM2[8];
    pr++;
     *pr = pM1[6]*pM2[0]+pM1[7]*pM2[1]+pM1[8]*pM2[2];
     pr++;
     *pr = pM1[6]*pM2[3]+pM1[7]*pM2[4]+pM1[8]*pM2[5];
     pr++;
     *pr = pM1[6]*pM2[6]+pM1[7]*pM2[7]+pM1[8]*pM2[8];
	  pr++;
     *pr = pM1[9]*pM2[0]+pM1[10]*pM2[1]+pM1[11]*pM2[2];
     pr++;
     *pr = pM1[9]*pM2[3]+pM1[10]*pM2[4]+pM1[11]*pM2[5];
     pr++;
     *pr = pM1[9]*pM2[6]+pM1[10]*pM2[7]+pM1[11]*pM2[8];
}

INLINE void MATRIX_vProductMMT3(float8 *pM1, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = pM1[0]*pM1[0]+pM1[1]*pM1[1]+pM1[2]*pM1[2];
     pr++;
     *pr = pM1[0]*pM1[3]+pM1[1]*pM1[4]+pM1[2]*pM1[5];
     pr++;
     *pr = pM1[0]*pM1[6]+pM1[1]*pM1[7]+pM1[2]*pM1[8];
     pr++;
     *pr = pMr[1];
     pr++;
     *pr = pM1[3]*pM1[3]+pM1[4]*pM1[4]+pM1[5]*pM1[5];
     pr++;
     *pr = pM1[3]*pM1[6]+pM1[4]*pM1[7]+pM1[5]*pM1[8];
    pr++;
     *pr = pMr[2];
     pr++;
     *pr = pMr[5];
     pr++;
     *pr = pM1[6]*pM1[6]+pM1[7]*pM1[7]+pM1[8]*pM1[8];
}

INLINE void MATRIX_vProduct83T3(float8 *pM1, float8 *pM2, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = pM1[0]*pM2[0]+pM1[3]*pM2[3]+pM1[6]*pM2[6]+pM1[9]*pM2[9]+
           pM1[12]*pM2[12]+pM1[15]*pM2[15]+pM1[18]*pM2[18]+pM1[21]*pM2[21];
     pr++;
     *pr = pM1[0]*pM2[1]+pM1[3]*pM2[4]+pM1[6]*pM2[7]+pM1[9]*pM2[10]+
           pM1[12]*pM2[13]+pM1[15]*pM2[16]+pM1[18]*pM2[19]+pM1[21]*pM2[22];
     pr++;
     *pr = pM1[0]*pM2[2]+pM1[3]*pM2[5]+pM1[6]*pM2[8]+pM1[9]*pM2[11]+
           pM1[12]*pM2[14]+pM1[15]*pM2[17]+pM1[18]*pM2[20]+pM1[21]*pM2[23];
     pr++;
     *pr = pM1[1]*pM2[0]+pM1[4]*pM2[3]+pM1[7]*pM2[6]+pM1[10]*pM2[9]+
           pM1[13]*pM2[12]+pM1[16]*pM2[15]+pM1[19]*pM2[18]+pM1[22]*pM2[21];
     pr++;
     *pr = pM1[1]*pM2[1]+pM1[4]*pM2[4]+pM1[7]*pM2[7]+pM1[10]*pM2[10]+
           pM1[13]*pM2[13]+pM1[16]*pM2[16]+pM1[19]*pM2[19]+pM1[22]*pM2[22];
     pr++;
     *pr = pM1[1]*pM2[2]+pM1[4]*pM2[5]+pM1[7]*pM2[8]+pM1[10]*pM2[11]+
           pM1[13]*pM2[14]+pM1[16]*pM2[17]+pM1[19]*pM2[20]+pM1[22]*pM2[23];
     pr++;
     *pr = pM1[2]*pM2[0]+pM1[5]*pM2[3]+pM1[8]*pM2[6]+pM1[11]*pM2[9]+
           pM1[14]*pM2[12]+pM1[17]*pM2[15]+pM1[20]*pM2[18]+pM1[23]*pM2[21];
     pr++;
     *pr = pM1[2]*pM2[1]+pM1[5]*pM2[4]+pM1[8]*pM2[7]+pM1[11]*pM2[10]+
           pM1[14]*pM2[13]+pM1[17]*pM2[16]+pM1[20]*pM2[19]+pM1[23]*pM2[22];
     pr++;
     *pr = pM1[2]*pM2[2]+pM1[5]*pM2[5]+pM1[8]*pM2[8]+pM1[11]*pM2[11]+
           pM1[14]*pM2[14]+pM1[17]*pM2[17]+pM1[20]*pM2[20]+pM1[23]*pM2[23];
}

INLINE void MATRIX_vProduct33T3(float8 *pM1, float8 *pM2, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = pM1[0]*pM2[0]+pM1[3]*pM2[3]+pM1[6]*pM2[6];
     pr++;
     *pr = pM1[0]*pM2[1]+pM1[3]*pM2[4]+pM1[6]*pM2[7];
     pr++;
     *pr = pM1[0]*pM2[2]+pM1[3]*pM2[5]+pM1[6]*pM2[8];
     pr++;
     *pr = pM1[1]*pM2[0]+pM1[4]*pM2[3]+pM1[7]*pM2[6];
     pr++;
     *pr = pM1[1]*pM2[1]+pM1[4]*pM2[4]+pM1[7]*pM2[7];
     pr++;
     *pr = pM1[1]*pM2[2]+pM1[4]*pM2[5]+pM1[7]*pM2[8];
     pr++;
     *pr = pM1[2]*pM2[0]+pM1[5]*pM2[3]+pM1[8]*pM2[6];
     pr++;
     *pr = pM1[2]*pM2[1]+pM1[5]*pM2[4]+pM1[8]*pM2[7];
     pr++;
     *pr = pM1[2]*pM2[2]+pM1[5]*pM2[5]+pM1[8]*pM2[8];
}

INLINE void MATRIX_vProduct43T3(float8 *pM1, float8 *pM2, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = pM1[0]*pM2[0]+pM1[3]*pM2[3]+pM1[6]*pM2[6]+pM1[9]*pM2[9];
     pr++;
     *pr = pM1[0]*pM2[1]+pM1[3]*pM2[4]+pM1[6]*pM2[7]+pM1[9]*pM2[10];
     pr++;
     *pr = pM1[0]*pM2[2]+pM1[3]*pM2[5]+pM1[6]*pM2[8]+pM1[9]*pM2[11];
     pr++;
     *pr = pM1[1]*pM2[0]+pM1[4]*pM2[3]+pM1[7]*pM2[6]+pM1[10]*pM2[9];
     pr++;
     *pr = pM1[1]*pM2[1]+pM1[4]*pM2[4]+pM1[7]*pM2[7]+pM1[10]*pM2[10];
     pr++;
     *pr = pM1[1]*pM2[2]+pM1[4]*pM2[5]+pM1[7]*pM2[8]+pM1[10]*pM2[11];
     pr++;
     *pr = pM1[2]*pM2[0]+pM1[5]*pM2[3]+pM1[8]*pM2[6]+pM1[11]*pM2[9];
     pr++;
     *pr = pM1[2]*pM2[1]+pM1[5]*pM2[4]+pM1[8]*pM2[7]+pM1[11]*pM2[10];
     pr++;
     *pr = pM1[2]*pM2[2]+pM1[5]*pM2[5]+pM1[8]*pM2[8]+pM1[11]*pM2[11];
}

INLINE void MATRIX_vProduct83T4(float8 *pM1, float8 *pM2, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = pM1[0]*pM2[0]+pM1[3]*pM2[4]+pM1[6]*pM2[8]+pM1[9]*pM2[12]+
           pM1[12]*pM2[16]+pM1[15]*pM2[20]+pM1[18]*pM2[24]+pM1[21]*pM2[28];
     pr++;
     *pr = pM1[0]*pM2[1]+pM1[3]*pM2[5]+pM1[6]*pM2[9]+pM1[9]*pM2[13]+
           pM1[12]*pM2[17]+pM1[15]*pM2[21]+pM1[18]*pM2[25]+pM1[21]*pM2[29];
     pr++;
     *pr = pM1[0]*pM2[2]+pM1[3]*pM2[6]+pM1[6]*pM2[10]+pM1[9]*pM2[14]+
           pM1[12]*pM2[18]+pM1[15]*pM2[22]+pM1[18]*pM2[26]+pM1[21]*pM2[30];
     pr++;
     *pr = pM1[0]*pM2[3]+pM1[3]*pM2[7]+pM1[6]*pM2[11]+pM1[9]*pM2[15]+
           pM1[12]*pM2[19]+pM1[15]*pM2[23]+pM1[18]*pM2[27]+pM1[21]*pM2[31];
     pr++;
     
     *pr = pM1[1]*pM2[0]+pM1[4]*pM2[4]+pM1[7]*pM2[8]+pM1[10]*pM2[12]+
           pM1[13]*pM2[16]+pM1[16]*pM2[20]+pM1[19]*pM2[24]+pM1[22]*pM2[28];
     pr++;
     *pr = pM1[1]*pM2[1]+pM1[4]*pM2[5]+pM1[7]*pM2[9]+pM1[10]*pM2[13]+
           pM1[13]*pM2[17]+pM1[16]*pM2[21]+pM1[19]*pM2[25]+pM1[22]*pM2[29];
     pr++;
     *pr = pM1[1]*pM2[2]+pM1[4]*pM2[6]+pM1[7]*pM2[10]+pM1[10]*pM2[14]+
           pM1[13]*pM2[18]+pM1[16]*pM2[22]+pM1[19]*pM2[26]+pM1[22]*pM2[30];
     pr++;
     *pr = pM1[1]*pM2[3]+pM1[4]*pM2[7]+pM1[7]*pM2[11]+pM1[10]*pM2[15]+
           pM1[13]*pM2[19]+pM1[16]*pM2[23]+pM1[19]*pM2[27]+pM1[22]*pM2[31];
     pr++;
     
     *pr = pM1[2]*pM2[0]+pM1[5]*pM2[4]+pM1[8]*pM2[8]+pM1[11]*pM2[12]+
           pM1[14]*pM2[16]+pM1[17]*pM2[20]+pM1[20]*pM2[24]+pM1[23]*pM2[28];
     pr++;
     *pr = pM1[2]*pM2[1]+pM1[5]*pM2[5]+pM1[8]*pM2[9]+pM1[11]*pM2[13]+
           pM1[14]*pM2[17]+pM1[17]*pM2[21]+pM1[20]*pM2[25]+pM1[23]*pM2[29];
     pr++;
     *pr = pM1[2]*pM2[2]+pM1[5]*pM2[6]+pM1[8]*pM2[10]+pM1[11]*pM2[14]+
           pM1[14]*pM2[18]+pM1[17]*pM2[22]+pM1[20]*pM2[26]+pM1[23]*pM2[30];
     pr++;
     *pr = pM1[2]*pM2[3]+pM1[5]*pM2[7]+pM1[8]*pM2[11]+pM1[11]*pM2[15]+
           pM1[14]*pM2[19]+pM1[17]*pM2[23]+pM1[20]*pM2[27]+pM1[23]*pM2[31];      
}

INLINE void MATRIX_vProduct84T3(float8 *pM1, float8 *pM2, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = pM1[0]*pM2[0]+pM1[4]*pM2[3]+pM1[8]*pM2[6]+pM1[12]*pM2[9]+
           pM1[16]*pM2[12]+pM1[20]*pM2[15]+pM1[24]*pM2[18]+pM1[28]*pM2[21];
     pr++;
     *pr = pM1[0]*pM2[1]+pM1[4]*pM2[4]+pM1[8]*pM2[7]+pM1[12]*pM2[10]+
           pM1[16]*pM2[13]+pM1[20]*pM2[16]+pM1[24]*pM2[19]+pM1[28]*pM2[22];
     pr++;
     *pr = pM1[0]*pM2[2]+pM1[4]*pM2[5]+pM1[8]*pM2[8]+pM1[12]*pM2[11]+
           pM1[16]*pM2[14]+pM1[20]*pM2[17]+pM1[24]*pM2[20]+pM1[28]*pM2[23];
     pr++;
     
     *pr = pM1[1]*pM2[0]+pM1[5]*pM2[3]+pM1[9]*pM2[6]+pM1[13]*pM2[9]+
           pM1[17]*pM2[12]+pM1[21]*pM2[15]+pM1[25]*pM2[18]+pM1[29]*pM2[21];
     pr++;
     *pr = pM1[1]*pM2[1]+pM1[5]*pM2[4]+pM1[9]*pM2[7]+pM1[13]*pM2[10]+
           pM1[17]*pM2[13]+pM1[21]*pM2[16]+pM1[25]*pM2[19]+pM1[29]*pM2[22];
     pr++;
     *pr = pM1[1]*pM2[2]+pM1[5]*pM2[5]+pM1[9]*pM2[8]+pM1[13]*pM2[11]+
           pM1[17]*pM2[14]+pM1[21]*pM2[17]+pM1[25]*pM2[20]+pM1[29]*pM2[23];
     pr++;
     
     *pr = pM1[2]*pM2[0]+pM1[6]*pM2[3]+pM1[10]*pM2[6]+pM1[14]*pM2[9]+
           pM1[18]*pM2[12]+pM1[22]*pM2[15]+pM1[26]*pM2[18]+pM1[30]*pM2[21];
     pr++;
     *pr = pM1[2]*pM2[1]+pM1[6]*pM2[4]+pM1[10]*pM2[7]+pM1[14]*pM2[10]+
           pM1[18]*pM2[13]+pM1[22]*pM2[16]+pM1[26]*pM2[19]+pM1[30]*pM2[22];
     pr++;
     *pr = pM1[2]*pM2[2]+pM1[6]*pM2[5]+pM1[10]*pM2[8]+pM1[14]*pM2[11]+
           pM1[18]*pM2[14]+pM1[22]*pM2[17]+pM1[26]*pM2[20]+pM1[30]*pM2[23];
     pr++;
     
     *pr = pM1[3]*pM2[0]+pM1[7]*pM2[3]+pM1[11]*pM2[6]+pM1[15]*pM2[9]+
           pM1[19]*pM2[12]+pM1[23]*pM2[15]+pM1[27]*pM2[18]+pM1[31]*pM2[21];
     pr++;
     *pr = pM1[3]*pM2[1]+pM1[7]*pM2[4]+pM1[11]*pM2[7]+pM1[15]*pM2[10]+
           pM1[19]*pM2[13]+pM1[23]*pM2[16]+pM1[27]*pM2[19]+pM1[31]*pM2[22];
     pr++;
     *pr = pM1[3]*pM2[2]+pM1[7]*pM2[5]+pM1[11]*pM2[8]+pM1[15]*pM2[11]+
           pM1[19]*pM2[14]+pM1[23]*pM2[17]+pM1[27]*pM2[20]+pM1[31]*pM2[23];
}

INLINE void MATRIX_vPrint(float8 *pM, uint32 u32Lines, uint32 u32Cols)
{
     float8 *p;
     uint32 i, j;
     p = pM;
     for (i = 0; i < u32Lines; i++)
     {
         for (j = 0; j < u32Cols; j++)
         {
             printf("%10.8f    ", *p);
             p++;
         }
         printf("\n");
     }
     printf("\n");
}

INLINE float8 MATRIX_f8Trace(float8 *pM, uint32 lc)
{
       float8 res;
       uint32 i;
       float8 *p;
       p = pM;
       res = *p;
       for (i = 1; i < lc; i++) 
       {
           p+=(lc+1);
           res += *p;
       }
       return res;
}

INLINE float8 MATRIX_f8Trace3(float8 *pM)
{
       return (pM[0] + pM[4] + pM[8]);
}

INLINE float8 MATRIX_f8Determinant3(float8 *pM)
{
	float8 res;
	res = pM[0]*(pM[4]*pM[8]-pM[5]*pM[7])+
		  pM[1]*(pM[5]*pM[6]-pM[3]*pM[8])+
		  pM[2]*(pM[3]*pM[7]-pM[4]*pM[6]);
       return res;
} 

INLINE void MATRIX_vTranspose(float8 *pM, float8 *pMr, uint32 u32Lines, uint32 u32Cols)
{
     float8 *p, *pr, *pc;
     uint32 i, j;
     p = pM;
     pc = pMr; 
     for(i = 0; i < u32Lines; i++)
     {
           pr = pc;
           for(j = 0; j < u32Cols; j++)
           {
                 *pr = *p;
                 p++;
                 pr += u32Lines;
           }
           pc++;
     }
}
           
INLINE void MATRIX_vTranspose33(float8 *pM, float8 *pMr)
{
     float8 *pr;
     pr = pMr;
     *pr = *pM;
     pr++; *pr = pM[3];     
     pr++; *pr = pM[6];
     pr++; *pr = pM[1];
     pr++; *pr = pM[4];
     pr++; *pr = pM[7];
     pr++; *pr = pM[2];
     pr++; *pr = pM[5];
     pr++; *pr = pM[8];
}

INLINE float8 MATRIX_f8Norm1(float8 *pM, uint32 u32lc)
{
       float8 *p, *pc;
       uint32 i,j;
       float8 res, sum;
       res = 0;
       pc = pM;
       for(i = 0; i < u32lc; i++)
       {
             p = pc;
             sum = (*p>0)?*p:(-*p);
             for(j = 1; j < u32lc; j++)
             {
                   //compute the sum
                   p+= u32lc;
                   sum += (*p>0)?*p:(-*p);
             }
             if (sum > res) {res = sum;};
             pc++;
       }
       return res;
}
                   
             
INLINE float8 MATRIX_f8NormInf(float8 *pM, uint32 u32lc)
{
       float8 *p;
       uint32 i,j;
       float8 res, sum;
       res = 0;
       p = pM;
       for(i = 0; i < u32lc; i++)
       {
             sum = (*p>0)?*p:(-*p);
             for(j = 1; j < u32lc; j++)
             {
                   //compute the sum
                   p++;
                   sum += (*p>0)?*p:(-*p);
             }
             if (sum > res) {res = sum;};
             p++;
       }
       return res;
}

// inverse
INLINE int8 MATRIX_iInverse3(float8 *pM, float8 *pr)
{
       float8 f8Det = MATRIX_f8Determinant3(pM);
	   if (f8Det == 0) return -1;
       *pr = (pM[4]*pM[8] - pM[5]*pM[7])/f8Det;
       pr[1] = (pM[2]*pM[7] - pM[1]*pM[8])/f8Det;
       pr[2] = (pM[1]*pM[5] - pM[2]*pM[4])/f8Det;       
       pr[3] = (pM[5]*pM[6] - pM[3]*pM[8])/f8Det;
       pr[4] = (pM[0]*pM[8] - pM[2]*pM[6])/f8Det;       
       pr[5] = (pM[2]*pM[3] - pM[0]*pM[5])/f8Det;
       pr[6] = (pM[3]*pM[7] - pM[4]*pM[6])/f8Det;  
       pr[7] = (pM[1]*pM[6] - pM[0]*pM[7])/f8Det; 
       pr[8] = (pM[0]*pM[4] - pM[1]*pM[3])/f8Det;     
	   return 0;
}

INLINE int8 MATRIX_iSymInverse3(float8 *pM, float8 *pr)
{
       float8 f8Det = MATRIX_f8Determinant3(pM);
	   if (f8Det == 0) return -1;
       *pr = (pM[4]*pM[8] - pM[5]*pM[7])/f8Det;
       pr[1] = (pM[2]*pM[7] - pM[1]*pM[8])/f8Det;
       pr[2] = (pM[1]*pM[5] - pM[2]*pM[4])/f8Det;       
       pr[3] = pr[1];
       pr[4] = (pM[0]*pM[8] - pM[2]*pM[6])/f8Det;       
       pr[5] = (pM[2]*pM[3] - pM[0]*pM[5])/f8Det;
       pr[6] = pr[2];  
       pr[7] = pr[5]; 
       pr[8] = (pM[0]*pM[4] - pM[1]*pM[3])/f8Det; 
	   return 0;
}
       
// sub-matrix extraction
INLINE void MATRIX_vExtractSubMatrix83(float8 *pM, uint32* pu32LineIdx, float8 *pMr)
{
       float8 *pSrc, *pDest;
	   uint32 u32LineIdx;

       u32LineIdx = pu32LineIdx[0];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       pDest = pMr;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[1];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[2];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[3];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[4];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[5];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[6];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[7];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; 
}

// sub-matrix extraction
INLINE void MATRIX_vExtractSubMatrix43(float8 *pM, uint32* pu32LineIdx, float8 *pMr)
{
       float8 *pSrc, *pDest;
	   uint32 u32LineIdx;

       u32LineIdx = pu32LineIdx[0];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       pDest = pMr;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[1];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[2];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++;
       u32LineIdx = pu32LineIdx[3];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc; pDest++; pSrc++; *pDest = *pSrc;
}


// sub-matrix addition
INLINE void MATRIX_vAddSubMatrix83(float8 *pM, float8 *pMa, uint32* pu32LineIdx)
{
       float8 *pSrc;
       float8 *pAdd;
	   uint32 u32LineIdx;
       pAdd = pMa;
	   u32LineIdx = pu32LineIdx[0];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[1];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[2];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[3];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[4];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[5];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[6];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[7];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd);
}

INLINE void MATRIX_vAddSubMatrix43(float8 *pM, float8 *pMa, uint32* pu32LineIdx)
{
       float8 *pSrc;
       float8 *pAdd;
	   uint32 u32LineIdx;
       pAdd = pMa;
	   u32LineIdx = pu32LineIdx[0];
	   u32LineIdx += (u32LineIdx << 1);
       pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[1];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[2];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++;
       u32LineIdx = pu32LineIdx[3];
	   u32LineIdx += (u32LineIdx << 1);
	   pSrc = pM + u32LineIdx;
       *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd); pAdd++; pSrc++; *pSrc+=(*pAdd);
}

// sub-matrix attribution
INLINE void MATRIX_vAttribSubMatrix83(float8 *pM, float8 *pMa, uint32* pu32LineIdx)
{
       float8 *pSrc;
       float8 *pAdd;
       pAdd = pMa;
       pSrc = pM + 3*pu32LineIdx[0];
       *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++;
       pSrc = pM + 3*pu32LineIdx[1];
       *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++;
       pSrc = pM + 3*pu32LineIdx[2];
       *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++;
       pSrc = pM + 3*pu32LineIdx[3];
       *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++;
       pSrc = pM + 3*pu32LineIdx[4];
       *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++;
       pSrc = pM + 3*pu32LineIdx[5];
       *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++;
       pSrc = pM + 3*pu32LineIdx[6];
       *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++;
       pSrc = pM + 3*pu32LineIdx[7];
       *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd); pAdd++; pSrc++; *pSrc=(*pAdd);
}

// add I matrix
INLINE void MATRIX_vAddI33(float8 *pM)
{
       pM[0]++;
       pM[4]++;
       pM[8]++;
}

INLINE void MATRIX_vAddScalarToDiag33(float8 *pM, float8 x)
{
      pM[0]+=x;
      pM[4]+=x;
      pM[8]+=x;        
}
       
int8 MATRIX_iSolve2(float8 *f8a, float8 *f8b, float8 *f8Res)
{
	float8 det = f8a[0]*f8a[3] - f8a[1]*f8a[2];
	if (det == 0) return -1;
	f8Res[0] = (f8b[0]*f8a[3] - f8b[1]*f8a[1])/det;
	f8Res[1] = (f8a[0]*f8b[1] - f8a[2]*f8b[0])/det;
	return 0;
}

int8 MATRIX_iSolve3(float8 *f8a, float8 *f8b, float8 *f8Res)
{
	float8 af8Inv[9];
	if (MATRIX_iInverse3(f8a, af8Inv) < 0) return -1;
	f8Res[0] = f8b[0]*af8Inv[0] + f8b[1]*af8Inv[1] + f8b[2]*af8Inv[2];
	f8Res[1] = f8b[0]*af8Inv[3] + f8b[1]*af8Inv[4] + f8b[2]*af8Inv[5];
	f8Res[2] = f8b[0]*af8Inv[6] + f8b[1]*af8Inv[7] + f8b[2]*af8Inv[8];
	return 0;
}
