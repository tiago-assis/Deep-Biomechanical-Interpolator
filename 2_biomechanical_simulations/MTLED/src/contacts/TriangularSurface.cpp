#include <stdio.h>
#include <math.h>
#include "vect.h"
#include "matrix.h"
#include "TriangularSurface.h"


CNode::CNode()
{
	af8Coordinates[0] = 0;
	af8Coordinates[1] = 0;
	af8Coordinates[2] = 0;
}

CNode::CNode(float8 *pf8Coord)
{
	af8Coordinates[0] = pf8Coord[0];
	af8Coordinates[1] = pf8Coord[1];
	af8Coordinates[2] = pf8Coord[2];
}

CNode CNode::operator+(CNode node)
{
	CNode result;
	VECT_vAdd(this->af8Coordinates, node.af8Coordinates, result.af8Coordinates, 3);
	return result;
}

CNode CNode::operator-(CNode node)
{
	CNode result;
	VECT_vSubstract(this->af8Coordinates, node.af8Coordinates, result.af8Coordinates, 3);
	return result;
}

CNode CNode::operator-()
{
	CNode result;
	result.af8Coordinates[0] = -this->af8Coordinates[0];
	result.af8Coordinates[1] = -this->af8Coordinates[1];
	result.af8Coordinates[2] = -this->af8Coordinates[2];
	return result;
}

CNode operator*(CNode node, float8 f8x)
{
	CNode result;
	VECT_vMultiplyScalar(node.af8Coordinates, f8x, result.af8Coordinates, 3);
	return result;
}

CNode operator*(float8 f8x, CNode node)
{
	CNode result;
	VECT_vMultiplyScalar(node.af8Coordinates, f8x, result.af8Coordinates, 3);
	return result;
}

float8 f8ScalarProduct(CNode node1, CNode node2)
{
	return VECT_f8ScalarProduct(node1.af8Coordinates, node2.af8Coordinates, 3);
}

void CNode::vSaveBinary(FILE *pf)
{
	fwrite(af8Coordinates, sizeof(float8), 3, pf);
}

CMyNode::CMyNode():CNode()
{
	boOnSurfaceEdge = false;
}

CMyNode::CMyNode(float8 *pf8Coord):CNode(pf8Coord)
{
	boOnSurfaceEdge = false;
}

CMyNode::~CMyNode(void)
{
	au32AdiacentTrianglesIndexes.clear();
	au8PozitionInAdiacentTriangles.clear();
	au32AdiacentSegmentsIndexes.clear();
	au8PozitionInAdiacentSegments.clear();
	au32AditionalSegmentsToCheck.clear();
	au32AditionalTrianglesToCheck.clear();
}

void CMyNode::vComputeNormal(CTriangleArray *pTriangles)
{
	float8 af8Normal[3] = {0, 0, 0};
	// add normals of surounding triangles
	for (uint32 j = 0; j < au32AdiacentTrianglesIndexes.size(); j++)
	{
		float8 *pf8N = pTriangles->ElementAt(au32AdiacentTrianglesIndexes.at(j))->af8NormalComponents;
		VECT_vAdd(pf8N, af8Normal, af8Normal, 3);
	}
	// normalize result
	float8 f8Norm = VECT_f8ScalarProduct(af8Normal, af8Normal, 3);
	VECT_vDivideScalar(af8Normal, sqrt(f8Norm), af8NormalComponents, 3);
}

void CMyNode::vGetAdiacentSegments(CSegmentArray *pSegments, uint32 u32NodeIdx)
{
	uint32 i;
	CSegment *pS;
	uint32 u32NumSegments = pSegments->u32GetCount();
	// save adiacent segments information
	for (i = 0; i < u32NumSegments; i++)
	{
		pS = pSegments->ElementAt(i);
		int8 i8Poz = pS->i8ContainsNode(u32NodeIdx);
		if (i8Poz >= 0) 
		{
			au32AdiacentSegmentsIndexes.push_back(i);
			au8PozitionInAdiacentSegments.push_back((uint8)i8Poz);
			if (pS->boOnSurfaceEdge) boOnSurfaceEdge = true;
		}
	}
}

void CMyNode::vCleanUpAditionalCheckLists(CSegmentArray *pSegments)
{
	// search segments in the aditional segment to check list and remove their neighbouring
	// triangles from the aditional triangles to check list
	int32 T1;
	uint32 i;
	uint32 u32Num = (uint32)au32AditionalSegmentsToCheck.size();
	for (i = 0; i < u32Num; i++)
	{
		uint32 u32SegIdx = (uint32)au32AditionalSegmentsToCheck.at(i);
		CSegment *pSeg = pSegments->ElementAt(u32SegIdx);
		T1 = pSeg->ai32AdiacentTrianglesIdx[0];
		vRemoveTriangleToCheck(T1);
		T1 = pSeg->ai32AdiacentTrianglesIdx[1];
		if (T1 >= 0) 
			vRemoveTriangleToCheck(T1);
	}
}

void CMyNode::vRemoveTriangleToCheck(uint32 u32T)
{
	uint32 u32Num = (uint32)au32AditionalTrianglesToCheck.size();
	for (uint32 i = 0; i < u32Num; i++)
	{
		if (au32AditionalTrianglesToCheck.at(i) == u32T)
		{
			au32AditionalTrianglesToCheck.erase(au32AditionalTrianglesToCheck.begin()+i);
			return;
		}
	}
}


CSegment::CSegment()
{
	ai32AdiacentTrianglesIdx[0] = -1;
	ai32AdiacentTrianglesIdx[1] = -1;
	boOnSurfaceEdge = true;
}

void CSegment::vSetNodeIndexes(uint32 u32N1, uint32 u32N2)
{
	au32NodeIdxs[0] = u32N1;
	au32NodeIdxs[1] = u32N2;
}

int8 CSegment::i8ContainsNode(uint32 u32NodeIdx)
{
	if (au32NodeIdxs[0] == u32NodeIdx) return 0;
	if (au32NodeIdxs[1] == u32NodeIdx) return 1;
	return -1;
}


void CSegment::vComputeVectorComponents(CNodeArray *pNodes)
{
	CMyNode *pN1 = pNodes->ElementAt(au32NodeIdxs[0]);
	CMyNode *pN2 = pNodes->ElementAt(au32NodeIdxs[1]);
	CNode n = *pN2-*pN1;
	VECT_vCopy(n.af8Coordinates, af8VectorComponents, 3);
	f8Length = sqrt(VECT_f8ScalarProduct(af8VectorComponents, af8VectorComponents, 3));
}

void CSegment::vComputeNormal(CTriangleArray *pTriangles)
{
	// compute normal as sum of surrounding triangles normals
	float8 *pf8N1 = pTriangles->ElementAt(ai32AdiacentTrianglesIdx[0])->af8NormalComponents;
	if (ai32AdiacentTrianglesIdx[1] >= 0)
	{
		float8 *pf8N2 = pTriangles->ElementAt(ai32AdiacentTrianglesIdx[1])->af8NormalComponents;
		VECT_vAdd(pf8N1, pf8N2, af8NormalComponents, 3);
		// normalize result
		float8 f8Norm = VECT_f8ScalarProduct(af8NormalComponents, af8NormalComponents, 3);
		VECT_vDivideScalar(af8NormalComponents, sqrt(f8Norm), af8NormalComponents, 3);
	}
	else
	{
		VECT_vCopy(pf8N1, af8NormalComponents, 3);
	}	
}

bool CSegment::boClosestPointCoord(CNodeArray *pNodes, CNode *pNode, float8 *r)
{
	uint32 u32N1 = au32NodeIdxs[0];
	uint32 u32N2 = au32NodeIdxs[1];
	CNode &n1 = *pNodes->ElementAt(u32N1);
	CNode &n2 = *pNodes->ElementAt(u32N2);
	CNode v2P = *pNode - n2;
	CNode v21 = n1 - n2;
	*r = f8ScalarProduct(v2P, v21)/f8ScalarProduct(v21, v21);
	if ((*r >= 0) && (*r <= 1)) return true;
	return false;
}


CTriangle::CTriangle(uint32 u32N1, uint32 u32N2, uint32 u32N3, CNodeArray *pNodeArr)
{
	au32NodeIndex[0] = u32N1;
	au32NodeIndex[1] = u32N2;
	au32NodeIndex[2] = u32N3;
	pNodeArray = pNodeArr;
	// compute the circumscribed triangle information
	vComputeCircumscribedCircle();
}

CTriangle::~CTriangle(void)
{

}

void CTriangle::vComputeCircumscribedCircle(void)
{
	CNode n1 = *pNodeArray->ElementAt(au32NodeIndex[0]);
	CNode n2 = *pNodeArray->ElementAt(au32NodeIndex[1]);
	CNode n3 = *pNodeArray->ElementAt(au32NodeIndex[2]);
	CNode d12 = n1 - n2;
	CNode d31 = n3 - n1;
	CNode d23 = n2 - n3;
	float8 f8a[4];
	f8a[0] = 2*f8ScalarProduct(d12, d31);
	f8a[1] = -2*f8ScalarProduct(d12, d23);
	f8a[2] = 2*f8ScalarProduct(d31, d31);
	f8a[3] = -2*f8ScalarProduct(d31, d23);
	CNode d123 = d23 - d31;
	float8 f8b[2];
	f8b[0] = -f8ScalarProduct(d12, d123);
	f8b[1] = f8ScalarProduct(d31, d31);
	float8 f8rs[2];
	MATRIX_iSolve2(f8a, f8b, f8rs);
	CircCircleCenter = f8rs[0]*n1 + f8rs[1]*n2 + (1 - f8rs[0] - f8rs[1])*n3;
	CNode do1 = CircCircleCenter - n1;
	f8CircCircleRadius = sqrt(f8ScalarProduct(do1, do1));
}

void CTriangle::vComputeBisector(uint8 u8NodePoz, float8 *pf8Bisector)
{
	CNode n1 = *pNodeArray->ElementAt(au32NodeIndex[u8NodePoz]);
	CNode n2 = *pNodeArray->ElementAt(au32NodeIndex[(u8NodePoz+1)%3]);
	CNode n3 = *pNodeArray->ElementAt(au32NodeIndex[(u8NodePoz+2)%3]);
	CNode d13 = n3 - n1;
	CNode d32 = n2 - n3;
	CNode d12 = n2 - n1;
	float8 f8L13 = sqrt(f8ScalarProduct(d13, d13));
	float8 f8L12 = sqrt(f8ScalarProduct(d12, d12));
	CNode v = d13 + (f8L13/(f8L13+f8L12))*d32;
	VECT_vCopy(v.af8Coordinates, pf8Bisector, 3);
}

bool CTriangle::boIsOptuse(uint8 u8NodePoz)
{
	CNode n1 = *pNodeArray->ElementAt(au32NodeIndex[u8NodePoz]);
	CNode n2 = *pNodeArray->ElementAt(au32NodeIndex[(u8NodePoz+1)%3]);
	CNode n3 = *pNodeArray->ElementAt(au32NodeIndex[(u8NodePoz+2)%3]);
	CNode d13 = n3 - n1;
	CNode d12 = n2 - n1;
	if (f8ScalarProduct(d12, d13) > 0) return false;
	return true;
}


void CTriangle::vGetMaximumSegmentProjection(float8 *pf8ProjOnX, float8 *pf8ProjOnY, float8 *pf8ProjOnZ)
{
	CNode n1 = *pNodeArray->ElementAt(au32NodeIndex[0]);
	CNode n2 = *pNodeArray->ElementAt(au32NodeIndex[1]);
	CNode n3 = *pNodeArray->ElementAt(au32NodeIndex[2]);
	CNode d13 = n3 - n1;
	CNode d32 = n2 - n3;
	CNode d12 = n2 - n1;
	*pf8ProjOnX = VECT_Abs(d13.af8Coordinates[0]);
	if (VECT_Abs(d32.af8Coordinates[0]) > *pf8ProjOnX) *pf8ProjOnX = VECT_Abs(d32.af8Coordinates[0]);
	if (VECT_Abs(d12.af8Coordinates[0]) > *pf8ProjOnX) *pf8ProjOnX = VECT_Abs(d12.af8Coordinates[0]);
	*pf8ProjOnY = VECT_Abs(d13.af8Coordinates[1]);
	if (VECT_Abs(d32.af8Coordinates[1]) > *pf8ProjOnY) *pf8ProjOnY = VECT_Abs(d32.af8Coordinates[1]);
	if (VECT_Abs(d12.af8Coordinates[1]) > *pf8ProjOnY) *pf8ProjOnY = VECT_Abs(d12.af8Coordinates[1]);
	*pf8ProjOnZ = VECT_Abs(d13.af8Coordinates[2]);
	if (VECT_Abs(d32.af8Coordinates[2]) > *pf8ProjOnZ) *pf8ProjOnZ = VECT_Abs(d32.af8Coordinates[2]);
	if (VECT_Abs(d12.af8Coordinates[2]) > *pf8ProjOnZ) *pf8ProjOnZ = VECT_Abs(d12.af8Coordinates[2]);
}

void CTriangle::vGetBoundingBox(float8 &f8MinX, float8 &f8MinY, float8 &f8MinZ, float8 &f8MaxX, float8 &f8MaxY, float8 &f8MaxZ)
{
	CNode n1 = *pNodeArray->ElementAt(au32NodeIndex[0]);
	CNode n2 = *pNodeArray->ElementAt(au32NodeIndex[1]);
	CNode n3 = *pNodeArray->ElementAt(au32NodeIndex[2]);
	f8MinX = n1.af8Coordinates[0]; f8MinY = n1.af8Coordinates[1]; f8MinZ = n1.af8Coordinates[2];
	f8MaxX = n1.af8Coordinates[0]; f8MaxY = n1.af8Coordinates[1]; f8MaxZ = n1.af8Coordinates[2];
	if (n2.af8Coordinates[0] < f8MinX) f8MinX = n2.af8Coordinates[0];
	else if (n2.af8Coordinates[0] > f8MaxX) f8MaxX = n2.af8Coordinates[0];
	if (n3.af8Coordinates[0] < f8MinX) f8MinX = n3.af8Coordinates[0];
	else if (n3.af8Coordinates[0] > f8MaxX) f8MaxX = n3.af8Coordinates[0];

	if (n2.af8Coordinates[1] < f8MinY) f8MinY = n2.af8Coordinates[1];
	else if (n2.af8Coordinates[1] > f8MaxY) f8MaxY = n2.af8Coordinates[1];
	if (n3.af8Coordinates[1] < f8MinY) f8MinY = n3.af8Coordinates[1];
	else if (n3.af8Coordinates[1] > f8MaxY) f8MaxY = n3.af8Coordinates[1];

	if (n2.af8Coordinates[2] < f8MinZ) f8MinZ = n2.af8Coordinates[2];
	else if (n2.af8Coordinates[2] > f8MaxZ) f8MaxZ = n2.af8Coordinates[2];
	if (n3.af8Coordinates[2] < f8MinZ) f8MinZ = n3.af8Coordinates[2];
	else if (n3.af8Coordinates[2] > f8MaxZ) f8MaxZ = n3.af8Coordinates[2];
}

void CTriangle::vSetSegmentIndexes(uint32 u32S1, uint32 u32S2, uint32 u32S3)
{
	au32SegmentIdx[0] = u32S1;
	au32SegmentIdx[1] = u32S2;
	au32SegmentIdx[2] = u32S3;
}

void CTriangle::vSaveBinary(FILE *pf)
{
	fwrite(au32NodeIndex, sizeof(uint32), 3, pf);
}

int8 CTriangle::i8ContainsNode(uint32 u32NodeIdx)
{
	if (au32NodeIndex[0] == u32NodeIdx) return 0;
	if (au32NodeIndex[1] == u32NodeIdx) return 1;
	if (au32NodeIndex[2] == u32NodeIdx) return 2;
	return -1;
}

bool CTriangle::boIsCoplanar(CMyNode *pNode)
{
	if (f8Distance(pNode) < 0.000001) return true;
	return false;
}

float8 CTriangle::f8Distance(CMyNode *pNode)
{
	float8 r,s;
	boClosestPointCoord(pNode, &r, &s);
	CNode P = r*(*pNodeArray->ElementAt(au32NodeIndex[0])) + 
		s*(*pNodeArray->ElementAt(au32NodeIndex[1])) +
		(1-s-r)*(*pNodeArray->ElementAt(au32NodeIndex[2]));
	CNode PT = P - *pNode;
	return sqrt(f8ScalarProduct(PT, PT));
}

float8 CTriangle::f8Area(void)
{
	float8 s, a, c, b;
	CMyNode &P1 = *pNodeArray->ElementAt(au32NodeIndex[0]);
	CMyNode &P2 = *pNodeArray->ElementAt(au32NodeIndex[1]);
	CMyNode &P3 = *pNodeArray->ElementAt(au32NodeIndex[2]);
	CNode d13 = P1 - P3;
	CNode d23 = P2 - P3;
	CNode d12 = P1 - P2;
	a = f8ScalarProduct(d13, d13);
	b = f8ScalarProduct(d23, d23);
	c = f8ScalarProduct(d12, d12);
	s = (a+b+c)/2;
	return sqrt(s*(s-a)*(s-b)*(s-c));
}

float8 CTriangle::f8SegmentIntersection(CNode *pN1, CNode *pN2)
{
	float8 r, s, t;
	CMyNode &P1 = *pNodeArray->ElementAt(au32NodeIndex[0]);
	CMyNode &P2 = *pNodeArray->ElementAt(au32NodeIndex[1]);
	CMyNode &P3 = *pNodeArray->ElementAt(au32NodeIndex[2]);
	// create the system of equations
	CNode ar = *pN1 - *pN2;
	CNode as = P3 - P1;
	CNode at = P3 - P2;
	CNode rhs = P3 - *pN2;
	float8 af8A[9];
	float8 af8AInv[9];
	af8A[0] = ar.af8Coordinates[0];
	af8A[1] = as.af8Coordinates[0];
	af8A[2] = at.af8Coordinates[0];
	af8A[3] = ar.af8Coordinates[1];
	af8A[4] = as.af8Coordinates[1];
	af8A[5] = at.af8Coordinates[1];
	af8A[6] = ar.af8Coordinates[2];
	af8A[7] = as.af8Coordinates[2];
	af8A[8] = at.af8Coordinates[2];
	int8 res = MATRIX_iInverse3(af8A, af8AInv);
	if (res < 0) return res;
	MATRIX_vProduct(af8AInv, rhs.af8Coordinates, af8A, 3, 3, 1);
	r = af8A[0]; s = af8A[1]; t = af8A[2];
	if ((r < 0) || (r > 1)) return -1;
	if ((s < 0) || (s > 1)) return -1;
	if ((t < 0) || (t > 1)) return -1;
	if (s+t > 1) return -1;
	return r;
}

bool CTriangle::boClosestPointCoord(CNode *pNode, float8 *x, float8 *y)
{
	float8 r, s;
	CMyNode &P1 = *pNodeArray->ElementAt(au32NodeIndex[0]);
	CMyNode &P2 = *pNodeArray->ElementAt(au32NodeIndex[1]);
	CMyNode &P3 = *pNodeArray->ElementAt(au32NodeIndex[2]);
	CNode d13 = P1 - P3;
	CNode d23 = P2 - P3;
	CNode dM3 = *pNode - P3;
	float8 f8a[4];
	f8a[0] = f8ScalarProduct(d13, d13);
	f8a[1] = f8ScalarProduct(d23, d13);
	f8a[2]= f8a[1];
	f8a[3] = f8ScalarProduct(d23, d23);
	float8 f8b[2];
	f8b[0] = f8ScalarProduct(d13, dM3);
	f8b[1] = f8ScalarProduct(d23, dM3);
	float8 f8r[2];
	MATRIX_iSolve2(f8a, f8b, f8r);
	r = f8r[0]; 
	*x = r;
	s = f8r[1]; 
	*y = s;
	if ((r >= 0) && (r <= 1) &&
		(s >= 0) && (s <= 1) &&
		(r+s <= 1))
		return true;
	return false;
}

CNode CTriangle::CGetCentreSphere(CNode *pNode)
{
	CMyNode &P1 = *pNodeArray->ElementAt(au32NodeIndex[0]);
	CMyNode &P2 = *pNodeArray->ElementAt(au32NodeIndex[1]);
	CMyNode &P3 = *pNodeArray->ElementAt(au32NodeIndex[2]);
	CNode v12 = P2 - P1;
	CNode v13 = P3 - P1;
	CNode v14 = *pNode - P1;
	CNode s12 = 0.5*(P2+P1);
	CNode s13 = 0.5*(P3+P1);
	CNode s14 = 0.5*(*pNode+P1);
	float8 af8a[9];
	float8 af8b[3];
	float8 af8c[3];
	VECT_vCopy(v12.af8Coordinates, af8a, 3);
	VECT_vCopy(v13.af8Coordinates, af8a+3, 3);
	VECT_vCopy(v14.af8Coordinates, af8a+6, 3);
	af8b[0] = f8ScalarProduct(v12, s12);
	af8b[1] = f8ScalarProduct(v13, s13);
	af8b[2] = f8ScalarProduct(v14, s14);
	MATRIX_iSolve3(af8a, af8b, af8c);
	CNode n(af8c);
	return n;
}

void CTriangle::vComputeNormal(CSegmentArray *pSegments)
{
	uint8 u8O1 = au8SegmentOrientation[0];
	uint8 u8O2 = au8SegmentOrientation[1];
	float8 *f8V1 = pSegments->ElementAt(au32SegmentIdx[0])->af8VectorComponents;
	float8 *f8V2 = pSegments->ElementAt(au32SegmentIdx[1])->af8VectorComponents;

	// vectorial product between the vectors = normal to triangle
	if (u8O1 == u8O2)
	{
		af8NormalComponents[0] = f8V1[1]*f8V2[2]-f8V1[2]*f8V2[1];
		af8NormalComponents[1] = f8V1[2]*f8V2[0]-f8V1[0]*f8V2[2];
		af8NormalComponents[2] = f8V1[0]*f8V2[1]-f8V1[1]*f8V2[0];
	}
	else
	{
		af8NormalComponents[0] = -f8V1[1]*f8V2[2]+f8V1[2]*f8V2[1];
		af8NormalComponents[1] = -f8V1[2]*f8V2[0]+f8V1[0]*f8V2[2];
		af8NormalComponents[2] = -f8V1[0]*f8V2[1]+f8V1[1]*f8V2[0];
	}
	// normalize result
	float8 f8Norm = VECT_f8ScalarProduct(af8NormalComponents, af8NormalComponents, 3);
	VECT_vDivideScalar(af8NormalComponents, sqrt(f8Norm), af8NormalComponents, 3);
}


bool CTriangle::boPossibleInfluenceZone(uint32 u32NodeIdx, float8 f8MaxPossiblePenetration)
{
#if 1
	CNode n(af8NormalComponents); // the normal to the triangle surface 
	// check that the direction of the normal is towards the node C
	CNode &T1 = *pNodeArray->ElementAt(au32NodeIndex[0]);
	CNode &T2 = *pNodeArray->ElementAt(au32NodeIndex[1]);
	CNode &T3 = *pNodeArray->ElementAt(au32NodeIndex[2]);
	CNode &C = *pNodeArray->ElementAt(u32NodeIdx);
	CNode T1C = C - T1;
	float8 f8L = sqrt(f8ScalarProduct(n, n));
	if (f8ScalarProduct(T1C, n) > 0)
		VECT_vMultiplyScalar(n.af8Coordinates, f8MaxPossiblePenetration*f8L, n.af8Coordinates, 3);
	else
		VECT_vMultiplyScalar(n.af8Coordinates, -f8MaxPossiblePenetration*f8L, n.af8Coordinates, 3);
	CNode D1, D2, D3;
	D1 = T1+n;
	D2 = T2+n;
	D3 = T3+n;
	CNode O1 = CircCircleCenter + n; 
	float8 r,s;
	bool boOInside = boClosestPointCoord(&O1, &r, &s);
	// construct an array of nodes that must be checked
	uint32 u32NumNodesToCheck = 6;
	if (boOInside) u32NumNodesToCheck++;
	CNode *pNodesToCheck = new CNode[u32NumNodesToCheck];
	pNodesToCheck[0] = D1; pNodesToCheck[1] = D2; pNodesToCheck[2] = D3;
	pNodesToCheck[3] = 0.5*(D1+D2); pNodesToCheck[4] = 0.5*(D2+D3); pNodesToCheck[5] = 0.5*(D3+D1);
	if (boOInside) pNodesToCheck[6] = O1;
	// points on the planes limiting the influence zone
	CNode M1 = 0.5*(C+T1);
	CNode M2 = 0.5*(C+T2);
	CNode M3 = 0.5*(C+T3);
	bool boInfluenceFound = false;
	for (uint32 i = 0; i < u32NumNodesToCheck; i++)
	{
		CNode &S = pNodesToCheck[i];
		if ((f8ScalarProduct(S - M1, C - M1) > 0) &&
			(f8ScalarProduct(S - M2, C - M2) > 0) &&
			(f8ScalarProduct(S - M3, C - M3) > 0))
		{
			boInfluenceFound = true;
			break;
		}
	}
	delete[] pNodesToCheck;
	return boInfluenceFound;

#else
	bool boU1, boU2, boU3; // check for angles >= 90;
	float8 r,s;
	CMyNode &P1 = *pNodeArray->ElementAt(au32NodeIndex[0]);
	CMyNode &P2 = *pNodeArray->ElementAt(au32NodeIndex[1]);
	CMyNode &P3 = *pNodeArray->ElementAt(au32NodeIndex[2]);
	CMyNode &node = *pNodeArray->ElementAt(u32NodeIdx);
	if (boIsCoplanar(&node)) return true;
	CNode v12 = P2 - P1;
	CNode v23 = P3 - P2;
	CNode v31 = P1 - P3;
	if (f8ScalarProduct(v12, v31) > 0) boU1 = true; else boU1 = false; 
	if (f8ScalarProduct(v23, v12) > 0) boU2 = true; else boU2 = false; 
	if (f8ScalarProduct(v31, v23) > 0) boU3 = true; else boU3 = false; 
	// check if the center of the circumscribed sphere is on the oposite side of the triangle
	// centre of circumscribed sphere
	CNode n = CGetCentreSphere(&node);
	// projection of sphere center on triangle
	bool boPoz = boClosestPointCoord(&n, &r, &s);
	CNode P = r*P1 + s*P2 + (1-s-r)*P3;
	CNode PO = n - P;
	CNode AT = node - P1;
	if (f8ScalarProduct(PO, AT) > 0) return false; // center of sphere does not penetrate the triangle plane
	// check if inside the triangle
	if (boPoz) return true;
	// check if on the right side of triangle
	if ((boU1 && (r>=0)) ||
		(boU2 && (s>=0)) ||
		(boU3 && (1-r-s>=0))) return true;
	// check if the projection of the influence zone is outside the triangle
	if (!boU1 && !boU2 && !boU3) return true;
	// the node
	boPoz = boClosestPointCoord(&node, &r, &s);
	// check if inside the triangle
	if (boPoz) return true;
	// check if on the right side of triangle
	if ((boU1 && (r>=0)) ||
		(boU2 && (s>=0)) ||
		(boU3 && (1-r-s>=0))) return true;
	// middle of segment
	if (boU1) n = 0.5*(node+P1);
	else if(boU2) n = 0.5*(node+P2);
	else if(boU3) n = 0.5*(node+P3);
	boPoz = boClosestPointCoord(&n, &r, &s);
	// check if inside the triangle
	if (boPoz) return true;
	// check if on the right side of triangle
	if ((boU1 && (r>=0)) ||
		(boU2 && (s>=0)) ||
		(boU3 && (1-r-s>=0))) return true;
	// centres of adiacent triangles
	CTriangle t1(au32NodeIndex[0], au32NodeIndex[1], u32NodeIdx, pNodeArray);
	boPoz = boClosestPointCoord(&t1.CircCircleCenter, &r, &s);
	// check if inside the triangle
	if (boPoz) return true;
	// check if on the right side of triangle
	if ((boU1 && (r>=0)) ||
		(boU2 && (s>=0)) ||
		(boU3 && (1-r-s>=0))) return true;
	CTriangle t2(au32NodeIndex[0], u32NodeIdx, au32NodeIndex[2], pNodeArray);
	boPoz = boClosestPointCoord(&t2.CircCircleCenter, &r, &s);
	// check if inside the triangle
	if (boPoz) return true;
	// check if on the right side of triangle
	if ((boU1 && (r>=0)) ||
		(boU2 && (s>=0)) ||
		(boU3 && (1-r-s>=0))) return true;
	// no influence detected
	return false;
#endif
}

CBucket::CBucket(void)
{
	u32NumPoints = 0;
	u32CurrentIdx = 0;
	pu32PointIndexes = NULL;
}

void CBucket::vSetSize(uint32 u32Size)
{
	u32NumPoints = u32Size;
	u32CurrentIdx = 0;
	if (u32NumPoints > 0) pu32PointIndexes = new uint32[u32NumPoints];
}

CBucket::~CBucket(void)
{
	if (u32NumPoints > 0) delete[] pu32PointIndexes;
}

void CBucket::vAddPointIndex(uint32 u32Idx)
{
	if (u32CurrentIdx < u32NumPoints)
	{
		pu32PointIndexes[u32CurrentIdx] = u32Idx;
		u32CurrentIdx++;
	}
#if (DEBUG_DEBUG == 1)
	else
	{
		DEBUG_vError();
	}
#endif
}

CBuckets::CBuckets(uint32 u32NumNodes, float8 *pf8NodesCoord, float8 f8dx, float8 f8dy, float8 f8dz)
{
	uint32 i, j, k, p;
	float8 x, y, z;
	u32NumNodes = u32NumNodes;
	pf8NodesCoordinates = new float8[u32NumNodes*3];
	VECT_vCopy(pf8NodesCoord, pf8NodesCoordinates, u32NumNodes*3);
	// find the number of buckets
	f8MinX = pf8NodesCoordinates[0];
    f8MinY = pf8NodesCoordinates[1];
	f8MinZ = pf8NodesCoordinates[2];
	f8MaxX = f8MinX;
	f8MaxY = f8MinY;
	f8MaxZ = f8MinZ;
	for (i = 1; i < u32NumNodes; i++)
	{
		x = pf8NodesCoordinates[3*i];
		y = pf8NodesCoordinates[3*i+1];
		z = pf8NodesCoordinates[3*i+2];
		if (x < f8MinX) f8MinX = x;
		else if (x > f8MaxX) f8MaxX = x;
		if (y < f8MinY) f8MinY = y;
		else if (y > f8MaxY) f8MaxY = y;
		if (z < f8MinZ) f8MinZ = z;
		else if (z > f8MaxZ) f8MaxZ = z;
	}
	f8MaxX += f8dx/2;
	f8MaxY += f8dy/2;
	f8MaxZ += f8dz/2;
	f8MinX -= f8dx/2;
	f8MinY -= f8dy/2;
	f8MinZ -= f8dz/2;

	u32NumX = (uint32)floor((f8MaxX - f8MinX)/f8dx);
	u32NumY = (uint32)floor((f8MaxY - f8MinY)/f8dy);
	u32NumZ = (uint32)floor((f8MaxZ - f8MinZ)/f8dz);
	f8dX = (f8MaxX - f8MinX)/u32NumX;
	f8dY = (f8MaxY - f8MinY)/u32NumY;
	f8dZ = (f8MaxZ - f8MinZ)/u32NumZ;
	f8BucketDiagonal2 = f8dX*f8dX + f8dY*f8dY + f8dZ*f8dZ;
	u32NumBuckets = u32NumX*u32NumY*u32NumZ;
	pBuckets = new CBucket[u32NumBuckets];
	// distribute the nodes into buckets
	uint32 *pu32NodesPerBucket = new uint32[u32NumBuckets];
	for (i = 0; i < u32NumBuckets; i++) pu32NodesPerBucket[i] = 0;
	for (p = 0; p < u32NumNodes; p++)
	{
		x = pf8NodesCoordinates[3*p];
		y = pf8NodesCoordinates[3*p+1];
		z = pf8NodesCoordinates[3*p+2];
		// identify the bucket
		i = u32GetBucketIndexOnX(x);
		j = u32GetBucketIndexOnY(y);
		k = u32GetBucketIndexOnZ(z);
		pu32NodesPerBucket[i*u32NumY*u32NumZ + j*u32NumZ + k]++;
	}
	// set the bucket size
	for (i = 0; i < u32NumBuckets; i++)
	{
		pBuckets[i].vSetSize(pu32NodesPerBucket[i]);
	}
	// add indexes of nodes to buckets
	for (p = 0; p < u32NumNodes; p++)
	{
		x = pf8NodesCoordinates[3*p];
		y = pf8NodesCoordinates[3*p+1];
		z = pf8NodesCoordinates[3*p+2];
		// identify the bucket
		i = u32GetBucketIndexOnX(x);
		j = u32GetBucketIndexOnY(y);
		k = u32GetBucketIndexOnZ(z);
		pBuckets[i*u32NumY*u32NumZ + j*u32NumZ + k].vAddPointIndex(p);
	}
	delete[] pu32NodesPerBucket;
}

CBuckets::~CBuckets(void)
{
	delete[] pf8NodesCoordinates;
	delete[] pBuckets;
}

uint32 CBuckets::u32GetBucketIndexOnX(float8 f8x)
{
	int32 i = (uint32)floor((f8x - f8MinX)/f8dX);
	if (i < 0) i = 0;
	else if ((uint32)i >= u32NumX) i = u32NumX-1;
	return (uint32)i;
}

uint32 CBuckets::u32GetBucketIndexOnY(float8 f8y)
{
	int32 i = (uint32)floor((f8y - f8MinY)/f8dY);
	if (i < 0) i = 0;
	else if ((uint32)i >= u32NumY) i = u32NumY-1;
	return (uint32)i;
}

uint32 CBuckets::u32GetBucketIndexOnZ(float8 f8z)
{
	int32 i = (uint32)floor((f8z - f8MinZ)/f8dZ);
	if (i < 0) i = 0;
	else if ((uint32)i >= u32NumZ) i = u32NumZ-1;
	return (uint32)i;
}

int32 CBuckets::i32GetBucketIndexOnX(float8 f8x)
{
	int32 i = (uint32)floor((f8x - f8MinX)/f8dX);
	if ((i < 0) || ((uint32)i >= u32NumX)) return -1;
	else return i;
}

int32 CBuckets::i32GetBucketIndexOnY(float8 f8y)
{
	int32 i = (uint32)floor((f8y - f8MinY)/f8dY);
	if ((i < 0) || ((uint32)i >= u32NumY)) return -1;
	else return i;
}

int32 CBuckets::i32GetBucketIndexOnZ(float8 f8z)
{
	int32 i = (uint32)floor((f8z - f8MinZ)/f8dZ);
	if ((i < 0) || ((uint32)i >= u32NumZ)) return -1;
	else return i;
}

uint32 *CBuckets::pu32GetNodesIndexesInBucket(uint32 i, uint32 j, uint32 k, uint32 *pu32NumNodes)
{
	uint32 u32Bucket = i*u32NumY*u32NumZ + j*u32NumZ + k;
	*pu32NumNodes = pBuckets[u32Bucket].u32NumPoints;
	return pBuckets[u32Bucket].pu32PointIndexes;
}

CTriangularSurface::CTriangularSurface(uint32 u32NumTria, uint32 *pu32NodesIdx, float8 *pf8NodeCoord, float8 f8MaxPossiblePenetrat)
{
	uint32 i;
    uint32 u32NumTriangles = u32NumTria;
	pu32NodeIndexes = pu32NodesIdx;
	pf8NodeCoordinates = pf8NodeCoord;
	f8MaxPossiblePenetration = f8MaxPossiblePenetrat;
	pBuckets = NULL;
	// find maximum node index
	uint32 u32MaxNodeIndex = 0;
	for (i = 0; i < 3*u32NumTriangles; i++)
	{ 
		if (pu32NodeIndexes[i] > u32MaxNodeIndex) u32MaxNodeIndex = pu32NodeIndexes[i];
	};
	// find number of master nodes and create a reindexing array and the node array
	int32 *pi32ReindexArray = new int32[u32MaxNodeIndex+1];
	for (i = 0; i <= u32MaxNodeIndex; i++) pi32ReindexArray[i] = -1;
	uint32 u32NumNodes = 0;
	for (i = 0; i < 3*u32NumTriangles; i++)
	{
		if (pi32ReindexArray[pu32NodeIndexes[i]] < 0)
		{
			pi32ReindexArray[pu32NodeIndexes[i]] = u32NumNodes;
			CMyNode *pCNode = new CMyNode(pf8NodeCoordinates + 3*pu32NodeIndexes[i]);
			aNodes.vAdd(pCNode);
			u32NumNodes++;
		}
	}
	// create the triangle array
	for (i = 0; i < u32NumTriangles; i++)
	{
		CTriangle *pCTria = new CTriangle(pi32ReindexArray[pu32NodeIndexes[i*3]],
			pi32ReindexArray[pu32NodeIndexes[i*3+1]], pi32ReindexArray[pu32NodeIndexes[i*3+2]],
			&aNodes);
		aTriangles.vAdd(pCTria);
	}


	// create an array with all the segments from all the triangles
	uint32 *pu32AllSegments = new uint32[u32NumTriangles*3*2];
	for (i = 0; i < u32NumTriangles; i++)
	{
		pu32AllSegments[i*6] = pi32ReindexArray[pu32NodeIndexes[i*3]];
		pu32AllSegments[i*6+1] = pi32ReindexArray[pu32NodeIndexes[i*3+1]];
		pu32AllSegments[i*6+2] = pi32ReindexArray[pu32NodeIndexes[i*3+1]];
		pu32AllSegments[i*6+3] = pi32ReindexArray[pu32NodeIndexes[i*3+2]];
		pu32AllSegments[i*6+4] = pi32ReindexArray[pu32NodeIndexes[i*3+2]];
		pu32AllSegments[i*6+5] = pi32ReindexArray[pu32NodeIndexes[i*3]];
	}
	// find the number of segments and reindex them
	uint32 u32NumMasterSegments = 0;
	int32 *pi32SegmentIndexes = new int32[u32NumTriangles*3];
	uint8 *pu8SegmentOrientation = new uint8[u32NumTriangles*3];
	for (i = 0; i < u32NumTriangles*3; i++) pi32SegmentIndexes[i] = -1;
	for (i = 0; i < u32NumTriangles*3; i++)
	{
		uint32 p1, p2;
		p1 = pu32AllSegments[i*2];
		p2 = pu32AllSegments[i*2+1];
		bool found = false;
		// search this segment in the previous segments
		for (uint32 j = 0; j < i; j++)
		{
			if ((p1 == pu32AllSegments[j*2]) && (p2 == pu32AllSegments[j*2+1]))
			{
				found = true;
				pi32SegmentIndexes[i] = pi32SegmentIndexes[j];
				pu8SegmentOrientation[i] = 0;
				break;
			} else if ((p2 == pu32AllSegments[j*2]) && (p1 == pu32AllSegments[j*2+1]))
			{
				found = true;
				pi32SegmentIndexes[i] = pi32SegmentIndexes[j];
				pu8SegmentOrientation[i] = 1;
				break;
			}
		}
		if (!found)
		{
			pi32SegmentIndexes[i] = u32NumMasterSegments;
			pu8SegmentOrientation[i] = 0;
			u32NumMasterSegments++;
		}
	}
	// add segment information to the triangles
	for (i = 0; i < u32NumTriangles; i++)
	{
		CTriangle *pCT = aTriangles.ElementAt(i);
		// segment indexes and orientation
		pCT->au32SegmentIdx[0] = pi32SegmentIndexes[i*3];
		pCT->au32SegmentIdx[1] = pi32SegmentIndexes[i*3+1];
		pCT->au32SegmentIdx[2] = pi32SegmentIndexes[i*3+2];
		pCT->au8SegmentOrientation[0] = pu8SegmentOrientation[i*3];
		pCT->au8SegmentOrientation[1] = pu8SegmentOrientation[i*3+1];
		pCT->au8SegmentOrientation[2] = pu8SegmentOrientation[i*3+2];
	}
	// segments info structure
	for (i = 0; i < u32NumMasterSegments; i++)
	{
		CSegment *pCS = new CSegment();
		// search the segment index in the list of segments
		bool foundFirst = false;
		for (uint32 j = 0; j < u32NumTriangles*3; j++)
		{
			if (pi32SegmentIndexes[j] == i) 
			{
				if (foundFirst == false)
				{
					// take the node indexes and adiacent triangles
					uint32 u32TriangleIdx = j/3;
					uint32 u32SegmentIdx = j%3;
					pCS->au32NodeIdxs[0] = pu32AllSegments[u32TriangleIdx*6+u32SegmentIdx*2];
					pCS->au32NodeIdxs[1] = pu32AllSegments[u32TriangleIdx*6+u32SegmentIdx*2+1];
					pCS->ai32AdiacentTrianglesIdx[0] = u32TriangleIdx;
					foundFirst = true;
				}
				else
				{
					// second triangle found
					pCS->ai32AdiacentTrianglesIdx[1] = j/3;
					pCS->boOnSurfaceEdge = false;
				}
			}
		}
		pCS->vComputeVectorComponents(&aNodes);
		aSegments.vAdd(pCS);
	}
	CMyNode *pstN;
	uint32 u32OriginalIndex;
	for (i = 0; i < u32NumNodes; i++)
	{
		pstN = aNodes.ElementAt(i);
		// Original node index
		for (uint32 j = 0; j <= u32MaxNodeIndex+1; j++)
		{
			if (pi32ReindexArray[j] == i)
			{
				u32OriginalIndex = j;
				break;
			}
		}
		// Indexes of adiacent triangles and position inside them
		for (uint32 j = 0; j < u32NumTriangles; j++)
		{
			if (pu32NodeIndexes[j*3] == u32OriginalIndex)
			{
				pstN->au32AdiacentTrianglesIndexes.push_back(j);
				pstN->au8PozitionInAdiacentTriangles.push_back(0);
			}
			else if (pu32NodeIndexes[j*3+1] == u32OriginalIndex)
			{
				pstN->au32AdiacentTrianglesIndexes.push_back(j);
				pstN->au8PozitionInAdiacentTriangles.push_back(1);
			}
			else if (pu32NodeIndexes[j*3+2] == u32OriginalIndex)
			{
				pstN->au32AdiacentTrianglesIndexes.push_back(j);
				pstN->au8PozitionInAdiacentTriangles.push_back(2);
			}
		}
	}

	// compute normals for triangles
	CTriangle *pstT;
	for (i = 0; i < u32NumTriangles; i++)
	{
		pstT = aTriangles.ElementAt(i);
		pstT->vComputeNormal(&aSegments);
	}
	// compute normals for segments
	CSegment *pstS;
	for (i = 0; i < u32NumMasterSegments; i++)
	{
		pstS = aSegments.ElementAt(i);
		pstS->vComputeNormal(&aTriangles);
	}
	// compute normals for nodes, adiacent segments
	for (i = 0; i < u32NumNodes; i++)
	{
		pstN = aNodes.ElementAt(i);
		pstN->vComputeNormal(&aTriangles);
		pstN->vGetAdiacentSegments(&aSegments, i);
	}

	delete[] pu8SegmentOrientation;
	delete[] pi32SegmentIndexes;
	delete[] pu32AllSegments;
	delete[] pi32ReindexArray;
	vStudySurface();
	//vPrintInfo("surf.txt");
}

CTriangularSurface::~CTriangularSurface(void)
{
	aNodes.vRemoveAll();
	aTriangles.vRemoveAll();
	aSegments.vRemoveAll();
	if (pBuckets != NULL) delete pBuckets;
}

void CTriangularSurface::vCreateTriangleBuckets(void)
{
	uint32 i;
	float8 f8MaxR;
	if (pBuckets != NULL) delete pBuckets;
	uint32 u32NumNodes = aNodes.u32GetCount();
	float8 *pf8NodesCoord = new float8[u32NumNodes*3];
	for (i = 0; i < u32NumNodes; i++)
	{
		pf8NodesCoord[i*3] = aNodes.ElementAt(i)->af8Coordinates[0];
		pf8NodesCoord[i*3+1] = aNodes.ElementAt(i)->af8Coordinates[1];
		pf8NodesCoord[i*3+2] = aNodes.ElementAt(i)->af8Coordinates[2];
	}
	f8MaxR = 0;
	uint32 u32NumTriangles = aTriangles.u32GetCount();
	for (i = 0; i < u32NumTriangles; i++)
	{
		float8 x = aTriangles.ElementAt(i)->f8CircCircleRadius;
		if (x > f8MaxR) f8MaxR = x;
	}
	pBuckets = new CBuckets(u32NumNodes, pf8NodesCoord, f8MaxR, f8MaxR, f8MaxR);
	delete[] pf8NodesCoord;
}

void CTriangularSurface::vCreateSegmentBuckets(void)
{
	uint32 i;
	float8 f8MaxR;
	if (pBuckets != NULL) delete pBuckets;
	uint32 u32NumNodes = aNodes.u32GetCount();
	float8 *pf8NodesCoord = new float8[u32NumNodes*3];
	for (i = 0; i < u32NumNodes; i++)
	{
		pf8NodesCoord[i*3] = aNodes.ElementAt(i)->af8Coordinates[0];
		pf8NodesCoord[i*3+1] = aNodes.ElementAt(i)->af8Coordinates[1];
		pf8NodesCoord[i*3+2] = aNodes.ElementAt(i)->af8Coordinates[2];
	}
	f8MaxR = 0;
	for (i = 0; i < aSegments.u32GetCount(); i++)
	{
		float8 x = aSegments.ElementAt(i)->f8Length;
		if (x > f8MaxR) f8MaxR = x;
	}
	f8MaxR = f8MaxR/2;
	pBuckets = new CBuckets(u32NumNodes, pf8NodesCoord, f8MaxR, f8MaxR, f8MaxR);
	delete[] pf8NodesCoord;
}

void CTriangularSurface::vStudySurface(void)
{
	uint32 i, j, k, t, n, i1, i2, j1, j2, k1, k2, u32NumNodesInBucket, u32NodeIdx;
	uint32 *pu32NodeIdxs;
	vCreateTriangleBuckets();
	uint32 u32NumTriangles = aTriangles.u32GetCount();
	for (t = 0; t < u32NumTriangles; t++)
	{
		// identify the bucket containing the center of circumscribed circle O
		CTriangle *pTriangle = aTriangles.ElementAt(t);
		CNode &rO = pTriangle->CircCircleCenter;
		i = pBuckets->u32GetBucketIndexOnX(rO.af8Coordinates[0]);
		j = pBuckets->u32GetBucketIndexOnY(rO.af8Coordinates[1]);
		k = pBuckets->u32GetBucketIndexOnZ(rO.af8Coordinates[2]);
		// identify the indexes of surrounding buckets
		if (i > 0) i1 = i-1; else i1 = 0;
		if (j > 0) j1 = j-1; else j1 = 0;
		if (k > 0) k1 = k-1; else k1 = 0;
		if (i < pBuckets->u32NumX - 1) i2 = i + 1; else i2 = pBuckets->u32NumX - 1;
		if (j < pBuckets->u32NumY - 1) j2 = j + 1; else j2 = pBuckets->u32NumY - 1;
		if (k < pBuckets->u32NumZ - 1) k2 = k + 1; else k2 = pBuckets->u32NumZ - 1;
		// Search in the bucket containing O and in the surrounding buckets for master nodes T
		for (i = i1; i <= i2; i++)
		{
			for (j = j1; j <= j2; j++)
			{
				for (k = k1; k <= k2; k++)
				{
					pu32NodeIdxs = pBuckets->pu32GetNodesIndexesInBucket(i, j, k, &u32NumNodesInBucket);
					for (n = 0; n < u32NumNodesInBucket; n++)
					{
						u32NodeIdx = pu32NodeIdxs[n];
						vStudyTriangleNodeRelation(t, u32NodeIdx); 
					}
				}
			}
		}
	}
	vCreateSegmentBuckets();
	for (t = 0; t < aSegments.u32GetCount(); t++)
	{
		// identify the bucket containing the center of circumscribed circle O
		CSegment *pSegment = aSegments.ElementAt(t);
		CNode rO = 0.5*(*aNodes.ElementAt(pSegment->au32NodeIdxs[0])+ *aNodes.ElementAt(pSegment->au32NodeIdxs[1]));
		i = pBuckets->u32GetBucketIndexOnX(rO.af8Coordinates[0]);
		j = pBuckets->u32GetBucketIndexOnY(rO.af8Coordinates[1]);
		k = pBuckets->u32GetBucketIndexOnZ(rO.af8Coordinates[2]);
		// identify the indexes of surrounding buckets
		if (i > 0) i1 = i-1; else i1 = 0;
		if (j > 0) j1 = j-1; else j1 = 0;
		if (k > 0) k1 = k-1; else k1 = 0;
		if (i < pBuckets->u32NumX - 1) i2 = i + 1; else i2 = pBuckets->u32NumX - 1;
		if (j < pBuckets->u32NumY - 1) j2 = j + 1; else j2 = pBuckets->u32NumY - 1;
		if (k < pBuckets->u32NumZ - 1) k2 = k + 1; else k2 = pBuckets->u32NumZ - 1;
		// Search in the bucket containing O and in the surrounding buckets for master nodes T
		for (i = i1; i <= i2; i++)
		{
			for (j = j1; j <= j2; j++)
			{
				for (k = k1; k <= k2; k++)
				{
					pu32NodeIdxs = pBuckets->pu32GetNodesIndexesInBucket(i, j, k, &u32NumNodesInBucket);
					for (n = 0; n < u32NumNodesInBucket; n++)
					{
						u32NodeIdx = pu32NodeIdxs[n];
						vStudySegmentNodeRelation(t, u32NodeIdx); 
					}
				}
			}
		}
	}
	// clean up lists
	for (t = 0; t < aNodes.u32GetCount(); t++)
	{
		CMyNode *pNode = aNodes.ElementAt(t);
		//pNode->vCleanUpAditionalCheckLists(&aSegments);
	}
}

#define EPS_ 1E-10

void CTriangularSurface::vStudyTriangleNodeRelation(uint32 u32Tidx, uint32 u32NodeIdx)
{
	//check that T does not belong to T1T2T3 and OT < R
	CTriangle *pTriangle = aTriangles.ElementAt(u32Tidx);
	float8 f8R2 = pTriangle->f8CircCircleRadius * pTriangle->f8CircCircleRadius;
	CNode &rO = pTriangle->CircCircleCenter;
	// check if node belongs to current triangle
	if (pTriangle->i8ContainsNode(u32NodeIdx) >= 0) return;
	// compute the distance to O
	CMyNode &n = *aNodes.ElementAt(u32NodeIdx);
	CNode dOT = n - rO;
	float8 f8dOT = f8ScalarProduct(dOT, dOT);
	// check OT < 2*R)
	if (f8dOT > 4*f8R2) return;
	//check the relative position using the normals
	if (VECT_f8ScalarProduct(n.af8NormalComponents, pTriangle->af8NormalComponents, 3) > 0) return;
	// check region of influence
	if (pTriangle->boPossibleInfluenceZone(u32NodeIdx, f8MaxPossiblePenetration))
	{
		// add triangle to the aditional triangles to check list for the node
		n.au32AditionalTrianglesToCheck.push_back(u32Tidx);
	}
}

void CTriangularSurface::vStudySegmentNodeRelation(uint32 u32Sidx, uint32 u32NodeIdx)
{
	CSegment *pSegment = aSegments.ElementAt(u32Sidx);
	if (pSegment->boOnSurfaceEdge) return;
	// check if the segment contains the node
	if (pSegment->au32NodeIdxs[0] == u32NodeIdx) return;
	if (pSegment->au32NodeIdxs[1] == u32NodeIdx) return;
	// check if the segment is collinear with the node
	CTriangle t(pSegment->au32NodeIdxs[0], pSegment->au32NodeIdxs[1], u32NodeIdx, &aNodes);
	if (t.f8Area() < EPS_) return;
	// check if the node and segment are on the same triangle
	int32 i32Tidx = pSegment->ai32AdiacentTrianglesIdx[0];
	if (aTriangles.ElementAt(i32Tidx)->i8ContainsNode(u32NodeIdx) >= 0) return;
	i32Tidx = pSegment->ai32AdiacentTrianglesIdx[1];
	if (i32Tidx >= 0)
	{
		if (aTriangles.ElementAt(i32Tidx)->i8ContainsNode(u32NodeIdx) >= 0) return;
	}
	// check distance node-segment
	float8 f8R2 = pSegment->f8Length/2;
	f8R2 *= f8R2;
	CNode rO = 0.5*(*aNodes.ElementAt(pSegment->au32NodeIdxs[0])+ *aNodes.ElementAt(pSegment->au32NodeIdxs[1]));
	CMyNode &n = *aNodes.ElementAt(u32NodeIdx);
	CNode dOT = n - rO;
	float8 f8dOT = f8ScalarProduct(dOT, dOT);
	CNode dOM = t.CircCircleCenter - rO;
	float8 f8dOM = f8ScalarProduct(dOM, dOM);
	// check influence zone
	if ((f8dOT > f8R2) && (f8dOM > f8MaxPossiblePenetration*f8MaxPossiblePenetration)) return;
	
	// check the relative position using the normals
	if (VECT_f8ScalarProduct(n.af8NormalComponents, pSegment->af8NormalComponents, 3) > 0) return;
	// add segment to the aditional segments to check list for the node
	n.au32AditionalSegmentsToCheck.push_back(u32Sidx);
}

