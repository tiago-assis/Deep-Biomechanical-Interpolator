#pragma once
#include <vector>
#include "cdef.h"
#include "utils.h"

class CTriangle;
class CSegment;

typedef CMyArray<CSegment> CSegmentArray;
typedef CMyArray<CTriangle> CTriangleArray;

class CNode
{
public:
	CNode();
	CNode(float8 *pf8Coord);
	~CNode(void) {};
	CNode operator+(CNode node);
	CNode operator-(CNode node);
	CNode operator-();
	friend  CNode operator*(CNode node, float8 f8x);
	friend  CNode operator*(float8 f8x, CNode node);
	friend float8 f8ScalarProduct(CNode node1, CNode node2);
	void vSaveBinary(FILE *pf);
	float8 af8Coordinates[3];
};

class CMyNode:public CNode
{
public:
	CMyNode();
	CMyNode(float8 *pf8Coord);
	~CMyNode(void);
	void vComputeNormal(CTriangleArray *pTriangles);
	void vGetAdiacentSegments(CSegmentArray *pSegments, uint32 u32NodeIdx);
	void vCleanUpAditionalCheckLists(CSegmentArray *pSegments);
	void vRemoveTriangleToCheck(uint32 u32T);
	float8 af8NormalComponents[3];
	// aditional information
	std::vector<uint32> au32AdiacentTrianglesIndexes;
	std::vector<uint8> au8PozitionInAdiacentTriangles;
	std::vector<uint32> au32AdiacentSegmentsIndexes;
	std::vector<uint8> au8PozitionInAdiacentSegments;
	std::vector<uint32> au32AditionalSegmentsToCheck;
	std::vector<uint32> au32AditionalTrianglesToCheck;
	bool boOnSurfaceEdge;
};

typedef CMyArray<CMyNode> CNodeArray;

class CSegment
{
public:
	CSegment();
	~CSegment(void) {};
	void vSetNodeIndexes(uint32 u32N1, uint32 u32N2);
	void vComputeVectorComponents(CNodeArray *pNodes);
	void vComputeNormal(CTriangleArray *pTriangles);
	bool boClosestPointCoord(CNodeArray *pNodes, CNode *pNode, float8 *r);
	int8 i8ContainsNode(uint32 u32NodeIdx);
	uint32 au32NodeIdxs[2];
	int32 ai32AdiacentTrianglesIdx[2];
	float8 af8NormalComponents[3];
	float8 af8VectorComponents[3];
	float8 f8Length;
	bool boOnSurfaceEdge;
};

class CTriangle
{
public:
	CTriangle(uint32 u32N1, uint32 u32N2, uint32 u32N3, CNodeArray *pNodeArr);
	~CTriangle(void);
	void vSaveBinary(FILE *pf);
	int8 i8ContainsNode(uint32 u32NodeIdx);
	bool boIsCoplanar(CMyNode *pNode);
	float8 f8Distance(CMyNode *pNode);
	float8 f8Area(void);
	bool boClosestPointCoord(CNode *pNode, float8 *r, float8 *s);
	float8 f8SegmentIntersection(CNode *pN1, CNode *pN2); // returns pozition on the segment, -1 if not intersecting
	bool boPossibleInfluenceZone(uint32 u32NodeIdx, float8 f8MaxPossiblePenetration);
	void vComputeCircumscribedCircle(void);
	void vSetSegmentIndexes(uint32 u32S1, uint32 u32S2, uint32 u32S3);
	void vComputeNormal(CSegmentArray *pSegments);
	void vComputeBisector(uint8 u8NodePoz, float8 *pf8Bisector);
	bool boIsOptuse(uint8 u8NodePoz);
	void vGetMaximumSegmentProjection(float8 *pf8ProjOnX, float8 *pf8ProjOnY, float8 *pf8ProjOnZ);
	void vGetBoundingBox(float8 &f8MinX, float8 &f8MinY, float8 &f8MinZ, float8 &f8MaxX, float8 &f8MaxY, float8 &f8MaxZ);
	CNode CGetCentreSphere(CNode *pNode);
	CNodeArray *pNodeArray;
	uint32 au32NodeIndex[3];
	uint32 au32SegmentIdx[3];
	uint8 au8SegmentOrientation[3];
	float8 f8CircCircleRadius;
	CNode CircCircleCenter;
	float8 af8NormalComponents[3];
};

class CBucket
{
public:
	CBucket(void);
	~CBucket(void);
	void vSetSize(uint32 u32Size);
	void vAddPointIndex(uint32 u32Idx);
	uint32 u32NumPoints;
	uint32 u32CurrentIdx;
	uint32 *pu32PointIndexes;
};

class CBuckets
{
public:
	CBuckets(uint32 u32NumNodes, float8 *pf8NodesCoord, float8 f8dx, float8 f8dy, float8 f8dz);
	~CBuckets(void);
	uint32 u32GetBucketIndexOnX(float8 f8x);
	uint32 u32GetBucketIndexOnY(float8 f8y);
	uint32 u32GetBucketIndexOnZ(float8 f8z);
	int32 i32GetBucketIndexOnX(float8 f8x);
	int32 i32GetBucketIndexOnY(float8 f8y);
	int32 i32GetBucketIndexOnZ(float8 f8z);
	uint32 *pu32GetNodesIndexesInBucket(uint32 i, uint32 j, uint32 k, uint32 *pu32NumNodes); 
	uint32 u32NumNodes;
	float8 *pf8NodesCoordinates; 
	// size of the box
	float8 f8MinX, f8MinY, f8MinZ, f8MaxX, f8MaxY, f8MaxZ; 
	// dimension of buckets
	float8 f8dX, f8dY, f8dZ;
	// number of buckets
	uint32 u32NumX, u32NumY, u32NumZ, u32NumBuckets;
	// size of bucket diagonal squared
	float8 f8BucketDiagonal2;
	CBucket *pBuckets;
};

class CTriangularSurface
{
public:
	CTriangularSurface(uint32 u32NumTria, uint32 *pu32NodesIdx, float8 *pf8NodeCoord, float8 f8MaxPossiblePenetration);
	~CTriangularSurface(void);
	void vCreateTriangleBuckets(void);
	void vCreateSegmentBuckets(void);
	void vStudySurface(void);
	void vStudyTriangleNodeRelation(uint32 u32Tidx, uint32 u32Nidx);
	void vStudySegmentNodeRelation(uint32 u32Sidx, uint32 u32Nidx);
	uint32 *pu32NodeIndexes; // array of point indexes that define the triangles, size = 3*u32NumTriangles
	float8 *pf8NodeCoordinates; // coordinates of the Points defining the nodes
	CNodeArray aNodes;
	CSegmentArray aSegments;
	CTriangleArray aTriangles;
	CBuckets *pBuckets;
	float8 f8MaxPossiblePenetration;
};
