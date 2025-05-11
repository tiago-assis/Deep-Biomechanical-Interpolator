#pragma once
#include "cdef.h"
#include "utils.h"
#include "TriangularSurface.h"

typedef struct {
	float8 f8InfluenceRadius2;
	uint8 u8NumAdiacentTriangles;
	uint8 *pu8PozitionInAdiacentTriangles;
	uint32 *pu32AdiacentTrianglesIndexes;
	uint32 *pu32NeighbouringSegmentsIndexes;
	float8 *pf8AdiacentTrianglesBisectors;
	uint8 *pu8SegmentsInfluencedByTriangle;
	uint8 u8NumAdiacentSegments;
	uint32 *pu32AdiacentSegmentsIndexes;
	int8 *pi8AdiacentSegmentsEnabled;
	float8 af8NormalComponents[3];
	uint8 u8OnSurfaceEdge;
	uint8 u8NumAditionalSegmentsToCheck;
	uint32 *pu32AditionalSegmentsToCheckIndexes;
	uint8 u8NumAditionalTrianglesToCheck;
	uint32 *pu32AditionalTrianglesToCheckIndexes;
} tstMasterNodeInfo;

typedef struct {
	float8 af8Node1Coordinates[3];
	float8 af8NormalComponents[3];
	float8 af8VectorComponents[3];
	int32 ai32AdiacentTriangles[2];
	uint8 u8OnSurfaceEdge;
	float8 f8Length2;
} tstMasterSegmentInfo;

typedef struct {
	float8 af8Node1Coordinates[3];
	float8 af8V12Components[3];
	float8 af8V13Components[3];
	float8 af8NormalComponents[3];
	float8 a11, a12, a22, det;
} tstMasterTriangleInfo;

class CContactPair
{
public:
	CContactPair(uint32 u32NumTriangles, uint32 *pu32NodeIndexes, float8 *pf8NodeCoordinates, uint32 u32NumContactNodes, uint32 *pu32ContactNodesIdx, float8 f8MaxPossiblePenetration);
	~CContactPair(void);
	int32 i32GetClosestNodeIndex(float8 *pf8Node, float8 *pf8MinDist2);
	uint8 u8ApplyContactToNode(float8 *pf8Node);
	void vApplyContacts(float8 *pf8CurrentNodePositions, uint8 *pu8NodeInContact);
	uint8 u8PerformAditionalChecks(float8 *pf8Node, uint32 u32ClosestNodeIdx, float8 *pf8MinDist2);
	bool boCheckTriangle(float8 *pf8Node, uint32 u32TriangleIdx, uint8 u8PozitionOfMasterNode, 
		uint32 u32NeighbouringSegment, float8 *pf8MinDist2);
	bool boCheckNeighbouringTriangle(float8 *pf8Node, uint32 u32TriangleIdx, float8 *pf8MinDist2);
	void vCheckSegment(float8 *pf8Node, uint32 u32SegIdx, float8 *pf8MinDist2);
	// Initialization function
	void vInitMasterSurface(CTriangularSurface *pCSurface);
	void vSetMaxPenetration(float8 f8MaxDisp) {f8MaxPenetration = 1.7321*f8MaxDisp;};
	void vUnMarkContactDOFs(uint8 *pu8ContactDOFs);
private:
	// index of node set and surface that define the pair
	float8 f8ActivationDistance;
	// derived variables
	uint32 u32NumContactNodes;
	uint32 *pu32ContactNodesIdx;
	uint32 u32NumTriangles;
	uint32 *pu32NodeIndexes; // array of point indexes that define the triangles, size = 3*u32NumTriangles
	float8 *pf8NodeCoordinates; // coordinates of the Points defining the nodes
	// other helpful variable
	uint32 u32NumMasterNodes;
	uint32 u32NumMasterSegments;
	tstMasterNodeInfo *pstMasterNodeInfo;
	tstMasterTriangleInfo *pstMasterTriangleInfo;
	tstMasterSegmentInfo *pstMasterSegmentInfo;
	// buckets for fast search
	CBuckets *pBuckets;
	// variables used for search
	uint32 u32ClosestTriangle, u32ClosestSegment;
	bool boClosestToTriangle, boClosestToSegment;
	float8 f8ProjVectX, f8ProjVectY, f8ProjVectZ;
	float8 f8MaxPenetration;
};

typedef CMyArray<CContactPair> CContactPairArray;