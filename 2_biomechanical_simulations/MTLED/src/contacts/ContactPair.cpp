#include <cstdlib>
#include <math.h>
#include "ContactPair.hpp"

CContactPair::CContactPair(uint32 u32NumTriangles_, uint32 *pu32NodeIndexes_, float8 *pf8NodeCoordinates_, uint32 u32NumContactNodes_, uint32 *pu32ContactNodesIdx_, float8 f8MaxPossiblePenetration_)
{
	u32NumContactNodes = u32NumContactNodes_;
	pu32ContactNodesIdx = pu32ContactNodesIdx_;
	u32NumTriangles = u32NumTriangles_;
	pu32NodeIndexes = pu32NodeIndexes_;
	pf8NodeCoordinates = pf8NodeCoordinates_;
	f8MaxPenetration = f8MaxPossiblePenetration_;
	pstMasterNodeInfo = NULL; 
	pstMasterTriangleInfo = NULL;
	pstMasterSegmentInfo = NULL;
	if (u32NumTriangles > 0) 
	{
		CTriangularSurface *pCSurface = new CTriangularSurface(u32NumTriangles, pu32NodeIndexes, pf8NodeCoordinates, f8MaxPenetration);
		vInitMasterSurface(pCSurface);
		delete pCSurface;
	}
}

CContactPair::~CContactPair(void)
{
	if (pstMasterNodeInfo != NULL) 
	{
		for (uint32 i = 0; i < u32NumMasterNodes; i++)
		{
			delete[] pstMasterNodeInfo[i].pu32AdiacentTrianglesIndexes;
			delete[] pstMasterNodeInfo[i].pu8PozitionInAdiacentTriangles;
			delete[] pstMasterNodeInfo[i].pu32NeighbouringSegmentsIndexes;
			delete[] pstMasterNodeInfo[i].pf8AdiacentTrianglesBisectors;
			delete[] pstMasterNodeInfo[i].pu32AdiacentSegmentsIndexes;
			delete[] pstMasterNodeInfo[i].pi8AdiacentSegmentsEnabled;
			delete[] pstMasterNodeInfo[i].pu8SegmentsInfluencedByTriangle;
			if (pstMasterNodeInfo[i].pu32AditionalSegmentsToCheckIndexes != NULL) delete[] pstMasterNodeInfo[i].pu32AditionalSegmentsToCheckIndexes;
			if (pstMasterNodeInfo[i].pu32AditionalTrianglesToCheckIndexes != NULL) delete[] pstMasterNodeInfo[i].pu32AditionalTrianglesToCheckIndexes;
		}
		delete[] pstMasterNodeInfo;
	}
	if (pstMasterTriangleInfo != NULL) delete[] pstMasterTriangleInfo;
	if (pstMasterSegmentInfo != NULL) delete[] pstMasterSegmentInfo;
	delete[] pf8NodeCoordinates;
	delete pBuckets;
}

void CContactPair::vUnMarkContactDOFs(uint8 *pu8ContactDOFs)
{
	for (uint32 i = 0; i < u32NumContactNodes; i++)
	{
		uint32 n = 3*pu32ContactNodesIdx[i];
		pu8ContactDOFs[n] = 0;
		pu8ContactDOFs[n+1] = 0;
		pu8ContactDOFs[n+2] = 0;
	}
}

void CContactPair::vInitMasterSurface(CTriangularSurface *pCSurface)
{
	uint32 i;
	// initialize master surface information and data structures 
	u32NumTriangles = pCSurface->aTriangles.u32GetCount();
	u32NumMasterNodes = pCSurface->aNodes.u32GetCount();
	u32NumMasterSegments = pCSurface->aSegments.u32GetCount();
	f8ActivationDistance = pCSurface->f8MaxPossiblePenetration;
	pf8NodeCoordinates = new float8[u32NumMasterNodes*3];
	for (i = 0; i < u32NumMasterNodes; i++)
	{
		VECT_vCopy(pCSurface->aNodes.ElementAt(i)->af8Coordinates, pf8NodeCoordinates+3*i, 3);
	}
	pstMasterNodeInfo = new tstMasterNodeInfo[u32NumMasterNodes];
	pstMasterTriangleInfo = new tstMasterTriangleInfo[u32NumTriangles];
	pstMasterSegmentInfo = new tstMasterSegmentInfo[u32NumMasterSegments];
	// set the information about segments
	tstMasterSegmentInfo *pstS;
	for (i = 0; i < u32NumMasterSegments; i++)
	{
		pstS = pstMasterSegmentInfo+i;
		// coordinates of first node
		uint32 u32N1Idx = pCSurface->aSegments.ElementAt(i)->au32NodeIdxs[0];
		VECT_vCopy(pCSurface->aNodes.ElementAt(u32N1Idx)->af8Coordinates, pstS->af8Node1Coordinates, 3);
		// vector components
		VECT_vCopy(pCSurface->aSegments.ElementAt(i)->af8VectorComponents, pstS->af8VectorComponents, 3);
		pstS->f8Length2 = VECT_f8ScalarProduct(pstS->af8VectorComponents, pstS->af8VectorComponents, 3);
		// normal components
		VECT_vCopy(pCSurface->aSegments.ElementAt(i)->af8NormalComponents, pstS->af8NormalComponents, 3);
		// on edge?
		pstS->u8OnSurfaceEdge = (uint8)pCSurface->aSegments.ElementAt(i)->boOnSurfaceEdge;
		// adiacent triangles
		pstS->ai32AdiacentTriangles[0] = pCSurface->aSegments.ElementAt(i)->ai32AdiacentTrianglesIdx[0];
		pstS->ai32AdiacentTriangles[1] = pCSurface->aSegments.ElementAt(i)->ai32AdiacentTrianglesIdx[1];
	}
	// set the information about triangles
	tstMasterTriangleInfo *pstT;
	for (i = 0; i < u32NumTriangles; i++)
	{
		pstT = pstMasterTriangleInfo+i;
		// coordinates of first node
		uint32 u32N1Idx = pCSurface->aTriangles.ElementAt(i)->au32NodeIndex[0];
		VECT_vCopy(pCSurface->aNodes.ElementAt(u32N1Idx)->af8Coordinates, pstT->af8Node1Coordinates, 3);
		// vector components
		uint32 u32N2Idx = pCSurface->aTriangles.ElementAt(i)->au32NodeIndex[1];
		uint32 u32N3Idx = pCSurface->aTriangles.ElementAt(i)->au32NodeIndex[2];
		CNode &n1 = *pCSurface->aNodes.ElementAt(u32N1Idx);
		CNode &n2 = *pCSurface->aNodes.ElementAt(u32N2Idx);
		CNode &n3 = *pCSurface->aNodes.ElementAt(u32N3Idx);
		CNode v = n2-n1;
		VECT_vCopy(v.af8Coordinates, pstT->af8V12Components, 3);
		v = n3-n1;
		VECT_vCopy(v.af8Coordinates, pstT->af8V13Components, 3);
		// normal components
		VECT_vCopy(pCSurface->aTriangles.ElementAt(i)->af8NormalComponents, pstT->af8NormalComponents, 3);
		// other variables
		pstT->a11 = VECT_f8ScalarProduct(pstT->af8V12Components, pstT->af8V12Components, 3);
		pstT->a22 = VECT_f8ScalarProduct(pstT->af8V13Components, pstT->af8V13Components, 3);
		pstT->a12 = VECT_f8ScalarProduct(pstT->af8V12Components, pstT->af8V13Components, 3);
		pstT->det = pstT->a11*pstT->a22 - pstT->a12*pstT->a12;
	}
	// set the information about nodes
	tstMasterNodeInfo *pstN;
	for (i = 0; i < u32NumMasterNodes; i++)
	{
		pstN = pstMasterNodeInfo+i;
		// number of adiacent triangles
		pstN->u8NumAdiacentTriangles = (uint8)pCSurface->aNodes.ElementAt(i)->au32AdiacentTrianglesIndexes.size();
		// on edge?
		pstN->u8OnSurfaceEdge = (uint8)pCSurface->aNodes.ElementAt(i)->boOnSurfaceEdge;
		// allocate memory for info about adiacent triangles and segments
		pstN->pu8PozitionInAdiacentTriangles = new uint8[pstN->u8NumAdiacentTriangles];
		pstN->pu32AdiacentTrianglesIndexes = new uint32[pstN->u8NumAdiacentTriangles];
		pstN->pu32NeighbouringSegmentsIndexes = new uint32[pstN->u8NumAdiacentTriangles];
		pstN->pf8AdiacentTrianglesBisectors = new float8[3*pstN->u8NumAdiacentTriangles];
		pstN->pu8SegmentsInfluencedByTriangle = new uint8[2*pstN->u8NumAdiacentTriangles];
		pstN->u8NumAdiacentSegments = (uint8)pCSurface->aNodes.ElementAt(i)->au32AdiacentSegmentsIndexes.size();
		pstN->pu32AdiacentSegmentsIndexes = new uint32[pstN->u8NumAdiacentSegments];
		pstN->pi8AdiacentSegmentsEnabled = new int8[pstN->u8NumAdiacentSegments];	
		// set info about adiacent triangles
		for (uint32 j = 0; j < pstN->u8NumAdiacentTriangles; j++)
		{
			uint32 u32Tidx = pCSurface->aNodes.ElementAt(i)->au32AdiacentTrianglesIndexes.at(j);
			pstN->pu32AdiacentTrianglesIndexes[j] = u32Tidx;
			uint8 u8Poz = pCSurface->aNodes.ElementAt(i)->au8PozitionInAdiacentTriangles.at(j);
			pstN->pu8PozitionInAdiacentTriangles[j] = u8Poz;
			uint32 u32OpositeSegIdx = pCSurface->aTriangles.ElementAt(u32Tidx)->au32SegmentIdx[(u8Poz+1)%3];
			pstN->pu32NeighbouringSegmentsIndexes[j] = u32OpositeSegIdx;
			pCSurface->aTriangles.ElementAt(u32Tidx)->vComputeBisector(u8Poz, pstN->pf8AdiacentTrianglesBisectors+3*j);
		}		
		// set influence radius
		pstN->f8InfluenceRadius2 = 0;
		for (uint32 j = 0; j < pstN->u8NumAdiacentSegments; j++)
		{
			uint32 u32Sidx = pCSurface->aNodes.ElementAt(i)->au32AdiacentSegmentsIndexes.at(j);
			float8 f8R = pCSurface->aSegments.ElementAt(u32Sidx)->f8Length;
			float8 f8Dist = f8R*f8R + f8ActivationDistance*f8ActivationDistance;
			if (f8Dist > pstN->f8InfluenceRadius2) pstN->f8InfluenceRadius2 = f8Dist;
		}
		// set info about adiacent segments
		for (uint32 j = 0; j < pstN->u8NumAdiacentSegments; j++)
		{
			uint32 u32Sidx = pCSurface->aNodes.ElementAt(i)->au32AdiacentSegmentsIndexes.at(j);
			pstN->pu32AdiacentSegmentsIndexes[j] = u32Sidx;
			pstN->pi8AdiacentSegmentsEnabled[j] = 1;
		}

		// check which adiacent segments are influenced by each adiacent triangle
		uint8 u8SegmentNo;
		int32 i32IdxFound;
		for (uint32 j = 0; j < pstN->u8NumAdiacentTriangles; j++)
		{
			u8SegmentNo = 0;
			uint32 u32Tidx = pstN->pu32AdiacentTrianglesIndexes[j];
			uint32 u32SegIdx = pCSurface->aTriangles.ElementAt(u32Tidx)->au32SegmentIdx[0];
			if (pCSurface->aSegments.ElementAt(u32SegIdx)->i8ContainsNode(i) >= 0)
			{
				i32IdxFound = VECT_i32FindElement(pstN->pu32AdiacentSegmentsIndexes, u32SegIdx, pstN->u8NumAdiacentSegments);
				if (i32IdxFound < 0) exit(111);
				pstN->pu8SegmentsInfluencedByTriangle[2*j+u8SegmentNo] = (uint8)i32IdxFound;
				u8SegmentNo++;
			}
			u32SegIdx = pCSurface->aTriangles.ElementAt(u32Tidx)->au32SegmentIdx[1];
			if (pCSurface->aSegments.ElementAt(u32SegIdx)->i8ContainsNode(i) >= 0)
			{
				i32IdxFound = VECT_i32FindElement(pstN->pu32AdiacentSegmentsIndexes, u32SegIdx, pstN->u8NumAdiacentSegments);
				if (i32IdxFound < 0) exit(111);
				pstN->pu8SegmentsInfluencedByTriangle[2*j+u8SegmentNo] = (uint8)i32IdxFound;
				u8SegmentNo++;
			}
			u32SegIdx = pCSurface->aTriangles.ElementAt(u32Tidx)->au32SegmentIdx[2];
			if (pCSurface->aSegments.ElementAt(u32SegIdx)->i8ContainsNode(i) >= 0)
			{
				i32IdxFound = VECT_i32FindElement(pstN->pu32AdiacentSegmentsIndexes, u32SegIdx, pstN->u8NumAdiacentSegments);
				if (i32IdxFound < 0) exit(111);
				pstN->pu8SegmentsInfluencedByTriangle[2*j+u8SegmentNo] = (uint8)i32IdxFound;
				u8SegmentNo++;
			}
			if (u8SegmentNo != 2)
			{
				exit(112);
			}
		}
				
		// set normal components
		VECT_vCopy(pCSurface->aNodes.ElementAt(i)->af8NormalComponents, pstN->af8NormalComponents, 3);
		// aditional segments that must be checked
		pstN->u8NumAditionalSegmentsToCheck = (uint8)pCSurface->aNodes.ElementAt(i)->au32AditionalSegmentsToCheck.size();
		if (pstN->u8NumAditionalSegmentsToCheck > 0)
		{
			pstN->pu32AditionalSegmentsToCheckIndexes = new uint32[pstN->u8NumAditionalSegmentsToCheck];
			for (uint32 j = 0; j < pstN->u8NumAditionalSegmentsToCheck; j++)
			{
				pstN->pu32AditionalSegmentsToCheckIndexes[j] = pCSurface->aNodes.ElementAt(i)->au32AditionalSegmentsToCheck.at(j);
			}
		}
		else pstN->pu32AditionalSegmentsToCheckIndexes = NULL;
		// aditional triangles that must be checked
		pstN->u8NumAditionalTrianglesToCheck = (uint8)pCSurface->aNodes.ElementAt(i)->au32AditionalTrianglesToCheck.size();
		if (pstN->u8NumAditionalTrianglesToCheck > 0)
		{
			pstN->pu32AditionalTrianglesToCheckIndexes = new uint32[pstN->u8NumAditionalTrianglesToCheck];
			for (uint32 j = 0; j < pstN->u8NumAditionalTrianglesToCheck; j++)
			{
				pstN->pu32AditionalTrianglesToCheckIndexes[j] = pCSurface->aNodes.ElementAt(i)->au32AditionalTrianglesToCheck.at(j);
			}
		}
		else pstN->pu32AditionalTrianglesToCheckIndexes = NULL;
	}
	// initialize the buckets for bucket search
	// find maximum triangle side length along axis
	float8 f8dx, f8dy, f8dz, f8Tx, f8Ty, f8Tz;
	f8dx = 0; f8dy = 0; f8dz = 0;
	for (i = 0; i < u32NumTriangles; i++)
	{
		pCSurface->aTriangles.ElementAt(i)->vGetMaximumSegmentProjection(&f8Tx, &f8Ty, &f8Tz);
		if (f8Tx > f8dx) f8dx = f8Tx;
		if (f8Ty > f8dy) f8dy = f8Ty;
		if (f8Tz > f8dz) f8dz = f8Tz;
	}
	if (f8dx < f8ActivationDistance) f8dx = f8ActivationDistance/0.7;
	if (f8dy < f8ActivationDistance) f8dy = f8ActivationDistance/0.7;
	if (f8dz < f8ActivationDistance) f8dz = f8ActivationDistance/0.7;
	pBuckets = new CBuckets(u32NumMasterNodes, pf8NodeCoordinates, 0.7*f8dx, 0.7*f8dy, 0.7*f8dz);
}

int32 CContactPair::i32GetClosestNodeIndex(float8 *pf8Node, float8 *pf8MinDist)
{
	int32 i, j, k;
	uint32 i1, i2, j1, j2, k1, k2, n;
	// identify the bucket containing the node
	i = pBuckets->i32GetBucketIndexOnX(pf8Node[0]);
	if (i < 0) return -1;
	j = pBuckets->i32GetBucketIndexOnY(pf8Node[1]);
	if (j < 0) return -1;
	k = pBuckets->i32GetBucketIndexOnZ(pf8Node[2]);
	if (k < 0) return -1;
	// identify the indexes of surrounding buckets
	if (i > 0) i1 = i-1; else i1 = 0;
	if (j > 0) j1 = j-1; else j1 = 0;
	if (k > 0) k1 = k-1; else k1 = 0;
	if ((uint32)i < pBuckets->u32NumX - 1) i2 = i + 1; else i2 = pBuckets->u32NumX - 1;
	if ((uint32)j < pBuckets->u32NumY - 1) j2 = j + 1; else j2 = pBuckets->u32NumY - 1;
	if ((uint32)k < pBuckets->u32NumZ - 1) k2 = k + 1; else k2 = pBuckets->u32NumZ - 1;
	*pf8MinDist = pBuckets->f8BucketDiagonal2;
	int32 i32ClosestNodeIndex = -1;
	float8 f8Vx, f8Vy, f8Vz;
	float8 *pf8MasterNode;
	float8 f8D2;
	uint32 u32NumNodesInBucket;
	uint32 *pu32NodeIdxs;
	// Search in the bucket containing the node and in the surrounding buckets for master nodes
	for (i = i1; (uint32)i <= i2; i++)
	{
		for (j = j1; (uint32)j <= j2; j++)
		{
			for (k = k1; (uint32)k <= k2; k++)
			{
				pu32NodeIdxs = pBuckets->pu32GetNodesIndexesInBucket(i, j, k, &u32NumNodesInBucket);
				for (n = 0; n < u32NumNodesInBucket; n++)
				{
					pf8MasterNode = pf8NodeCoordinates + pu32NodeIdxs[n]*3;
					f8Vx = pf8MasterNode[0] - pf8Node[0];
					f8Vy = pf8MasterNode[1] - pf8Node[1];
					f8Vz = pf8MasterNode[2] - pf8Node[2];
					f8D2 = f8Vx*f8Vx + f8Vy*f8Vy + f8Vz*f8Vz;
					if (f8D2 < *pf8MinDist)
					{
						*pf8MinDist = f8D2;
						i32ClosestNodeIndex = pu32NodeIdxs[n];
					}
				}
			}
		}
	}
	return i32ClosestNodeIndex;
}

uint8 CContactPair::u8ApplyContactToNode(float8 *pf8Node)
{
	float8 f8MinDist2;
	int32 i32ClosestMasterNode = i32GetClosestNodeIndex(pf8Node, &f8MinDist2);
#if (DEBUG_DEBUG == 1)
	if ((DEBUG_u32NodeIdx == 1557) && DEBUG_boStartDebug)
		DEBUG_vPrintMessage("Closest node: %u.", DEBUG_pu32NodeIndexes[i32ClosestMasterNode]);
	
#endif
	// check if the node is close enough
	if (i32ClosestMasterNode < 0) return 0;
	tstMasterNodeInfo *pstN = pstMasterNodeInfo+i32ClosestMasterNode;
	float8 *pf8MasterNodeCoord = pf8NodeCoordinates + 3*i32ClosestMasterNode;
	f8ProjVectX = pf8Node[0] - pf8MasterNodeCoord[0];
	f8ProjVectY = pf8Node[1] - pf8MasterNodeCoord[1];
	f8ProjVectZ = pf8Node[2] - pf8MasterNodeCoord[2];
	boClosestToTriangle = false;
	boClosestToSegment = false;
	if (f8MinDist2 > pstN->f8InfluenceRadius2) 
	{
		return u8PerformAditionalChecks(pf8Node, i32ClosestMasterNode, &f8MinDist2);
	}
	// find triangles on which the node may project
	uint8 u8NumAdiacentTriangles = pstN->u8NumAdiacentTriangles;
	uint8 i;
	float8 f8P1Nx, f8P1Ny, f8P1Nz;
	f8P1Nx = f8ProjVectX;
	f8P1Ny = f8ProjVectY;
	f8P1Nz = f8ProjVectZ;
	float8 *pf8Bisector;
	for (i = 0; i < u8NumAdiacentTriangles; i++)
	{
		// check the scalar product with the triangle bisector
		pf8Bisector = pstN->pf8AdiacentTrianglesBisectors + 3*i;
		if (pf8Bisector[0]*f8P1Nx + pf8Bisector[1]*f8P1Ny + pf8Bisector[2]*f8P1Nz > 0)
		{
			if (boCheckTriangle(pf8Node, pstN->pu32AdiacentTrianglesIndexes[i],
				pstN->pu8PozitionInAdiacentTriangles[i], pstN->pu32NeighbouringSegmentsIndexes[i], &f8MinDist2))
			{
				//mark segments that surround this triangle
				uint8 u8SegIdx = pstN->pu8SegmentsInfluencedByTriangle[2*i];
				pstN->pi8AdiacentSegmentsEnabled[u8SegIdx] = -1;
				u8SegIdx = pstN->pu8SegmentsInfluencedByTriangle[2*i+1];
				pstN->pi8AdiacentSegmentsEnabled[u8SegIdx] = -1;
			}
		}
	}
	// check surrounding segments
	uint8 u8NumAdiacentsegments = pstN->u8NumAdiacentSegments;
	uint32 u32SegIdx;
	for (i = 0; i < u8NumAdiacentsegments; i++)
	{
		if (pstN->pi8AdiacentSegmentsEnabled[i] > 0)
		{
			u32SegIdx = pstN->pu32AdiacentSegmentsIndexes[i];
			vCheckSegment(pf8Node, u32SegIdx, &f8MinDist2);
		}
		else pstN->pi8AdiacentSegmentsEnabled[i] = 1; //re-enable segment
	}
	// no matter what is closer, perform aditional checks (this will finalize the analysis)
	return u8PerformAditionalChecks(pf8Node, i32ClosestMasterNode, &f8MinDist2);
}

void CContactPair::vCheckSegment(float8 *pf8Node, uint32 u32SegIdx, float8 *pf8MinDist2)
{
	tstMasterSegmentInfo *pstS;
	float8 f8P1Px, f8P1Py, f8P1Pz, r;
	pstS = pstMasterSegmentInfo + u32SegIdx;
	float8 *pf8P1 = pstS->af8Node1Coordinates;
	float8 *pf8P1P2 = pstS->af8VectorComponents;
	f8P1Px = pf8Node[0] - pf8P1[0];
	f8P1Py = pf8Node[1] - pf8P1[1];
	f8P1Pz = pf8Node[2] - pf8P1[2];
	r = (f8P1Px*pf8P1P2[0] + f8P1Py*pf8P1P2[1] + f8P1Pz*pf8P1P2[2])/pstS->f8Length2;
	if ((r < 0) || (r > 1)) return;
	f8P1Px -= r*pf8P1P2[0];
	f8P1Py -= r*pf8P1P2[1];
	f8P1Pz -= r*pf8P1P2[2];
	r = f8P1Px*f8P1Px + f8P1Py*f8P1Py + f8P1Pz*f8P1Pz;
	if (r < *pf8MinDist2)
	{
		*pf8MinDist2 = r;
		boClosestToSegment = true;
		boClosestToTriangle = false;
		u32ClosestSegment = u32SegIdx;
		f8ProjVectX = f8P1Px;
		f8ProjVectY = f8P1Py;
		f8ProjVectZ = f8P1Pz;
	}
}

bool CContactPair::boCheckNeighbouringTriangle(float8 *pf8Node, uint32 u32TriangleIdx, float8 *pf8MinDist2)
{
	tstMasterTriangleInfo *pstT = pstMasterTriangleInfo+u32TriangleIdx;
	float8 f8P1Px, f8P1Py, f8P1Pz, r, s, b1, b2;
	float8 *pf8P1 = pstT->af8Node1Coordinates;
	float8 *pf8V12 = pstT->af8V12Components;
	float8 *pf8V13 = pstT->af8V13Components;
	f8P1Px = pf8Node[0] - pf8P1[0];
	f8P1Py = pf8Node[1] - pf8P1[1];
	f8P1Pz = pf8Node[2] - pf8P1[2];
	b1 = pf8V12[0]*f8P1Px + pf8V12[1]*f8P1Py + pf8V12[2]*f8P1Pz;
	b2 = pf8V13[0]*f8P1Px + pf8V13[1]*f8P1Py + pf8V13[2]*f8P1Pz;
	r = (b1*pstT->a22 - b2*pstT->a12)/pstT->det;
	s = (b2*pstT->a11 - b1*pstT->a12)/pstT->det;
	if ((s >= 1) || (s <= 0)) return false;  
	if ((r >= 1) || (r <= 0)) return false;
	if (r + s < 1)
	{
		// projection is on this triangle - check distance
		f8P1Px -= (r*pf8V12[0]+s*pf8V13[0]);
		f8P1Py -= (r*pf8V12[1]+s*pf8V13[1]);
		f8P1Pz -= (r*pf8V12[2]+s*pf8V13[2]);
		r = f8P1Px*f8P1Px + f8P1Py*f8P1Py + f8P1Pz*f8P1Pz;
		if (r < *pf8MinDist2)
		{
			*pf8MinDist2 = r;
			boClosestToSegment = false;
			boClosestToTriangle = true;
			u32ClosestTriangle = u32TriangleIdx;
			f8ProjVectX = f8P1Px;
			f8ProjVectY = f8P1Py;
			f8ProjVectZ = f8P1Pz;
		}
		return true;
	}
	return false;
}

bool CContactPair::boCheckTriangle(float8 *pf8Node, uint32 u32TriangleIdx, uint8 u8PozitionOfMasterNode, 
		uint32 u32NeighbouringSegment, float8 *pf8MinDist2)
{
	tstMasterTriangleInfo *pstT = pstMasterTriangleInfo+u32TriangleIdx;
	float8 f8P1Px, f8P1Py, f8P1Pz, r, s, b1, b2, r1, s1, t1;
	uint8 i;
	float8 af8InternalCoord[3];
	float8 *pf8P1 = pstT->af8Node1Coordinates;
	float8 *pf8V12 = pstT->af8V12Components;
	float8 *pf8V13 = pstT->af8V13Components;
	f8P1Px = pf8Node[0] - pf8P1[0];
	f8P1Py = pf8Node[1] - pf8P1[1];
	f8P1Pz = pf8Node[2] - pf8P1[2];
	b1 = pf8V12[0]*f8P1Px + pf8V12[1]*f8P1Py + pf8V12[2]*f8P1Pz;
	b2 = pf8V13[0]*f8P1Px + pf8V13[1]*f8P1Py + pf8V13[2]*f8P1Pz;
	r = (b1*pstT->a22 - b2*pstT->a12)/pstT->det;
	s = (b2*pstT->a11 - b1*pstT->a12)/pstT->det;
	af8InternalCoord[0] = 1-r-s;
	af8InternalCoord[1] = r;
	af8InternalCoord[2] = s;
	r1 = af8InternalCoord[u8PozitionOfMasterNode];
	i = u8PozitionOfMasterNode + 1;
	if (i > 2) i = 0;
	s1 = af8InternalCoord[i];
	i++;
	if (i > 2) i = 0;
	t1 = af8InternalCoord[i];
	if ((s1 > 1) || (s1 < 0)) return false;  
	if ((t1 > 1) || (t1 < 0)) return false;
	if (r1 > 1) return false;
	if (r1 > 0)
	{
		// projection is on this triangle - check distance
		f8P1Px -= (r*pf8V12[0]+s*pf8V13[0]);
		f8P1Py -= (r*pf8V12[1]+s*pf8V13[1]);
		f8P1Pz -= (r*pf8V12[2]+s*pf8V13[2]);
		r = f8P1Px*f8P1Px + f8P1Py*f8P1Py + f8P1Pz*f8P1Pz;
		if (r < *pf8MinDist2)
		{
			*pf8MinDist2 = r;
			boClosestToSegment = false;
			boClosestToTriangle = true;
			u32ClosestTriangle = u32TriangleIdx;
			f8ProjVectX = f8P1Px;
			f8ProjVectY = f8P1Py;
			f8ProjVectZ = f8P1Pz;
		}
	}
	else
	{
		// check neighbouring segment 
		vCheckSegment(pf8Node, u32NeighbouringSegment, pf8MinDist2);
	}
	return true;
}

uint8 CContactPair::u8PerformAditionalChecks(float8 *pf8Node, uint32 u32ClosestNodeIdx, float8 *pf8MinDist2)
{
	uint32 i;
	// perform aditional checks for triangles
	tstMasterNodeInfo *pstN = pstMasterNodeInfo+u32ClosestNodeIdx;
	uint8 u8AditionalTrianglesToCheck = pstN->u8NumAditionalTrianglesToCheck;
	uint32 u32Tidx;
	for (i = 0; i < u8AditionalTrianglesToCheck; i++)
	{
		u32Tidx = pstN->pu32AditionalTrianglesToCheckIndexes[i];
		boCheckNeighbouringTriangle(pf8Node, u32Tidx, pf8MinDist2);
	}
	// perform aditional checks for segments
	tstMasterSegmentInfo *pstS;
	uint8 u8AditionalSegmentsToCheck = pstN->u8NumAditionalSegmentsToCheck;
	uint32 u32Sidx;
	int32 i32Tidx;
	for (i = 0; i < u8AditionalSegmentsToCheck; i++)
	{
		u32Sidx = pstN->pu32AditionalSegmentsToCheckIndexes[i];
		pstS = pstMasterSegmentInfo + u32Sidx;
		i32Tidx = pstS->ai32AdiacentTriangles[0];
		vCheckSegment(pf8Node, u32Sidx, pf8MinDist2);
	}
	// all checks done, check for penetration and bring node to surface
	// get normal to surface
	float8 *pf8Normal;
	if (boClosestToTriangle)
	{
		pf8Normal = pstMasterTriangleInfo[u32ClosestTriangle].af8NormalComponents;
	}
	else if (boClosestToSegment)
	{
		if (pstMasterSegmentInfo[u32ClosestSegment].u8OnSurfaceEdge > 0) return 0;
		pf8Normal = pstMasterSegmentInfo[u32ClosestSegment].af8NormalComponents;
	}
	else 
	{
		if (pstN->u8OnSurfaceEdge > 0) return 0;
		pf8Normal = pstN->af8NormalComponents;
	}
	// check for penetration
	float8 f8Penetration = f8ProjVectX*pf8Normal[0] + f8ProjVectY*pf8Normal[1] + f8ProjVectZ*pf8Normal[2];
	if (f8Penetration < 0)
	{
		if (-f8Penetration < f8MaxPenetration)
		{
			// bring node to surface
			pf8Node[0] -= f8ProjVectX;
			pf8Node[1] -= f8ProjVectY;
			pf8Node[2] -= f8ProjVectZ;
			return 1;
		}
	}
	return 0;
}

void CContactPair::vApplyContacts(float8 *pf8CurrentNodePositions, uint8 *pu8NodeInContact)
{
	uint32 i, u32NodeIdx;
	for (i = 0; i < u32NumContactNodes; i++)
	{
		u32NodeIdx = pu32ContactNodesIdx[i];
		pu8NodeInContact[u32NodeIdx] = u8ApplyContactToNode(pf8CurrentNodePositions+3*u32NodeIdx);
	}
}



