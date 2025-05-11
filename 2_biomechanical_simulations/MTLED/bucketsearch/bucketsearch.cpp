/****************************************************************************** 
 * 
 *  file:  bucketsearch.cpp
 * 
 *  Copyright (c) 2016, Grand R. Joldes.
 *  All rights reverved.
 * 
 *  
 *  THE SOFTWARE IS PROVIDED _AS IS_, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
 *  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 *  DEALINGS IN THE SOFTWARE.  
 *  
 *****************************************************************************/ 
#include <boost/thread/thread.hpp>
#include <boost/atomic.hpp>

#include "bucketsearch.hpp"

namespace BucketSearch {

buckets::buckets(double _xMin, double _yMin, double _zMin, double _sizeX, double _sizeY,
	double _sizeZ, unsigned int _numX, unsigned int _numY, unsigned int _numZ)
{
	stBuckets_.xMin = _xMin; stBuckets_.yMin = _yMin; stBuckets_.zMin = _zMin;
	stBuckets_.sizeX = _sizeX; stBuckets_.sizeY = _sizeY; stBuckets_.sizeZ = _sizeZ;
	stBuckets_.numX = _numX; stBuckets_.numY = _numY; stBuckets_.numZ = _numZ;
	stBuckets_.uiStrideZ = _numX*_numY;
	stBuckets_.uiNumBuckets = stBuckets_.uiStrideZ*_numZ;	
};

BucketSearch::BucketSearch(buckets *pBuckets_, double *pPointsCoord_, unsigned int uiNumPoints_):pBuckets(pBuckets_), pPointsCoord(pPointsCoord_), uiNumPoints(uiNumPoints_)
{
	pstBucketedPoints = new tstBucketedPoints;
	pstBucketedPoints->puiNumPointsPerBucket = new unsigned int[pBuckets_->uiGetNumberOfBuckets()];
	pstBucketedPoints->puiStartingIndex = new unsigned int[pBuckets_->uiGetNumberOfBuckets()];
	pstBucketedPoints->puiLastIndex = new unsigned int[pBuckets_->uiGetNumberOfBuckets()];
	pstBucketedPoints->puiNextIndex = new unsigned int[uiNumPoints];
}

BucketSearch::~BucketSearch()
{
	delete[] pstBucketedPoints->puiNumPointsPerBucket;
	delete[] pstBucketedPoints->puiStartingIndex;
	delete[] pstBucketedPoints->puiLastIndex;
	delete[] pstBucketedPoints->puiNextIndex;
	delete pstBucketedPoints;
}

void BucketSearch::vDistributePointsToBuckets(void)
{
	// init memory
	unsigned int uiNumBuckets = pBuckets->uiGetNumberOfBuckets();
	unsigned int i, pointIndex;
	for (i = 0; i < uiNumBuckets; i++)
	{
		vSetNumPointsInBucket(i, 0);
	}
	// distribute points in buckets
	for (pointIndex = 0; pointIndex < uiNumPoints; pointIndex++)
	{
		double xp = pPointsCoord[3*pointIndex];
		double yp = pPointsCoord[3*pointIndex+1];
		double zp = pPointsCoord[3*pointIndex+2];
		unsigned int idxBucketX = pBuckets->uiGetBucketIndexX(xp);
		unsigned int idxBucketY = pBuckets->uiGetBucketIndexY(yp);
		unsigned int idxBucketZ = pBuckets->uiGetBucketIndexZ(zp);
		unsigned int idxLinear = pBuckets->uiGetLinearBucketIndex(idxBucketX, idxBucketY, idxBucketZ);
		if (uiGetNumPointsInBucket(idxLinear) == 0)
		{
			vSetNumPointsInBucket(idxLinear, 1);
			vSetFirstPointInBucket(idxLinear, pointIndex);
			vSetLastPointInBucket(idxLinear, pointIndex);
		} else 
		{
			unsigned int uiLastPointIdx = uiGetLastPointInBucket(idxLinear);
			vSetNextPointInBucket(uiLastPointIdx, pointIndex);
			vSetLastPointInBucket(idxLinear, pointIndex);
			pstBucketedPoints->puiNumPointsPerBucket[idxLinear]++;
		}
	}
}

void BucketSearch::vFindNeigbours(double *pdPoint, const std::vector<double> &influence_radiuses, std::vector<uint32_t> &neighbors_indices)
{
	// find bucket containing particle
	double xp = pdPoint[0];
	double yp = pdPoint[1];
	double zp = pdPoint[2];
	unsigned int uiBucketIdxX = pBuckets->uiGetBucketIndexX(xp);
	unsigned int uiBucketIdxY = pBuckets->uiGetBucketIndexY(yp);
	unsigned int uiBucketIdxZ = pBuckets->uiGetBucketIndexZ(zp);
	// set indexes of buckets to search for neighbours
	unsigned int uiMinBucketIdxX, uiMaxBucketIdxX;
	unsigned int uiMinBucketIdxY, uiMaxBucketIdxY;
	unsigned int uiMinBucketIdxZ, uiMaxBucketIdxZ;
	if (uiBucketIdxX > 0) uiMinBucketIdxX = uiBucketIdxX - 1;
	else uiMinBucketIdxX = uiBucketIdxX;
	uiMaxBucketIdxX = uiBucketIdxX + 1;
	if (uiMaxBucketIdxX >= pBuckets->uiGetNumBucketsX()) uiMaxBucketIdxX = uiBucketIdxX;
	if (uiBucketIdxY > 0) uiMinBucketIdxY = uiBucketIdxY - 1;
	else uiMinBucketIdxY = uiBucketIdxY;
	uiMaxBucketIdxY = uiBucketIdxY + 1;
	if (uiMaxBucketIdxY >= pBuckets->uiGetNumBucketsY()) uiMaxBucketIdxY = uiBucketIdxY;
	if (uiBucketIdxZ > 0) uiMinBucketIdxZ = uiBucketIdxZ - 1;
	else uiMinBucketIdxZ = uiBucketIdxZ;
	uiMaxBucketIdxZ = uiBucketIdxZ + 1;
	if (uiMaxBucketIdxZ >= pBuckets->uiGetNumBucketsZ()) uiMaxBucketIdxZ = uiBucketIdxZ;
	
	// Clear neighbors container
	neighbors_indices.clear();

	// go through the neighbouring buckets
	for (unsigned int idxZ = uiMinBucketIdxZ; idxZ <= uiMaxBucketIdxZ; idxZ++)
	{
		for (unsigned int idxY = uiMinBucketIdxY; idxY <= uiMaxBucketIdxY; idxY++)
		{
			for (unsigned int idxX = uiMinBucketIdxX; idxX <= uiMaxBucketIdxX; idxX++)
			{
				unsigned int linearIdx = pBuckets->uiGetLinearBucketIndex(idxX, idxY, idxZ);
				unsigned int numPointsInBucket = uiGetNumPointsInBucket(linearIdx);
				uint32_t uiNeighbourIdx;
				for (unsigned int pIdx = 0; pIdx < numPointsInBucket; pIdx++)
				{
					if (pIdx == 0) uiNeighbourIdx = uiGetFirstPointInBucket(linearIdx);
					else uiNeighbourIdx = uiGetNextPointInBucket(uiNeighbourIdx);
					// check distance
					double xn = pPointsCoord[3 * uiNeighbourIdx];
					double yn = pPointsCoord[3 * uiNeighbourIdx + 1];
					double zn = pPointsCoord[3 * uiNeighbourIdx + 2];
					double dist2 = (xn - xp) * (xn - xp) + (yn - yp) * (yn - yp) + (zn - zp) * (zn - zp);
					double dRadius = influence_radiuses[uiNeighbourIdx];
					if (dist2 < dRadius * dRadius)
					{
						neighbors_indices.emplace_back(uiNeighbourIdx);
					}
				}
			}
		}
	}
}

#if 0
void BucketSearch::vOperateOnNeighbours(boost::function2<void, unsigned int, unsigned int> vOperation, unsigned int uiDimensions, unsigned int uiPointIdx)
{
	// find bucket containing particle
	float xp = pPointsCoord[3*uiPointIdx];
	float yp = pPointsCoord[3*uiPointIdx+1];
	float zp = pPointsCoord[3*uiPointIdx+2];
	unsigned int uiBucketIdxX = pBuckets->uiGetBucketIndexX(xp);
	unsigned int uiBucketIdxY = pBuckets->uiGetBucketIndexY(yp);
	unsigned int uiBucketIdxZ = pBuckets->uiGetBucketIndexZ(zp);
	// set indexes of buckets to search for neighbours
	unsigned int uiMinBucketIdxX, uiMaxBucketIdxX;
	unsigned int uiMinBucketIdxY, uiMaxBucketIdxY;
	unsigned int uiMinBucketIdxZ, uiMaxBucketIdxZ;
	if (uiBucketIdxX > 0) uiMinBucketIdxX = uiBucketIdxX-1;
	else uiMinBucketIdxX = uiBucketIdxX;
	uiMaxBucketIdxX = uiBucketIdxX+1;
	if (uiMaxBucketIdxX >= pBuckets->uiGetNumBucketsX()) uiMaxBucketIdxX = uiBucketIdxX;
	if (uiBucketIdxY > 0) uiMinBucketIdxY = uiBucketIdxY-1;
	else uiMinBucketIdxY = uiBucketIdxY;
	uiMaxBucketIdxY = uiBucketIdxY+1;
	if (uiMaxBucketIdxY >= pBuckets->uiGetNumBucketsY()) uiMaxBucketIdxY = uiBucketIdxY;
	if (uiDimensions == 3)
	{
		if (uiBucketIdxZ > 0) uiMinBucketIdxZ = uiBucketIdxZ-1;
		else uiMinBucketIdxZ = uiBucketIdxZ;
		uiMaxBucketIdxZ = uiBucketIdxZ+1;
		if (uiMaxBucketIdxZ >= pBuckets->uiGetNumBucketsZ()) uiMaxBucketIdxZ = uiBucketIdxZ;
	}
	else
	{
		uiMinBucketIdxZ = uiBucketIdxZ;
		uiMaxBucketIdxZ = uiBucketIdxZ;
	}	
	// go through the neighbouring buckets
	for (unsigned int idxZ = uiMinBucketIdxZ; idxZ <= uiMaxBucketIdxZ; idxZ++)
	{
		for (unsigned int idxY = uiMinBucketIdxY; idxY <= uiMaxBucketIdxY; idxY++)
		{
			for (unsigned int idxX = uiMinBucketIdxX; idxX <= uiMaxBucketIdxX; idxX++)
			{
				unsigned int linearIdx = pBuckets->uiGetLinearBucketIndex(idxX, idxY,idxZ);
				unsigned int numPointsInBucket = uiGetNumPointsInBucket(linearIdx);
				unsigned int uiNeighbourIdx;
				for (unsigned int pIdx = 0; pIdx < numPointsInBucket; pIdx++)
				{
					if (pIdx == 0) uiNeighbourIdx = uiGetFirstPointInBucket(linearIdx);
					else uiNeighbourIdx = uiGetNextPointInBucket(uiNeighbourIdx);
					// apply operation
					if (uiPointIdx != uiNeighbourIdx)
					{
						vOperation(uiPointIdx, uiNeighbourIdx);
					}
				}
			}
		}
	}
}

void BucketSearch::vOperateOnNeighboursUsingMask(boost::function2<void, unsigned int, unsigned int> vOperation, unsigned int uiDimensions, unsigned int uiPointIdx)
{
	// find bucket containing particle
	float xp = pPointsCoord[3*uiPointIdx];
	float yp = pPointsCoord[3*uiPointIdx+1];
	float zp = pPointsCoord[3*uiPointIdx+2];
	unsigned int uiBucketIdxX = pBuckets->uiGetBucketIndexX(xp);
	unsigned int uiBucketIdxY = pBuckets->uiGetBucketIndexY(yp);
	unsigned int uiBucketIdxZ = pBuckets->uiGetBucketIndexZ(zp);
	// first operate on particles from same bucket
	unsigned int linearIdx = pBuckets->uiGetLinearBucketIndex(uiBucketIdxX, uiBucketIdxY,uiBucketIdxZ);
	unsigned int numPointsInBucket = uiGetNumPointsInBucket(linearIdx);
	unsigned int uiNeighbourIdx;
	for (unsigned int pIdx = 0; pIdx < numPointsInBucket; pIdx++)
	{
		if (pIdx == 0) uiNeighbourIdx = uiGetFirstPointInBucket(linearIdx);
		else uiNeighbourIdx = uiGetNextPointInBucket(uiNeighbourIdx);
		// apply operation
		if (uiPointIdx < uiNeighbourIdx)  // make sure interaction is unique
		{
			vOperation(uiPointIdx, uiNeighbourIdx);
		}
	}
	//////// operate on buckets in the same z plane //////
	unsigned int uiMinBucketIdxX, uiMaxBucketIdxX;
	unsigned int uiMinBucketIdxY, uiMaxBucketIdxY;
	unsigned int uiMinBucketIdxZ;
	// operate on buckets on the same y plane
	if (uiBucketIdxX > 0) 
	{
		uiMinBucketIdxX = uiBucketIdxX-1;
		linearIdx = pBuckets->uiGetLinearBucketIndex(uiMinBucketIdxX, uiBucketIdxY,uiBucketIdxZ);
		numPointsInBucket = uiGetNumPointsInBucket(linearIdx);
		for (unsigned int pIdx = 0; pIdx < numPointsInBucket; pIdx++)
		{
			if (pIdx == 0) uiNeighbourIdx = uiGetFirstPointInBucket(linearIdx);
			else uiNeighbourIdx = uiGetNextPointInBucket(uiNeighbourIdx);
			vOperation(uiPointIdx, uiNeighbourIdx);
		}

	}
	else uiMinBucketIdxX = uiBucketIdxX;
	// operate on buckets on plane y-1
	uiMaxBucketIdxX = uiBucketIdxX+1;
	if (uiMaxBucketIdxX >= pBuckets->uiGetNumBucketsX()) uiMaxBucketIdxX = uiBucketIdxX;
	if (uiBucketIdxY > 0) 
	{
		uiMinBucketIdxY = uiBucketIdxY-1;
		for (unsigned int idxX = uiMinBucketIdxX; idxX <= uiMaxBucketIdxX; idxX++)
		{
			linearIdx = pBuckets->uiGetLinearBucketIndex(idxX, uiMinBucketIdxY,uiBucketIdxZ);
			numPointsInBucket = uiGetNumPointsInBucket(linearIdx);
			for (unsigned int pIdx = 0; pIdx < numPointsInBucket; pIdx++)
			{
				if (pIdx == 0) uiNeighbourIdx = uiGetFirstPointInBucket(linearIdx);
				else uiNeighbourIdx = uiGetNextPointInBucket(uiNeighbourIdx);
				// apply operation
				vOperation(uiPointIdx, uiNeighbourIdx);
			}
		}
	}
	else uiMinBucketIdxY = uiBucketIdxY;
	//////// operate on buckets in the z-1 plane //////
	if (uiDimensions == 3)
	{
		if (uiBucketIdxZ > 0)
		{
			uiMinBucketIdxZ = uiBucketIdxZ-1;
			uiMaxBucketIdxY = uiBucketIdxY+1;
			if (uiMaxBucketIdxY >= pBuckets->uiGetNumBucketsY()) uiMaxBucketIdxY = uiBucketIdxY;
			for (unsigned int idxY = uiMinBucketIdxY; idxY <= uiMaxBucketIdxY; idxY++)
			{
				for (unsigned int idxX = uiMinBucketIdxX; idxX <= uiMaxBucketIdxX; idxX++)
				{
					linearIdx = pBuckets->uiGetLinearBucketIndex(idxX, idxY, uiMinBucketIdxZ);
					numPointsInBucket = uiGetNumPointsInBucket(linearIdx);
					for (unsigned int pIdx = 0; pIdx < numPointsInBucket; pIdx++)
					{
						if (pIdx == 0) uiNeighbourIdx = uiGetFirstPointInBucket(linearIdx);
						else uiNeighbourIdx = uiGetNextPointInBucket(uiNeighbourIdx);
						vOperation(uiPointIdx, uiNeighbourIdx);
						
					}
				}
			}
		}
	}
}
#endif
} // namespace