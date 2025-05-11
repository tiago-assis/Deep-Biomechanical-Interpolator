/****************************************************************************** 
 * 
 *  file:  bucketsearch.hpp
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

#ifndef __BUCKETSEARCH_HPP__
#define __BUCKETSEARCH_HPP__

namespace BucketSearch {

class buckets {
public:
	/* Typedef for Buckets definition. */
	typedef struct stBuckets {
		double xMin, yMin, zMin;		// origin
		double sizeX, sizeY, sizeZ;  // bucket size
		unsigned int numX, numY, numZ; // number of buckets
		unsigned int uiNumBuckets;
		unsigned int uiStrideZ;
	} tstBuckets;
	buckets(double _xMin, double _yMin, double _zMin, double _sizeX, double _sizeY,
		double _sizeZ, unsigned int _numX, unsigned int _numY, unsigned int _numZ);
	~buckets() {};
	unsigned int uiGetNumBucketsX(void) {return stBuckets_.numX; };
	unsigned int uiGetNumBucketsY(void) {return stBuckets_.numY; };
	unsigned int uiGetNumBucketsZ(void) {return stBuckets_.numZ; };
	unsigned int uiGetBucketIndexX(double xCoord) {return (unsigned int)((xCoord-stBuckets_.xMin)/stBuckets_.sizeX);};
	unsigned int uiGetBucketIndexY(double yCoord) {return (unsigned int)((yCoord-stBuckets_.yMin)/stBuckets_.sizeY);};
	unsigned int uiGetBucketIndexZ(double zCoord) {return (unsigned int)((zCoord-stBuckets_.zMin)/stBuckets_.sizeZ);};
	unsigned int uiGetNumberOfBuckets(void) {return stBuckets_.uiNumBuckets;};
	unsigned int uiGetLinearBucketIndex(unsigned int idxX, unsigned int idxY, unsigned int idxZ) {
		return (idxX + stBuckets_.numX*idxY + stBuckets_.uiStrideZ*idxZ); };
	tstBuckets *pstGetBuckets(void) {return &stBuckets_;};

private:
	tstBuckets stBuckets_;
};

class BucketSearch {
public:
	typedef struct stBucketedPoints {
		unsigned int *puiNumPointsPerBucket; // Number of points in each bucket; must be allocated to numX*numY*numZ
		unsigned int *puiStartingIndex;		// Index of the first point in each bucket; must be allocated to numX*numY*numZ
		unsigned int *puiLastIndex;		// Index of the last point in each bucket; must be allocated to numX*numY*numZ
		unsigned int *puiNextIndex;	    // Index of next point in each bucket; must be allocated to uiNumPoints
	} tstBucketedPoints;

	BucketSearch(buckets *pBuckets_, double *pdPointsCoords, unsigned int uiNumPoints);
	~BucketSearch();

	void vDistributePointsToBuckets(void);
	
	/* Access functions */
	unsigned int *puiGetNumPointsPerBucketPointer(void) {return pstBucketedPoints->puiNumPointsPerBucket;};
	unsigned int *puiGetStartingIndexPointer(void) {return pstBucketedPoints->puiStartingIndex;};
	unsigned int *puiGetLastIndexPointer(void) {return pstBucketedPoints->puiLastIndex;};
	unsigned int *puiGetNextIndexPointer(void) {return pstBucketedPoints->puiNextIndex;};
	// number of points in a given bucket
	unsigned int uiGetNumPointsInBucket(unsigned int uiLinearBucketIndex) { 
		return pstBucketedPoints->puiNumPointsPerBucket[uiLinearBucketIndex]; };

	// index of first point in a bucket
	unsigned int uiGetFirstPointInBucket(unsigned int uiLinearBucketIndex) {
		return pstBucketedPoints->puiStartingIndex[uiLinearBucketIndex]; };

	// index of last point placed in a bucket
	unsigned int uiGetLastPointInBucket(unsigned int uiLinearBucketIndex) {
		return pstBucketedPoints->puiLastIndex[uiLinearBucketIndex]; };

	// index of next point in bucket
	unsigned int uiGetNextPointInBucket(unsigned int uiPreviousPointIndex) {
		return pstBucketedPoints->puiNextIndex[uiPreviousPointIndex]; };

	void vSetNumPointsInBucket(unsigned int uiLinearBucketIndex, unsigned int uiNum) { 
		pstBucketedPoints->puiNumPointsPerBucket[uiLinearBucketIndex] = uiNum; };

	// index of first point in a bucket
	void vSetFirstPointInBucket(unsigned int uiLinearBucketIndex, unsigned int idx) {
		pstBucketedPoints->puiStartingIndex[uiLinearBucketIndex] = idx; };

	// index of last point placed in a bucket
	void vSetLastPointInBucket(unsigned int uiLinearBucketIndex, unsigned int idx) {
		pstBucketedPoints->puiLastIndex[uiLinearBucketIndex] = idx;; };

	// index of next point in bucket
	void vSetNextPointInBucket(unsigned int uiPreviousPointIndex, unsigned int idx) {
		pstBucketedPoints->puiNextIndex[uiPreviousPointIndex] = idx; };

	buckets *pGetBucketsPointer(void) {return pBuckets;};
	unsigned int uiGetNumPoints(void) {return uiNumPoints;};

	void vFindNeigbours(double *pdPoint, const std::vector<double> &influence_radiuses, std::vector<uint32_t> &neighbors_indices);

protected:
	tstBucketedPoints *pstBucketedPoints;
	buckets *pBuckets;
	double *pPointsCoord;
	unsigned int uiNumPoints;
};

} // namespace
#endif /* __BUCKETSEARCH_HPP__ */