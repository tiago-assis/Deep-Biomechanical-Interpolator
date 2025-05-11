#if !defined(_UTILS_H_)
#define _UTILS_H_

#include <stdio.h>
#include <stdio.h>
#include "cdef.h"
#include "vect.h"
#include <vector>

class CIndexArray
{
public:
	CIndexArray():u32Size(0), pu32Indexes(NULL) {};
	CIndexArray(uint32 u32Sz, uint32 *pu32Idx);
	~CIndexArray() {if (pu32Indexes != NULL) delete[] pu32Indexes;};
	CIndexArray(const CIndexArray &s);
	CIndexArray& operator=(const CIndexArray &s);
	uint32 &operator[](const uint32 i) {return pu32Indexes[i];};
	uint32 u32GetSize(void) {return u32Size;};
	uint32 *pu32GetIndexes(void) {return pu32Indexes;};
	void vSetSize(uint32 u32Sz); 
	void vSetIndexes(uint32 *pu32Idx) {VECT_vCopy(pu32Idx, pu32Indexes, u32Size);};
	void vSetData(uint32 u32Sz, uint32 *pu32Idx);
	void vSaveBinary(FILE *pf);
	void vReadBinary(FILE *pf);
	static void vSkipBinary(FILE *pf);
private:
	uint32 u32Size;
	uint32 *pu32Indexes;
};

template <class T> class CMyArray
{
public:
	CMyArray(void) {};
	virtual ~CMyArray(void) {vRemoveAll();};
	T* &ElementAt(uint32 u32Idx) {return apTArray.at(u32Idx);};
	void vRemoveAll(void);
	void vRemoveAt(uint32 u32Idx);
	void vAdd(T *pT) {apTArray.push_back(pT);};
	void vInsertAt(uint32 u32Idx, T *pT) {apTArray.InsertAt(u32Idx, pT);};
	uint32 u32GetCount(void) {return (uint32)apTArray.size();};
	T* &operator[](const uint32 i) {return apTArray.at(i);};
private:
	std::vector<T*> apTArray;
};


template <class T>
void CMyArray<T>::vRemoveAll(void)
{
	uint32 n = (uint32)apTArray.size();
	T* pT;
	for(uint32 i = 0; i < n; i++)
	{
		pT = apTArray.at(i);
		if (pT != NULL) delete pT;
	}
	apTArray.clear();
}

template <class T>
void CMyArray<T>::vRemoveAt(uint32 u32Idx)
{
	T* pT = apTArray.at(u32Idx);
	if (pT != NULL) delete pT;
	apTArray.erase(apTArray.begin()+u32Idx);
}




#endif