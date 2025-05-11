#include "utils.h"

CIndexArray::CIndexArray(uint32 u32Sz, uint32 *pu32Idx)
: u32Size(u32Sz)
{
	pu32Indexes = new uint32[u32Sz];
	VECT_vCopy(pu32Idx, pu32Indexes, u32Sz);
}

CIndexArray::CIndexArray(const CIndexArray &s)
: u32Size(s.u32Size)
{
	pu32Indexes = new uint32[u32Size];
	VECT_vCopy(s.pu32Indexes, pu32Indexes, u32Size);
}

CIndexArray& CIndexArray::operator=(const CIndexArray &s)
{
	if (pu32Indexes != NULL) delete[] pu32Indexes;
	u32Size = s.u32Size;
	pu32Indexes = new uint32[u32Size];
	VECT_vCopy(s.pu32Indexes, pu32Indexes, u32Size);
	return *this;
}

void CIndexArray::vSetSize(uint32 u32Sz)
{
	if (pu32Indexes != NULL) delete[] pu32Indexes;
	u32Size = u32Sz;
	pu32Indexes = new uint32[u32Sz];
}

void CIndexArray::vSetData(uint32 u32Sz, uint32 *pu32Idx)
{
	if (pu32Indexes != NULL) delete[] pu32Indexes;
	u32Size = u32Sz;
	pu32Indexes = new uint32[u32Sz];
	VECT_vCopy(pu32Idx, pu32Indexes, u32Sz);
}

void CIndexArray::vSaveBinary(FILE *pf)
{
	fwrite(&u32Size, sizeof(uint32), 1, pf);
	fwrite(pu32Indexes, sizeof(uint32), u32Size, pf);
}

void CIndexArray::vReadBinary(FILE *pf)
{
	fread(&u32Size, sizeof(uint32), 1, pf);
	pu32Indexes = new uint32[u32Size];
	fread(pu32Indexes, sizeof(uint32), u32Size, pf);
}

void CIndexArray::vSkipBinary(FILE *pf)
{
	uint32 u32Sz;
	fread(&u32Sz, sizeof(uint32), 1, pf);
	fseek(pf, sizeof(uint32) * u32Sz, SEEK_CUR);
}

