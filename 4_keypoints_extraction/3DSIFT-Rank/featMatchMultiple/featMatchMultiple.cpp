/////////////////////////////////////////////////////
//                                                 //
// File: featMatchMultiple.cpp                     //
// Author: Matthew Toews                           //
// Contributor: Laurent Chauvin                    //
//                                                 //
/////////////////////////////////////////////////////

#include <iostream>
#include <thread>

#include "nifti1_io.h"
#include "FeatureIOnifti.h"
#include "featMatchUtilities.h"
#include "TextFile.h"

#define MAX_CORES 32

//
// matchAllToAll()
//
// Match all images to all images.
//
int
matchAllToAll(
	vector< char * > &vecNames,
	vector< int > &vecMatched,
	vector< vector<Feature3DInfo> > &vvInputFeats,
	int iNeighbors,	// number of nearest neighbors to consider
	vector< int > &viLabels, // labels for each feature set
	float fGeometryWeight = -1,
	char *pcOutputFile = 0,
	int iImgSplit = -1  // image to split on for testing
)
{
	printf("Creating NN index structure, NN=%d, image split=%d, ", iNeighbors, iImgSplit);

	msNearestNeighborApproximateInit(vvInputFeats, iNeighbors, viLabels, fGeometryWeight, iImgSplit);

	printf("done.\n");

	vector< int > viImgCounts;
	vector< float > viLabelCounts;
	vector< float > vfLabelLogLikelihood;

	// Count labels
	viImgCounts.resize(vecNames.size(), 0);

	// Assume labels go from 
	int iMaxLabel = 0;
	for (int i = 0; i < viLabels.size(); i++)
	{
		if (viLabels[i] > iMaxLabel)
		{
			iMaxLabel = viLabels[i];
		}
	}
	viLabelCounts.resize(iMaxLabel + 1);
	vfLabelLogLikelihood.resize(iMaxLabel + 1);

	// Store votes
	FILE *outfileVotes = fopen("matching_votes.txt", "wt");
	FILE* outfileVoteCount = fopen("vote_count.txt", "wt");
	float **ppfMatchingVotes = new float*[vvInputFeats.size()];
	int **ppiLabelVotes = new int*[vvInputFeats.size()];
	for (int i = 0; i < vvInputFeats.size(); i++)
	{
		ppfMatchingVotes[i] = new float[vvInputFeats.size()];
		ppiLabelVotes[i] = new int[vvInputFeats.size()];
		for (int j = 0; j < vvInputFeats.size(); j++)
		{
			ppfMatchingVotes[i][j] = 0.0;
			ppiLabelVotes[i][j] = 0;
		}
	}	
	
	// Try max core detection with std::thread::hardware_concurrency() (C++11)
	int maxCores = std::thread::hardware_concurrency();
	if (maxCores <= 0 || maxCores > MAX_CORES) {
		maxCores = MAX_CORES;
	}
	int nCores = vvInputFeats.size() < maxCores ? vvInputFeats.size() : maxCores;

	// Create image chunks
	int iChunkSize = std::ceil(float(vvInputFeats.size()) / nCores);
	
	std::vector< std::pair<int,int> > vpChunkStartEnd;
	for (int i = 0; i < nCores; ++i)
	  {
	    int iStart = i*iChunkSize;
	    int iEnd = 0;

	    if (iStart > vvInputFeats.size())
	      {
		// Not enough data for all cores
		continue;
	      }

	    iEnd = (i+1)*iChunkSize-1;

	    if (iEnd > vvInputFeats.size()-1)
	      {
		// Not enough data for last core. Clip.
		iEnd = vvInputFeats.size()-1;
	      }
	    std::pair<int,int> pStartEnd(iStart, iEnd);
	    vpChunkStartEnd.push_back(pStartEnd);
	  }

#pragma omp parallel for num_threads(nCores) schedule(static,1)
	for (int n = 0; n < vpChunkStartEnd.size(); ++n)
	    {
	      for (int i = vpChunkStartEnd[n].first; i <= vpChunkStartEnd[n].second; i++)
			{
			  #pragma omp critical
			  {
			  std::cout << "Searching image " << i << " of " << vvInputFeats.size() << std::endl;
			  }
			  //printf("Searching image %d of %d ... ", i, vvInputFeats.size());
			  msNearestNeighborApproximateSearchSelf(i, viImgCounts, viLabelCounts, vfLabelLogLikelihood, ppfMatchingVotes, ppiLabelVotes);
			}
	    }

	// Output votes
	for (int i = 0; i < vvInputFeats.size(); i++)
	{
		for (int j = 0; j < vvInputFeats.size(); j++)
		{
			float vote = (i == j) ? 0.0 : ppfMatchingVotes[i][j] / ppiLabelVotes[i][j];
			fprintf(outfileVotes, "%f\t", ppfMatchingVotes[i][j]);
			fprintf(outfileVoteCount, "%d\t", ppiLabelVotes[i][j]);
		}
		fprintf(outfileVotes, "\n");
		fprintf(outfileVoteCount, "\n");
		delete[] ppfMatchingVotes[i];
	}
	delete[] ppfMatchingVotes;
	fclose(outfileVotes);
	fclose(outfileVoteCount);

	msNearestNeighborApproximateDelete();

	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Volumetric Feature matching v1.1\n");
		printf("Determines robust alignment solution mapping coordinates in image 2, 3, ... to image 1.\n");
		printf("Usage: %s [options] <input keys 1> <input keys 2> ... \n", "featMatchMultiple");
		printf("  <input keys 1, ...>: input key files, produced from featExtract.\n");
		printf("  <output transform>: output text file with linear transform from keys 2 -> keys 1.\n");
		return -1;
	}


	// Output command in file for log purpose
	FILE *commandFile = fopen("_command.txt", "wt");
	for (int i = 0; i < argc; ++i)
	  {
	    fprintf(commandFile, "%s ", argv[i]);
	  }
	fprintf(commandFile,"\n");
	fclose(commandFile);
	
	int iArg = 1;
	int bMultiModal = 0;
	int bExpandedMatching = 0;
	int bOnlyReorientedFeatures = 1;
	char *pcOutputFileName = "report.txt";
	char *pcInputFileList = 0;
	char *pcLabelFile = 0;
	int iMatchType = 2;
	int bViewFeatures = 0;
	int iNeighbors = 5; 
	float fGeometryWeight = -1;
	float iImgSplit = -1;
	while (iArg < argc && argv[iArg][0] == '-')
	{
		switch (argv[iArg][1])
		{
		case 'o': case 'O':
			// General output file name
			iArg++;
			pcOutputFileName = argv[iArg];
			iArg++;
			break;

		case 'r': case 'R':
			// Option to not use reoriented features - for volumes that are already aligned
			bOnlyReorientedFeatures = 1;
			if (argv[iArg][2] == '-')
			{
				// Only 
				bOnlyReorientedFeatures = 0;
			}
			iArg++;
			break;

		case 'n': case 'N':

			//
			// Number of nearest neighbors to consider
			//
			iArg++;
			iNeighbors = atoi(argv[iArg]);
			iArg++;
			break;

		case 'f': case 'F':
			//
			// Input features are listed in a single file (for long file lists)
			//
			iArg++;
			pcInputFileList = argv[iArg];
			iArg++;
			break;

		default:
			printf("Error: unknown command line argument: %s\n", argv[iArg]);
			return -1;
			break;
		}
	}

	FILE *outfileR = fopen(pcOutputFileName, "wt");
	fclose(outfileR);

	vector< char * > vecNames;
	vector< int > vLabels;
	int iTotalFeats = 0;
	int iFeatVec = 0;
	int iFeatVecTotal = 0;

	// Create file list for read in - allows input file, for long lists of files
	// for example, longer than the linux limit: >> getconf ARG_MAX
	TextFile tfNames;
	if (pcInputFileList)
	{
		if (tfNames.Read(pcInputFileList) != 0)
		{
			printf("Error: could not read input file name list: %s\n");
			return -1;
		}
		iFeatVec = 0;
		for (int i = 0; i < tfNames.Lines(); i++)
		{
			if (strlen(tfNames.GetLine(i)) > 0)
			{
				vecNames.push_back(tfNames.GetLine(i));
				iFeatVec++;
			}
		}
		iFeatVecTotal = iFeatVec;
	}
	else
	{
		iFeatVec = 0;
		iFeatVecTotal = argc - iArg;
		for (int i = iArg; i < argc; i++)
		{
			vecNames.push_back(argv[i]);
			iFeatVec++;
		}
		assert(iFeatVec == iFeatVecTotal);
	}

	// read in labels
	TextFile tfLabels;
	if (pcLabelFile)
	{
		if (tfLabels.Read(pcLabelFile) != 0)
		{
			printf("Error: could not read input file name list: %s\n");
			return -1;
		}
		iFeatVec = 0;
		for (int i = 0; i < tfLabels.Lines(); i++)
		{
			if (strlen(tfLabels.GetLine(i)) > 0)
			{
				int iLabel = atoi(tfLabels.GetLine(i));
				vLabels.push_back(iLabel);
				iFeatVec++;
			}
		}
		assert(vLabels.size() == vecNames.size());
	}
	else
	{
		// Default labels = simply image indices
		vLabels.resize(vecNames.size());
		for (int i = 0; i < vLabels.size(); i++)
		{
			vLabels[i] = i;
		}
	}

	FILE *outfileNames = fopen("_names.txt", "wt");
	for (iFeatVec = 0; iFeatVec < iFeatVecTotal; iFeatVec++)
	{
		fprintf(outfileNames, "%s\t%d\n", vecNames[iFeatVec], vLabels[iFeatVec]);
	}
	fclose(outfileNames);

	assert(vecNames.size() == iFeatVecTotal);

	vector< int > vecMatched;
	vector< vector<Feature3DInfo> > vvInputFeats;
	vecMatched.resize(vecNames.size(), -1);
	vvInputFeats.resize(vecNames.size());

	for (iFeatVec = 0; iFeatVec < iFeatVecTotal; iFeatVec++)
	{
		char pcImg1[400];
		sprintf(pcImg1, "%s", vecNames[iFeatVec]);
		char *pch = strrchr(pcImg1, '\\');
		if (pch) pch++; else pch = pcImg1;

		int bSameName = 0;
		for (int j = 0; j < iFeatVec && bSameName == 0; j++)
		{
			if (strcmp(vecNames[iFeatVec], vecNames[j]) == 0)
			{
				bSameName = 1;
			}
		}

		printf("Reading file %d: %s...", iFeatVec, pch);

		if (msFeature3DVectorInputText(vvInputFeats[iFeatVec], vecNames[iFeatVec], 140) < 0)
		{
			printf("Error: could not open feature file %d: %s\n", iFeatVec, vecNames[iFeatVec]);
			continue;
		}

		if (bOnlyReorientedFeatures)
		{
			removeNonReorientedFeatures(vvInputFeats[iFeatVec]);
		}
		else
		{
			removeReorientedFeatures(vvInputFeats[iFeatVec]);
		}

		iTotalFeats += vvInputFeats[iFeatVec].size();
		printf("feats: %d, total: %d\n", vvInputFeats[iFeatVec].size(), iTotalFeats);
	}

	vecNames.resize(iFeatVec);
	vvInputFeats.resize(iFeatVec);
	vecMatched.resize(iFeatVec, -1);

	FILE *outfile = fopen("feature_count.txt", "wt");
	for (int i = 0; i < vvInputFeats.size(); i++)
	{
		fprintf(outfile, "%d\t%d\n", i, vvInputFeats[i].size());
	}
	fclose(outfile);


	matchAllToAll(vecNames, vecMatched, vvInputFeats, iNeighbors, vLabels, fGeometryWeight, pcOutputFileName, iImgSplit);

	return 0;
}

