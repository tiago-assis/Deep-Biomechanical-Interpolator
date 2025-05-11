// contact_test.cpp : Defines the entry point for the console application.
//

#include "contacts.h"

int main(int argc, char* argv[])
{
	unsigned long ulNumTriangles = 2;                      // number of triangles in the master mesh
	unsigned long pulNodeIndexes[] = {0, 1, 2, 3, 0, 1};   // index of nodes defining the triangles - 3 nodes for each triangle, coordinates of these nodes are in the next array
	double pdNodeCoordinates[] = {0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1};  // coordinates of nodes defining the triangles - 3 coordinates for each node
	unsigned long ulNumContactNodes = 1;					// number of slave nodes
	unsigned long pulContactNodesIndexes[] = {0};			// indexes of the slave nodes
	double dMaxPossiblePenetration = 1;						// maximum displacement of a slave node for any time step

	double pdPenetratingNodePoz[3];							// position of all nodes, 3 values per node - slave nodes are identified using indexes from pulContactNodesIndexes
	pdPenetratingNodePoz[0] = 0.1;
	pdPenetratingNodePoz[1] = 0.1;
	pdPenetratingNodePoz[2] = 0.1;

	// this must be called to initialize the contacts
	CONTACTS_vInit(ulNumTriangles, pulNodeIndexes, pdNodeCoordinates, ulNumContactNodes, pulContactNodesIndexes, dMaxPossiblePenetration);
	// this is called after each displacement computation - the penetrating nodes will be moved back on the surface
	CONTACTS_vApplyContacts(pdPenetratingNodePoz);  
	pdPenetratingNodePoz[2] = 0.05;
	CONTACTS_vApplyContacts(pdPenetratingNodePoz);

	// call this at the end to free allocated memory
	CONTACTS_vFreeMem();
	return 0;
}

