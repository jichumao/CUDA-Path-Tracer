#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;


#if BVH_ENABLED
class LBVHNode {
public:
	AABB boundingBox;
	int secondChildOffset;
	bool isLeaf;
	LBVHNode() : boundingBox(), secondChildOffset(-1), isLeaf(false) {}
};

class BVHNode {
	friend class Scene;
public:
	AABB boundingBox;
	BVHNode* left, * right;
	BVHNode();
	BVHNode(AABB aabb);
	void collapseIntoSingleAABB(std::vector<AABB>& boundingBoxes);
};

void buildBVH(BVHNode*& node, std::vector<AABB>& boundingBoxes);
void nofOfNodesInBVH(BVHNode* node, int& count);
int flattenBVH(std::vector<LBVHNode>& flattenedBVH, BVHNode* node, int& offset);
#endif

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
	void loadFromGltf(const std::string& gltfName, Geom& gltfMesh);
public:
    Scene(string filename);
    ~Scene();

	std::vector<Geom> geoms;
	std::vector<Material> materials;
	std::vector<Triangle> meshTris;
    std::vector<Texture> textures;
    std::vector<glm::vec3> texturesData;
    RenderState state;

    bool enable_skybox;
    Texture* skyboxTexture;

#if BVH_ENABLED
	bool bvhBuilt = false;
	BVHNode* root;
	std::vector<LBVHNode> flattenedBVH;
	std::vector<AABB> boundingBoxes;
	void computeAABB(Geom geom);
#endif
};
