#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"

#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"
#include "stb_image.h"
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;

#if ENVIRONMENT_MAP_ENABLED
	int width, height, channels;
	int desired_channels = 4;
    //std::string fileName = "meadow_2_4k.hdr";
	//std::string fileName = "meadow_4k.hdr";
	//std::string fileName = "satara_night_4k.hdr";
	//std::string fileName = "shanghai_bund_4k.hdr";
	std::string fileName = "small_apartment_1.hdr";
	//std::string fileName = "interior_atelier_soft_daylight.hdr";
	std::string dir_skyboxTex = "../resources/environment_maps/" + fileName;
	float* h_image = stbi_loadf(dir_skyboxTex.c_str(), &width, &height, &channels, desired_channels);
	if (!h_image) {
		std::cerr << "Failed to load SKYBOX image!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// Assign to skyboxTexture
	skyboxTexture = new Texture();
	skyboxTexture->width = width;
	skyboxTexture->height = height;
	skyboxTexture->numChannels = desired_channels;
	skyboxTexture->type = SkyboxMap;
	skyboxTexture->data = h_image;
	enable_skybox = true;
#endif
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            const float& roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f - roughness;

			const auto& col2 = p["SPECRGB"];
			newMaterial.specular.color = glm::vec3(col2[0], col2[1], col2[2]);
            
		}
		else if (p["TYPE"] == "Refractive")
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.indexOfRefraction = p["IOR"];
			newMaterial.hasRefractive = 1.0f;

            const auto& col2 = p["SPECRGB"];
            newMaterial.specular.color = glm::vec3(col2[0], col2[1], col2[2]);
		}
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];

	uint32_t gid = 0;
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;

        newGeom.geometryid = gid++;
		newGeom.materialid = MatNameToID[p["MATERIAL"]];
		const auto& trans = p["TRANS"];
		const auto& rotat = p["ROTAT"];
		const auto& scale = p["SCALE"];
		newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
		newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
		newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
		newGeom.transform = utilityCore::buildTransformationMatrix(
			newGeom.translation, newGeom.rotation, newGeom.scale);
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
		else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh_gltf")
        {
			newGeom.type = MESH;
			loadFromGltf(p["FILE"], newGeom);
		}
		else {
			std::cerr << "Unknown object type: " << type << std::endl;
			continue;
		}

#if BVH_ENABLED
		computeAABB(newGeom);
#endif
        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];

#if DEPTH_OF_FIELD
	camera.lensRadius = cameraData["LENSRADIUS"];
	camera.focalLength = cameraData["FOCALLENGTH"];
#endif

    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
	return glm::vec3(m * v);
}

// Reference https://www.slideshare.net/slideshow/gltf-20-reference-guide/78149291#1
void Scene::loadFromGltf(const std::string& gltfName, Geom& meshGeom) {

	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	std::string dir_gltf = "../resources/" + gltfName + "/glTF/" + gltfName + ".gltf";
	bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, dir_gltf);

	if (!warn.empty()) {
		std::cout << "Warn: " << warn << std::endl;
	}

	if (!err.empty()) {
		std::cerr << "Error: " << err << std::endl;
	}

    if (!ret) {
		std::cerr << "Failed to load glTF: " << dir_gltf << std::endl;
		return;
    }

#if BOUNDING_VOLUME_INTERSECTION_CULLING_ENABLED 1
	meshGeom.min = glm::vec3(std::numeric_limits<float>::max());
	meshGeom.max = glm::vec3(std::numeric_limits<float>::lowest());
#endif
    meshGeom.startTriangleIndex = meshTris.size();
	// For each mesh in the glTF file
    for (const auto& mesh : model.meshes) {
        // For each primitive in the mesh
        for (const auto& primitive : mesh.primitives) {

            const float* positions = nullptr;
            const float* normals = nullptr;
            const float* texcoords = nullptr;

            size_t vertexCount = 0;

            for (const auto& attr : primitive.attributes) {
                const tinygltf::Accessor& accessor = model.accessors[attr.second];
                const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
                const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

                if (attr.first == "POSITION") {
                    positions = reinterpret_cast<const float*>(dataPtr);
                    vertexCount = accessor.count;
                }
                else if (attr.first == "NORMAL") {
                    normals = reinterpret_cast<const float*>(dataPtr);
                    meshGeom.hasNormals = true;
                }
                else if (attr.first == "TEXCOORD_0") {
                    texcoords = reinterpret_cast<const float*>(dataPtr);
                    meshGeom.hasUVs = true;
                }
            }
#if BOUNDING_VOLUME_INTERSECTION_CULLING_ENABLED
			// Calculate the bounding box
			for (size_t i = 0; i < vertexCount; ++i) {
				glm::vec3 pos(
					positions[i * 3],
					positions[i * 3 + 1],
					positions[i * 3 + 2]
				);
				meshGeom.min = glm::min(multiplyMV(meshGeom.inverseTransform, glm::vec4(pos, 1.0f)),
					meshGeom.min);
				meshGeom.max = glm::max(multiplyMV(meshGeom.inverseTransform, glm::vec4(pos, 1.0f)),
				meshGeom.max);
			}
#endif
			// Get the indices from the primitive
			std::vector<unsigned int> indices;
			if (primitive.indices >= 0) {
				const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
				const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
				const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
				const unsigned char* indexData = indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset;

				indices.resize(indexAccessor.count);
                // 5123
				if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
					const uint16_t* buf = reinterpret_cast<const uint16_t*>(indexData);
					for (size_t i = 0; i < indexAccessor.count; ++i) {
						indices[i] = static_cast<unsigned int>(buf[i]);
					}
				}
                // 5125
				else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
					const uint32_t* buf = reinterpret_cast<const uint32_t*>(indexData);
					for (size_t i = 0; i < indexAccessor.count; ++i) {
						indices[i] = static_cast<unsigned int>(buf[i]);
					}
				}
				else {
					std::cerr << "Unsupported Indices component" << std::endl;
					continue;
				}
			}
			else {
				// if no indices, generate them
				indices.resize(vertexCount);
				for (unsigned int i = 0; i < vertexCount; ++i) {
					indices[i] = i;
				}
			}
			// Add the texture to the material
            if (primitive.material >= 0) {
                int idx = model.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index;
				if (idx >= 0) {
					const tinygltf::Texture& texture = model.textures[idx];
					const tinygltf::Image& image = model.images[texture.source];

					Texture tex;
					tex.id = textures.size();
					tex.width = image.width;
					tex.height = image.height;
					tex.numChannels = image.component;
					tex.type = AlbedoMap;
                    tex.startIdx = texturesData.size();

					// The start index of the texture data in texturesData
					meshGeom.albedoTextureId = tex.id;
					meshGeom.hasAlbedo = true;
                    
					// Add color from image to texturesData
					for (size_t i = 0; i < image.image.size(); i += image.component) {
						glm::vec3 color;
						if (image.component == 1) {
							color = glm::vec3(image.image[i]/255.f);
						}
						else if (image.component == 4) {
							color = glm::vec3(image.image[i]/255.f, image.image[i + 1] / 255.f, image.image[i + 2] / 255.f);
						}
						else {
							std::cerr << "Unsupported number of channels in texture" << std::endl;
							continue;
						}
						texturesData.push_back(color);
					}
                    tex.endIdx = texturesData.size() - 1;
                    textures.push_back(tex);
				}
            }
			// Create triangles from the indices
			for (size_t i = 0; i + 2 < indices.size(); i += 3) {
				Triangle tri;
				auto setVertex = [&](Vertex& vert, unsigned int idx) {
					vert.position = glm::vec3(
						positions[idx * 3],
						positions[idx * 3 + 1],
						positions[idx * 3 + 2]
					);
					vert.normal = normals ? glm::vec3(
						normals[idx * 3],
						normals[idx * 3 + 1],
						normals[idx * 3 + 2]
					) : glm::vec3(0.0f);
					vert.uv = texcoords ? glm::vec2(
						texcoords[idx * 2],
						texcoords[idx * 2 + 1]
					) : glm::vec2(0.0f);
					};

				setVertex(tri.v0, indices[i]);
				setVertex(tri.v1, indices[i + 1]);
				setVertex(tri.v2, indices[i + 2]);

				meshTris.push_back(tri);
			}
        }

        meshGeom.endTriangleIndex = meshTris.size() - 1;
    }
}
#if BVH_ENABLED
void Scene::computeAABB(Geom geom) {
	switch (geom.type) {
	case(SPHERE): {
		glm::vec3 minPos = glm::vec3(geom.transform * glm::vec4(-0.5f, -0.5f, -0.5f, 1.f));
		glm::vec3 maxPos = glm::vec3(geom.transform * glm::vec4(0.5f, 0.5f, 0.5f, 1.f));
		boundingBoxes.push_back(AABB(minPos, maxPos, (minPos + maxPos) / 2.f, geom));
		break;
	}
	case(CUBE): {
		glm::vec3 minPos = glm::vec3(std::numeric_limits<float>::max()), maxPos = -minPos;
		glm::vec3 center = glm::vec3(geom.transform * glm::vec4(0.f, 0.f, 0.f, 1.f));
		for (float i = -0.5; i <= 0.5; i += 1.f) {
			for (float j = -0.5; j <= 0.5; j += 1.f) {
				for (float k = -0.5; k <= 0.5; k += 1.f) {
					glm::vec3 pos = glm::vec3(geom.transform * glm::vec4(i, j, k, 1.f));
					minPos = glm::min(minPos, pos);
					maxPos = glm::max(maxPos, pos);
				}
			}
		}
		boundingBoxes.push_back(AABB(minPos, maxPos, center, geom));
		break;
	}
	case(MESH): {
		for (int i = geom.startTriangleIndex; i <= geom.endTriangleIndex; i++) {
			glm::vec3 minPos = glm::vec3(std::numeric_limits<float>::max()), maxPos = -minPos;
			glm::vec3 p1 = glm::vec3(geom.transform * glm::vec4(meshTris[i].v0.position, 1.f));
			glm::vec3 p2 = glm::vec3(geom.transform * glm::vec4(meshTris[i].v1.position, 1.f));
			glm::vec3 p3 = glm::vec3(geom.transform * glm::vec4(meshTris[i].v2.position, 1.f));
			minPos = glm::min(p1, glm::min(p2, p3));
			maxPos = glm::max(p1, glm::max(p2, p3));
			glm::vec3 centroid = (p1 + p2 + p3) / 3.f;
			boundingBoxes.push_back(AABB(minPos, maxPos, centroid, geom, i));
		}
		break;
	}
	default: cout << "Couldn't compute bounding box!" << endl;
	}
}

BVHNode::BVHNode() :boundingBox(), left(nullptr), right(nullptr) {}
BVHNode::BVHNode(AABB aabb) : boundingBox(aabb), left(nullptr), right(nullptr) {}

void BVHNode::collapseIntoSingleAABB(std::vector<AABB>& boundingBoxes) {
	glm::vec3 minPos = glm::vec3(std::numeric_limits<float>::max()), maxPos = -minPos;
	for (const AABB& aabb : boundingBoxes) {
		minPos = glm::min(minPos, aabb.minPos);
		maxPos = glm::max(maxPos, aabb.maxPos);
	}
	this->boundingBox = AABB(minPos, maxPos);
}

bool xSort(AABB a, AABB b) { return a.centroid.x < b.centroid.x; }
bool ySort(AABB a, AABB b) { return a.centroid.y < b.centroid.y; }
bool zSort(AABB a, AABB b) { return a.centroid.z < b.centroid.z; }

void splitSAH(std::vector<AABB>& boundingBoxes) {
	std::sort(boundingBoxes.begin(), boundingBoxes.end(), xSort);
	float xLen = boundingBoxes.back().centroid.x - boundingBoxes.front().centroid.x;
	std::sort(boundingBoxes.begin(), boundingBoxes.end(), ySort);
	float yLen = boundingBoxes.back().centroid.y - boundingBoxes.front().centroid.y;
	std::sort(boundingBoxes.begin(), boundingBoxes.end(), zSort);
	float zLen = boundingBoxes.back().centroid.z - boundingBoxes.front().centroid.z;
	if (xLen > yLen && xLen > zLen) {
		std::sort(boundingBoxes.begin(), boundingBoxes.end(), xSort);
	}
	else if (yLen >= xLen && yLen >= zLen) {
		std::sort(boundingBoxes.begin(), boundingBoxes.end(), ySort);
	}
}

void buildBVH(BVHNode*& node, std::vector<AABB>& boundingBoxes) {
	node = new BVHNode();
	node->collapseIntoSingleAABB(boundingBoxes);
	if (boundingBoxes.size() <= 2) {
		node->left = new BVHNode(boundingBoxes[0]);
		if (boundingBoxes.size() == 2) {
			node->right = new BVHNode(boundingBoxes[1]);
		}
		return;
	}
	else {
		splitSAH(boundingBoxes);
		node->collapseIntoSingleAABB(boundingBoxes);
		int splitIdx = floor(boundingBoxes.size() / 2.f);
		buildBVH(node->left, std::vector<AABB>(boundingBoxes.begin(), boundingBoxes.begin() + splitIdx));
		buildBVH(node->right, std::vector<AABB>(boundingBoxes.begin() + splitIdx, boundingBoxes.end()));
	}
}

void nofOfNodesInBVH(BVHNode* node, int& count) {
	count++;
	if (node->left != __nullptr) {
		nofOfNodesInBVH(node->left, count);
	}
	if (node->right != __nullptr) {
		nofOfNodesInBVH(node->right, count);
	}
}

int flattenBVH(std::vector<LBVHNode>& flattenedBVH, BVHNode* node, int& offset) {
	int currentOffset = offset++;
	flattenedBVH[currentOffset].boundingBox = node->boundingBox;
	if (node->left) {
		flattenBVH(flattenedBVH, node->left, offset);
		if (node->right) {
			flattenedBVH[currentOffset].secondChildOffset = flattenBVH(flattenedBVH, node->right, offset);
		}
	}
	else {
		flattenedBVH[currentOffset].isLeaf = true;
	}
	return currentOffset;
}
#endif