#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
void scatter_diffuse(PathSegment& pathSegment,
	glm::vec3 normal,
	const Material& m,
	thrust::default_random_engine& rng) {
	pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
	pathSegment.accumColor *= m.color; 
}

__host__ __device__
void scatter_specular(PathSegment& pathSegment,
	glm::vec3 normal,
	const Material& m) {
	pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
	pathSegment.accumColor *= m.specular.color;
}

// Implementation referenced from CIS 5610
__host__ __device__
void scatter_transmissive(PathSegment& pathSegment,
	glm::vec3 normal,
	const Material& m) {

	float eta;

	if (glm::dot(-pathSegment.ray.direction, normal) >= 0) {
		eta = 1.f / m.indexOfRefraction;
	}
	else {
		eta = m.indexOfRefraction;
		normal *= -1.f;
	}

	pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, eta);

	if (glm::length(pathSegment.ray.direction) == 0) { 
        scatter_specular(pathSegment, normal, m);
	}
	else {
		pathSegment.accumColor *= m.specular.color;
	}
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    // Implementation referenced form CIS 5610
	
	// if material is reflective
    if (m.hasReflective > 0.0f) {
        // if material is not 100% reflective
		if (m.hasReflective < 1.0f) {
			thrust::uniform_real_distribution<float> u01(0, 1);
			if (u01(rng) < 0.5) {
				scatter_diffuse(pathSegment, normal, m, rng);
			}
			else {
				scatter_specular(pathSegment, normal, m);
			}
			pathSegment.accumColor *= 2.0;
           // float diffuse = 1 - m.hasReflective;
		}
		// 100% reflective
		else {
			scatter_specular(pathSegment, normal, m);
		}
    }
    // if material is refractive(glass)
	else if (m.hasRefractive > 0.0f) {
        scatter_transmissive(pathSegment, normal, m);
	}
	// if material is 100% diffuse
	else {
		scatter_diffuse(pathSegment, normal, m, rng);
    }

    pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction;
}
