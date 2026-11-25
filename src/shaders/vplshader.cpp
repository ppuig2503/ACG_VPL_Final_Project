#include "vplshader.h"
#include "../core/hemisphericalsampler.h"
#include "../core/utils.h"

#include <algorithm>
#include <numbers>
#include <random>

namespace
{
    constexpr double pi = std::numbers::pi;

    // Both camera and light paths use the refraction ratio convention of NeeShader.
    Ray specularBounce(const Ray& ray, const Intersection& hit)
    {
        const Material& material = hit.shape->getMaterial();
        const Vector3D wo = -ray.d;
        Vector3D n = hit.normal;
        if (material.hasTransmission()) {
            double ratio = material.getIndexOfRefraction();
            if (dot(wo, n) < 0.0) {
                ratio = 1.0 / ratio;
                n = -n;
            }
            const double cosine = dot(n, wo);
            const double radicand = 1.0 - ratio * ratio * (1.0 - cosine * cosine);
            if (radicand >= 0.0) {
                Ray refracted(hit.itsPoint, -ratio * wo + n * (ratio * cosine - std::sqrt(radicand)), ray.depth + 1);
                refracted.isRefractedRay = true;
                return refracted;
            }
        }
        Ray reflected(hit.itsPoint, 2.0 * dot(n, wo) * n - wo, ray.depth + 1);
        reflected.isReflectedRay = true;
        return reflected;
    }

}

VplShader::VplShader(Vector3D bgColor, double maxGeometry_)
    : Shader(bgColor), maxGeometry(std::max(0.0, maxGeometry_))
{}

void VplShader::generateVPLs(const std::vector<Shape*>& objList, const std::vector<LightSource*>& lsList,
                             int numPaths, int maxDepth)
{
    vplList.clear();
    if (lsList.empty() || numPaths <= 0 || maxDepth <= 0)
        return;

    HemisphericalSampler sampler;
    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<size_t> chooseLight(0, lsList.size() - 1);
    std::bernoulli_distribution chooseHemisphere;
    const double pathWeight = static_cast<double>(lsList.size()) / numPaths;

    for (int path = 0; path < numPaths; ++path) {
        const LightSource& light = *lsList[chooseLight(generator)];
        const Vector3D origin = light.sampleLightPosition();
        const double area = light.getArea();
        Vector3D direction;
        Vector3D flux;
        if (dynamic_cast<const PointLightSource*>(&light)) {
            direction = sampler.getSample(Vector3D(0, chooseHemisphere(generator) ? 1 : -1, 0));
            flux = light.getIntensity() * (pathWeight * 4.0 * pi);
        } else if (area > 0.0) {
            const Vector3D n = light.getNormal().normalized();
            direction = sampler.getSample(n);
            flux = light.getIntensity() * (pathWeight * area * 2.0 * pi * std::max(0.0, dot(n, direction)));
        } else {
            continue;
        }

        Ray ray(origin, direction);
        for (int bounce = 0; bounce < maxDepth; ++bounce) {
            Intersection hit;
            if (!Utils::getClosestIntersection(ray, objList, hit))
                break;
            const Material& material = hit.shape->getMaterial();
            if (material.hasSpecular() || material.hasTransmission()) {
                ray = specularBounce(ray, hit);
                continue;
            }
            if (!material.hasDiffuseOrGlossy() || dot(hit.normal, -ray.d) <= 0.0)
                break;

            vplList.push_back({hit.itsPoint, hit.normal, flux * material.getDiffuseReflectance() / pi});
            if (bounce + 1 == maxDepth)
                break;
            const Vector3D next = sampler.getSample(hit.normal);
            flux = flux * material.getReflectance(hit.normal, -ray.d, next)
                 * (2.0 * pi * std::max(0.0, dot(hit.normal, next)));
            ray = Ray(hit.itsPoint, next, ray.depth + 1);
        }
    }
}

Vector3D VplShader::computeColor(const Ray& r, const std::vector<Shape*>& objList, const std::vector<LightSource*>& lsList) const
{
    int maxDepth = 4;
	if (r.depth >= maxDepth) { // too avoid infinite recursion, we stop the recursion at a certain depth
        return bgColor;
    }
    Intersection its;
    Vector3D wo = -r.d;

    Vector3D color(0.0, 0.0, 0.0);
    if (Utils::getClosestIntersection(r, objList, its)) {

        if (its.shape->getMaterial().hasSpecular()) {
            //MIRROR
            Vector3D wr = 2 * dot(its.normal, wo) * its.normal - wo;
            Ray reflectionRay = Ray(its.itsPoint, wr, r.depth + 1);
			reflectionRay.isReflectedRay = true; // Mark the ray as a reflected ray
            color = computeColor(reflectionRay, objList, lsList);
        }

        else if (its.shape->getMaterial().hasTransmission()) {
            Vector3D normal = its.normal;
            double mu_t = its.shape->getMaterial().getIndexOfRefraction();
            if (dot(wo, normal) < 0) {
                mu_t = 1 / mu_t;
                normal = -normal;
            }

            double radicand = 1 - mu_t * mu_t * (1 - dot(normal, wo) * dot(normal, wo));
            if (radicand >= 0) { // NO total internal reflection, compute the refracted ray
                Vector3D wt = -mu_t * wo + normal * (mu_t * dot(normal, wo) - sqrt(radicand));
                Ray refracRay = Ray(its.itsPoint, wt, r.depth + 1);
				refracRay.isRefractedRay = true; // Mark the ray as a refracted ray
                color = computeColor(refracRay, objList, lsList);
            }
            else { // Total internal reflection, color = Specular reflection, acts like a mirror
                Vector3D wr = 2 * dot(normal, wo) * normal - wo;
                Ray reflectionRay = Ray(its.itsPoint, wr, r.depth + 1);
				reflectionRay.isReflectedRay = true; // Mark the ray as a reflected ray
                color = computeColor(reflectionRay, objList, lsList);
            }
        }

        else if ((its.shape->getMaterial().isEmissive() and r.depth == 0)) {
            color = its.shape->getMaterial().getEmissiveRadiance();
        }

        else if (its.shape->getMaterial().hasDiffuseOrGlossy()) {
			if (r.isReflectedRay || r.isRefractedRay) {
				color = its.shape->getMaterial().getEmissiveRadiance(); // If the ray is reflected or refracted, return the emissive radiance of the material
			}
            // DIRECT RADIANCE (one NEE sample per light)
            for (const LightSource* light : lsList) {
                Vector3D y = light->sampleLightPosition();
                Vector3D wi = y - its.itsPoint;
                double distance = wi.length();
                if (distance <= 2 * Epsilon)
                    continue;
                Vector3D wi_normalized = wi / distance;

                double G = fmax(0, dot(its.normal, wi_normalized)) / (distance * distance);
                double pdf = 1.0; // A point source has a fixed position.
                if (!dynamic_cast<const PointLightSource*>(light)) {
                    double area = light->getArea();
                    if (area <= 0.0)
                        continue;
                    pdf = 1.0 / area;
                    G *= fmax(0, dot(light->getNormal().normalized(), -wi_normalized));
                }

                Ray shadowRay(its.itsPoint, wi_normalized, 0, Epsilon, distance - Epsilon);
                if (G > 0.0 && !Utils::hasIntersection(shadowRay, objList)) {
                    Vector3D brdf = its.shape->getMaterial().getReflectance(its.normal, wo, wi_normalized);
                    color += light->getIntensity() * brdf * G / pdf;
                }
            }

            // INDIRECT RADIANCE (connect this hit to each precomputed VPL)
            for (const VPL& vpl : vplList) {
                Vector3D wi = vpl.pos - its.itsPoint;
                double distance = wi.length();
                if (distance <= 2 * Epsilon)
                    continue;
                Vector3D wi_normalized = wi / distance;
                double G = fmax(0, dot(its.normal, wi_normalized))
                         * fmax(0, dot(vpl.normal, -wi_normalized)) / (distance * distance);
                G = std::min(G, maxGeometry);

                Ray shadowRay(its.itsPoint, wi_normalized, 0, Epsilon, distance - Epsilon);
                if (G > 0.0 && !Utils::hasIntersection(shadowRay, objList)) {
                    Vector3D brdf = its.shape->getMaterial().getReflectance(its.normal, wo, wi_normalized);
                    color += vpl.flux * brdf * G;
                }
            }
        }
    }
    return color;
}
