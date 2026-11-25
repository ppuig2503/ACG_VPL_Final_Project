#include "vplshader.h"
#include "../core/hemisphericalsampler.h"
#include "../core/utils.h"

#include <algorithm>
#include <limits>
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

    // A zero emitter normal denotes an isotropic point source.
    Vector3D connectLight(const Intersection& hit, const Vector3D& wo,
                          const Vector3D& pos, const Vector3D& normal, const Vector3D& weight,
                          double geometryLimit, const std::vector<Shape*>& objects)
    {
        const Vector3D offset = pos - hit.itsPoint;
        const double distance = offset.length();
        if (distance <= 2.0 * Epsilon)
            return Vector3D(0.0);

        const Vector3D wi = offset / distance;
        const double receiverCos = std::max(0.0, dot(hit.normal, wi));
        const double emitterCos = normal.lengthSq() == 0.0 ? 1.0 : std::max(0.0, dot(normal, -wi));
        const double geometry = std::min(geometryLimit, receiverCos * emitterCos / (distance * distance));
        if (geometry <= 0.0 || Utils::hasIntersection(Ray(hit.itsPoint, wi, 0, Epsilon, distance - Epsilon), objects))
            return Vector3D(0.0);

        return weight * hit.shape->getMaterial().getReflectance(hit.normal, wo, wi) * geometry;
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
        if (area > 0.0) {
            const Vector3D n = light.getNormal().normalized();
            direction = sampler.getSample(n);
            flux = light.getIntensity() * (pathWeight * area * 2.0 * pi * std::max(0.0, dot(n, direction)));
        } else if (light.getNormal().lengthSq() == 0.0) {
            direction = sampler.getSample(Vector3D(0, chooseHemisphere(generator) ? 1 : -1, 0));
            flux = light.getIntensity() * (pathWeight * 4.0 * pi);
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

Vector3D VplShader::computeColor(const Ray& r, const std::vector<Shape*>& objList,
                                 const std::vector<LightSource*>& lsList) const
{
    constexpr size_t maxCameraDepth = 4;
    Intersection hit;
    if (r.depth >= maxCameraDepth || !Utils::getClosestIntersection(r, objList, hit))
        return bgColor;

    const Material& material = hit.shape->getMaterial();
    if (material.hasSpecular() || material.hasTransmission())
        return computeColor(specularBounce(r, hit), objList, lsList);

    Vector3D color(0.0);
    if (material.isEmissive()) {
        if (r.depth == 0)
            return material.getEmissiveRadiance();
        if (r.isReflectedRay || r.isRefractedRay)
            color = material.getEmissiveRadiance();
    }
    if (!material.hasDiffuseOrGlossy())
        return color;

    for (const LightSource* light : lsList) {
        const double area = light->getArea();
        const Vector3D normal = light->getNormal();
        const bool point = normal.lengthSq() == 0.0;
        color += connectLight(hit, -r.d, light->sampleLightPosition(), point ? normal : normal.normalized(),
                              light->getIntensity() * (point ? 1.0 : area),
                              std::numeric_limits<double>::infinity(), objList);
    }
    for (const VPL& vpl : vplList)
        color += connectLight(hit, -r.d, vpl.pos, vpl.normal, vpl.flux, maxGeometry, objList);
    return color;
}