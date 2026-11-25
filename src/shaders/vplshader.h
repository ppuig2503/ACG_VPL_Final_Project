#ifndef VPLSHADER_H
#define VPLSHADER_H

#include "shader.h"

struct VPL
{
    Vector3D pos;
    Vector3D normal;
    Vector3D flux; // Incident path flux times the diffuse BRDF at this vertex.
};

class VplShader : public Shader
{
public:
    explicit VplShader(Vector3D bgColor = Vector3D(0.0), double maxGeometry = 1.0);

    void generateVPLs(const std::vector<Shape*>& objList,
                      const std::vector<LightSource*>& lsList, int numPaths, int maxDepth);
    Vector3D computeColor(const Ray& r, const std::vector<Shape*>& objList,
                          const std::vector<LightSource*>& lsList) const override;

private:
    double maxGeometry;
    std::vector<VPL> vplList;
};

#endif // VPLSHADER_H