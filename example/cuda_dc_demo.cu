#include <cstdio>
#include <cstring>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "renderer/renderer.hpp"
#include <helper_cuda.h>
#include "cuda_dc/cuda_dc.hpp"

static void glfw_error_callback(int error, const char* description)
{
    (void)error;
	std::fprintf(stderr, "GLFW error: %s\n", description);
}
static void GLAPIENTRY gl_error_callback(
		GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const GLchar* message,
		const void* userParam)
{
    (void)source;
    (void)id;
    (void)length;
    (void)userParam;
	if(type == GL_DEBUG_TYPE_ERROR)
		fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
			(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
			type, severity, message);
}

// All density/gradient functors work in normalized <0,1> range for all axes

// Torus - ring shape, good for testing curved surfaces with topology
static constexpr float __device__ __constant__ torus_R = 0.3f;  // Major radius (center to tube center)
static constexpr float __device__ __constant__ torus_r = 0.12f; // Minor radius (tube radius)
struct torusDensityFunctor
{
    cuda_dc::density_t __device__ operator()(cuda_dc::vec3_t v, cuda_dc::density_t prev)
    {
        (void)prev;
        cuda_dc::vec3_t c{0.5f, 0.5f, 0.5f};
        v = v - c;
        // Distance to torus: sqrt((sqrt(x^2 + z^2) - R)^2 + y^2) - r
        float q = sqrtf(v[0]*v[0] + v[2]*v[2]) - torus_R;
        return sqrtf(q*q + v[1]*v[1]) - torus_r;
    }
};
struct torusGradientFunctor
{
    cuda_dc::gradient_t __device__ operator()(cuda_dc::vec3_t v)
    {
        cuda_dc::vec3_t c{0.5f, 0.5f, 0.5f};
        v = v - c;
        float len_xz = sqrtf(v[0]*v[0] + v[2]*v[2]);
        if(len_xz < 1e-6f) len_xz = 1e-6f;
        float q = len_xz - torus_R;
        float len_qy = sqrtf(q*q + v[1]*v[1]);
        if(len_qy < 1e-6f) return cuda_dc::gradient_t{0, 1, 0};
        return cuda_dc::gradient_t{
            q * v[0] / (len_xz * len_qy),
            v[1] / len_qy,
            q * v[2] / (len_xz * len_qy)
        };
    }
};

// Gyroid - beautiful triply periodic minimal surface
static constexpr float __device__ __constant__ gyroid_scale = 6.0f * 3.14159265f;
static constexpr float __device__ __constant__ gyroid_thickness = 0.03f;
struct gyroidDensityFunctor
{
    cuda_dc::density_t __device__ operator()(cuda_dc::vec3_t v, cuda_dc::density_t prev)
    {
        (void)prev;
        float x = v[0] * gyroid_scale;
        float y = v[1] * gyroid_scale;
        float z = v[2] * gyroid_scale;
        float g = sinf(x)*cosf(y) + sinf(y)*cosf(z) + sinf(z)*cosf(x);
        return fabsf(g) - gyroid_thickness * gyroid_scale;
    }
};
struct gyroidGradientFunctor
{
    cuda_dc::gradient_t __device__ operator()(cuda_dc::vec3_t v)
    {
        float x = v[0] * gyroid_scale;
        float y = v[1] * gyroid_scale;
        float z = v[2] * gyroid_scale;
        float g = sinf(x)*cosf(y) + sinf(y)*cosf(z) + sinf(z)*cosf(x);
        float sign = (g >= 0) ? 1.0f : -1.0f;
        cuda_dc::gradient_t grad{
            sign * (cosf(x)*cosf(y) - sinf(z)*sinf(x)),
            sign * (-sinf(x)*sinf(y) + cosf(y)*cosf(z)),
            sign * (-sinf(y)*sinf(z) + cosf(z)*cosf(x))
        };
        return grad.norm();
    }
};

// Gear - cylinder with sinusoidal teeth, smooth wave-like profile
static constexpr float __device__ __constant__ gear_outer_r = 0.35f;   // Outer radius (tooth tips)
static constexpr float __device__ __constant__ gear_inner_r = 0.25f;  // Inner radius (tooth valleys)
static constexpr float __device__ __constant__ gear_hole_r = 0.08f;   // Center hole radius
static constexpr float __device__ __constant__ gear_half_h = 0.1f;    // Half height
static constexpr int __device__ __constant__ gear_num_teeth = 13;
struct gearDensityFunctor
{
    cuda_dc::density_t __device__ operator()(cuda_dc::vec3_t v, cuda_dc::density_t prev)
    {
        (void)prev;
        v = v - cuda_dc::vec3_t{0.5f, 0.5f, 0.5f};
        
        float r = sqrtf(v[0]*v[0] + v[2]*v[2]);
        float angle = atan2f(v[2], v[0]);
        
        // Sinusoidal tooth profile: radius varies as sin(angle * num_teeth)
        // Maps from inner_r (valley) to outer_r (peak)
        float tooth_amplitude = (gear_outer_r - gear_inner_r) * 0.5f;
        float tooth_center_r = (gear_outer_r + gear_inner_r) * 0.5f;
        float gear_sin = sinf(angle * gear_num_teeth);
        float gear_r = tooth_center_r + tooth_amplitude * gear_sin;
        
        // Distance to gear profile (2D)
        float d_gear_profile = r - gear_r;
        
        // Top/bottom caps
        float d_top = fabsf(v[1]) - gear_half_h;
        
        // Intersect with height (extrusion)
        float d_gear_body = fmaxf(d_gear_profile, d_top);
        
        // Subtract center hole
        float d_hole = gear_hole_r - r;
        
        return fmaxf(d_gear_body, d_hole);
    }
};
struct gearGradientFunctor
{
    cuda_dc::gradient_t __device__ operator()(cuda_dc::vec3_t v)
    {
        v = v - cuda_dc::vec3_t{0.5f, 0.5f, 0.5f};
        
        float r = sqrtf(v[0]*v[0] + v[2]*v[2]);
        float angle = atan2f(v[2], v[0]);
        
        // Compute gear radius at this angle
        float tooth_amplitude = (gear_outer_r - gear_inner_r) * 0.5f;
        float tooth_center_r = (gear_outer_r + gear_inner_r) * 0.5f;
        float gear_r = tooth_center_r + tooth_amplitude * sinf(angle * gear_num_teeth);
        
        float d_gear_profile = r - gear_r;
        float d_top = fabsf(v[1]) - gear_half_h;
        float d_hole = gear_hole_r - r;
        
        // Check if we're on the hole surface
        if(d_hole > fmaxf(d_gear_profile, d_top))
        {
            if(r < 1e-6f) return cuda_dc::gradient_t{-1, 0, 0};
            return cuda_dc::gradient_t{-v[0]/r, 0, -v[2]/r};
        }
        
        // Check if we're on top/bottom
        if(d_top > d_gear_profile)
        {
            return cuda_dc::gradient_t{0, (v[1] > 0) ? 1.0f : -1.0f, 0};
        }
        
        // On the sinusoidal gear edge - compute analytical gradient
        // The surface is r = gear_r(angle), so implicit: F = r - gear_r(angle) = 0
        // grad F = (dF/dx, dF/dy, dF/dz)
        // dF/dr = 1, dF/dangle = -d(gear_r)/dangle = -tooth_amplitude * cos(angle * num_teeth) * num_teeth
        if(r < 1e-6f) return cuda_dc::gradient_t{1, 0, 0};
        
        // Convert from polar gradient to Cartesian
        // dr/dx = x/r, dr/dz = z/r
        // dangle/dx = -z/r^2, dangle/dz = x/r^2
        float dgear_dangle = tooth_amplitude * cosf(angle * gear_num_teeth) * gear_num_teeth;
        
        float grad_x = v[0]/r + dgear_dangle * v[2]/(r*r);
        float grad_z = v[2]/r - dgear_dangle * v[0]/(r*r);
        
        cuda_dc::gradient_t grad{grad_x, 0, grad_z};
        return grad.norm();
    }
};

constexpr float __device__ __constant__ sphere_r = 0.35f;
struct sphereDensityFunctor
{
    cuda_dc::density_t __device__ operator()(cuda_dc::vec3_t v, cuda_dc::density_t prev)
    {
        (void)prev;
        // Sphere centered at (0.5, 0.5, 0.5) with radius 0.45
        
        cuda_dc::vec3_t center{0.5f, 0.5f, 0.5f};
        return (v - center).lenSqr() / (sphere_r * sphere_r) - 1.0f;
    }
};
struct sphereGradientFunctor
{
    cuda_dc::gradient_t __device__ operator()(cuda_dc::vec3_t v)
    {
        // Gradient of sphere SDF: normalized direction from center
        cuda_dc::vec3_t center{0.5f, 0.5f, 0.5f};
        return (v - center).norm();
    }
};

// Cube with sphere cut out from one corner - demonstrates edge preservation
// The cube has sharp edges, and the sphere cutout creates a smooth curved surface
// Dual contouring should preserve both the sharp cube edges AND the smooth sphere surface
static constexpr float __device__ __constant__ cube_size = 0.7f;
static constexpr float __device__ __constant__ cutout_radius = 0.45f;
static constexpr cuda_dc::vec3_t __device__ __constant__ cutout_center = 
    cuda_dc::vec3_t{1,1,1} * (1.0f - 0.5f * (1.f - cube_size));
struct cubeSphereCutoutDensityFunctor
{
    cuda_dc::density_t __device__ operator()(cuda_dc::vec3_t v, cuda_dc::density_t prev)
    {
        cuda_dc::vec3_t center_axis_dist = v - cuda_dc::vec3_t{0.5f, 0.5f, 0.5f};
        center_axis_dist = {fabsf(center_axis_dist[0]), fabsf(center_axis_dist[1]), fabsf(center_axis_dist[2])};
        float cube_dist = fmaxf(fmaxf(center_axis_dist[0], center_axis_dist[1]), center_axis_dist[2]) - cube_size * 0.5f;

        float cutout_dist = -(v - cutout_center).len() + cutout_radius;

        return fmaxf(cutout_dist, cube_dist);
    }
};

struct cubeSphereCutoutGradientFunctor
{
    cuda_dc::gradient_t __device__ operator()(cuda_dc::vec3_t v)
    {
        cuda_dc::vec3_t center_axis_dist = v - cuda_dc::vec3_t{0.5f, 0.5f, 0.5f};
        uint8_t max_axis = 0;
        max_axis = (fabsf(center_axis_dist[1]) > fabsf(center_axis_dist[0])) ? 1 : max_axis;
        max_axis = (fabsf(center_axis_dist[2]) > fabsf(center_axis_dist[max_axis])) ? 2 : max_axis;

        float cube_dist = fabsf(center_axis_dist[max_axis]) - cube_size * 0.5f;

        cuda_dc::gradient_t cube_gradient = cuda_dc::gradient_t{
            (max_axis == 0) ? ((center_axis_dist[0] > 0) ? 1.f : -1.f) : 0.f,
            (max_axis == 1) ? ((center_axis_dist[1] > 0) ? 1.f : -1.f) : 0.f,
            (max_axis == 2) ? ((center_axis_dist[2] > 0) ? 1.f : -1.f) : 0.f
        };

        cuda_dc::gradient_t cutout_gradient = cutout_center - v;

        float cutout_dist = -(v - cutout_center).len() + cutout_radius;

        if(cube_dist < cutout_dist)
            return cutout_gradient.norm();
        else
            return cube_gradient.norm();
    }
};

// Default resolution
constexpr uint32_t DEFAULT_GRID_SIZE = 128;

// Model enumeration for runtime selection
enum class Model {
    SPHERE,
    CUBE_SPHERE_CUTOUT,
    TORUS,
    GYROID,
    GEAR
};

// Template function to run dual contouring with specific model
template<typename DensityFunctor, typename GradientFunctor>
cuda_dc::Mesh runWithModel(uint32_t gridSize)
{
    return cuda_dc::RunDualContouring(DensityFunctor{}, GradientFunctor{}, gridSize, 2.f);
}

// Model dispatch function
cuda_dc::Mesh generateMesh(Model model, uint32_t gridSize)
{
    switch(model)
    {
        case Model::SPHERE:
            return runWithModel<sphereDensityFunctor, sphereGradientFunctor>(gridSize);
        case Model::CUBE_SPHERE_CUTOUT:
            return runWithModel<cubeSphereCutoutDensityFunctor, cubeSphereCutoutGradientFunctor>(gridSize);
        case Model::TORUS:
            return runWithModel<torusDensityFunctor, torusGradientFunctor>(gridSize);
        case Model::GYROID:
            return runWithModel<gyroidDensityFunctor, gyroidGradientFunctor>(gridSize);
        case Model::GEAR:
            return runWithModel<gearDensityFunctor, gearGradientFunctor>(gridSize);
        default:
            return runWithModel<sphereDensityFunctor, sphereGradientFunctor>(gridSize);
    }
}

const char* modelToString(Model model)
{
    switch(model)
    {
        case Model::SPHERE: return "sphere";
        case Model::CUBE_SPHERE_CUTOUT: return "cube_sphere_cutout";
        case Model::TORUS: return "torus";
        case Model::GYROID: return "gyroid";
        case Model::GEAR: return "gear";
        default: return "unknown";
    }
}

bool parseModel(const char* str, Model& model)
{
    if(strcmp(str, "sphere") == 0) { model = Model::SPHERE; return true; }
    if(strcmp(str, "cube_sphere_cutout") == 0) { model = Model::CUBE_SPHERE_CUTOUT; return true; }
    if(strcmp(str, "torus") == 0) { model = Model::TORUS; return true; }
    if(strcmp(str, "gyroid") == 0) { model = Model::GYROID; return true; }
    if(strcmp(str, "gear") == 0) { model = Model::GEAR; return true; }
    return false;
}

void printHelp(const char* programName)
{
    printf("Usage: %s [options]\n", programName);
    printf("\nOptions:\n");
    printf("  -res <value>     Set grid resolution, must be power of 2 between 8 and 1024 (default: %u)\n", DEFAULT_GRID_SIZE);
    printf("  -model <name>    Select SDF model to render\n");
    printf("  -h               Show this help message\n");
    printf("\nAvailable models:\n");
    printf("  sphere             - Simple sphere\n");
    printf("  cube_sphere_cutout - Cube with spherical cutout (sharp edges)\n");
    printf("  torus              - Torus/donut shape\n");
    printf("  gyroid             - Triply periodic minimal surface\n");
    printf("  gear               - Gear with rectangular teeth\n");
}

int main(int argc, char** argv)
{
    // Parse command line arguments
    uint32_t gridSize = DEFAULT_GRID_SIZE;
    Model model = Model::GEAR;
    
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            printHelp(argv[0]);
            return 0;
        }
        else if(strcmp(argv[i], "-res") == 0)
        {
            if(i + 1 >= argc)
            {
                fprintf(stderr, "Error: -res requires a value\n");
                return 1;
            }
            gridSize = static_cast<uint32_t>(atoi(argv[++i]));
            if(gridSize < 8 || gridSize > 1024 || (gridSize & (gridSize - 1)) != 0)
            {
                fprintf(stderr, "Error: resolution must be power of 2 between 8 and 1024\n");
                return 1;
            }
        }
        else if(strcmp(argv[i], "-model") == 0)
        {
            if(i + 1 >= argc)
            {
                fprintf(stderr, "Error: -model requires a model name\n");
                return 1;
            }
            if(!parseModel(argv[++i], model))
            {
                fprintf(stderr, "Error: unknown model '%s'. Use -h for list of models.\n", argv[i]);
                return 1;
            }
        }
        else
        {
            fprintf(stderr, "Error: unknown option '%s'. Use -h for help.\n", argv[i]);
            return 1;
        }
    }
    
    printf("Generating mesh: model=%s, resolution=%u\n", modelToString(model), gridSize);
    cuda_dc::Mesh genMesh = generateMesh(model, gridSize);
    
    // rendering setup

    glfwSetErrorCallback(glfw_error_callback);
    if(!glfwInit())
    {
        const char *errorString;
        glfwGetError(&errorString);
        printf("GLFW initialization failed: %s\n", errorString);
        return 1;
    }

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "renderer demo", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    GLenum glewInitCode = glewInit();
    if(glewInitCode != GLEW_OK)
    {
        printf("GLEW initialization failed: %s\n", glewGetErrorString(glewInitCode));
        return 1;
    }
    glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(gl_error_callback, nullptr);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    

    glClearColor(0.5f, 0.5f, 0.5f, 1.f);

    stbi_set_flip_vertically_on_load(1);
    // renderer
    
    // camera setup
    render::Camera camera;
    camera.perspective(60, 1280, 720);
    camera.transform.position({0.f, 2.5f, 0.f});
    camera.transform.orientation(glm::quat(glm::vec3(glm::radians(-45.f), 0.f, 0.f)));
    camera.projection();
    camera.transform.matrix();
    camera.transform.inverse();
    camera.Use();

    // mesh setup
    render::TypedSharedBuffer<render::InstanceData> meshInstanceBuffer{1};
    render::Transform meshTransform{meshInstanceBuffer, &meshInstanceBuffer.data()->model, &meshInstanceBuffer.data()->inverse_model};
    render::Mesh mesh(genMesh.vertex_count(), (glm::vec3*)genMesh.verts);
    mesh.initElements(genMesh.index_count(), genMesh.indices);

    meshTransform.position({0, 0, -2.5f});
    meshTransform.inverse();
    meshTransform.matrix();

    // lighting setup
    render::FragmentShaderBRDF::Lighting lighting;
    lighting.uniformData.ambientLight = glm::vec3{0.1f, 0.1f, 0.1f};
    lighting.uniformData.lightColor = glm::vec3{1.f, 1.f, 1.f};
    lighting.uniformData.view_lightDirection = glm::normalize(glm::vec3{1.f, 1.f, 1.f});


    // shader creation
    render::shader_location = "renderer/renderer/shader/processed";
    render::ShaderProgramBRDF shaderBRDF;

    //  material setup
    render::FragmentShaderBRDF::Material material;

    // setting lighting to use 
    lighting.Use();

    // setting shader to use
    shaderBRDF.Use();

    while(!glfwWindowShouldClose(window))
    {
        material.Use();
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mesh.Draw(meshInstanceBuffer);
        meshTransform.orientation(glm::quat({0.f, glm::radians(0.2f), 0.f}) * meshTransform.orientation());
        meshTransform.inverse();
        meshTransform.matrix();

        glfwSwapBuffers(window);
    }
    return 0;
}