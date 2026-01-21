#include <cstdio>
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

// Test resolution
constexpr uint32_t GRID_SIZE = 256;

// Uncomment the desired test case:
// #define TEST_SPHERE
#define TEST_CUBE_SPHERE_CUTOUT

#ifdef TEST_SPHERE
using TestDensityFunctor = sphereDensityFunctor;
using TestGradientFunctor = sphereGradientFunctor;
#endif

#ifdef TEST_CUBE_SPHERE_CUTOUT
using TestDensityFunctor = cubeSphereCutoutDensityFunctor;
using TestGradientFunctor = cubeSphereCutoutGradientFunctor;
#endif

int main()
{
    cuda_dc::Mesh genMesh = cuda_dc::RunDualContouring(TestDensityFunctor{}, TestGradientFunctor{}, GRID_SIZE, 2.f);
    
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