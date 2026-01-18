#define CUDA

#include <helper_cuda.h>
#include <cuda_dc.hpp>
#include <cstdio>
#include "types.hpp"

// stage 1 kernel calculating density values from a given function
// 
// calculates densities for voxel uniformly placed in volume offset - (offset + (scale, scale, scale) )
// writes in z order
template<DensityFunction F>
__global__ void gen_density_map(uint32_t size, vec_t<float, 3> offset, float scale, F func, density_t* density_data)
{
    unsigned int n = blockDim.x * gridDim.x;
    unsigned int morton = blockIdx.x*blockDim.x + threadIdx.x;
    while(morton < size * size * size)
    {
        vec_t<uint32_t, 3> voxelId = from_morton(morton);
        vec3_t sample_pos = vec3_t{voxelId}/size * scale + offset;
        density_data[morton] = func(sample_pos, density_data[morton]);
        morton += n;
    }
}

// stage 2 kernel generating sorted active voxels list
// first value in voxel_data is amount of active voxels
__global__ void gen_voxel_data(uint32_t size, density_t* data, uint32_t *voxel_data)
{
    const uint32_t size_cubed = size*size*size;
    extern  __shared__  uint32_t temp[];
    const uint32_t morton = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t laneId = threadIdx.x % 32;
    const uint32_t warpId = threadIdx.x / 32;

    vec_t<uint32_t, 3> voxelID = from_morton(morton);
    
    bool hasUnder0 = false, hasOver0 = false;

    density_t sample = data[morton];
    hasUnder0 = hasUnder0 || sample <= 0;
    hasOver0 = hasOver0 || sample > 0;
    if(voxelID[0] < size - 1)
    {
        sample = data[to_morton(voxelID + vec_t<uint32_t, 3>{1,0,0})];
        hasUnder0 = hasUnder0 || sample <= 0;
        hasOver0 = hasOver0 || sample > 0;
    }
    if(voxelID[1] < size - 1)
    {
        sample = data[to_morton(voxelID + vec_t<uint32_t, 3>{0,1,0})];
        hasUnder0 = hasUnder0 || sample <= 0;
        hasOver0 = hasOver0 || sample > 0;

        if(voxelID[0] < size - 1)
        {
            sample = data[to_morton(voxelID + vec_t<uint32_t, 3>{1,1,0})];
            hasUnder0 = hasUnder0 || sample <= 0;
            hasOver0 = hasOver0 || sample > 0;
        }
    }
    
    if(voxelID[2] + 1 < size)
    {
        sample = data[to_morton(voxelID + vec_t<uint32_t, 3>{0,0,1})];
        hasUnder0 = hasUnder0 || sample <= 0;
        hasOver0 = hasOver0 || sample > 0;
        if(voxelID[0] + 1 < size)
        {
            sample = data[to_morton(voxelID + vec_t<uint32_t, 3>{1,0,1})];
            hasUnder0 = hasUnder0 || sample <= 0;
            hasOver0 = hasOver0 || sample > 0;
        }
        if(voxelID[1] + 1 < size)
        {
            sample = data[to_morton(voxelID + vec_t<uint32_t, 3>{0,1,1})];
            hasUnder0 = hasUnder0 || sample <= 0;
            hasOver0 = hasOver0 || sample > 0;

            if(voxelID[0] + 1 < size)
            {
                sample = data[to_morton(voxelID + vec_t<uint32_t, 3>{1,1,1})];
                hasUnder0 = hasUnder0 || sample <= 0;
                hasOver0 = hasOver0 || sample > 0;
            }
        }
    }

    bool isActive = hasOver0 && hasUnder0;

    // 1st in-warp reduction
    uint32_t activeInWarpMap = __ballot_sync(0xFFFFFFFF, isActive);
    uint32_t inWarpOffset = __popc(activeInWarpMap << (32 - laneId));
    uint32_t activeInWarp = __popc(activeInWarpMap);
    
    // prepare values for 2nd reduction
    if(threadIdx.x%32 == 0) temp[threadIdx.x/32] = activeInWarp;

    // sync before 2nd reduction
    __syncthreads();
    
    if(warpId == 0)
    {
        // read values for 2nd reduction
        uint32_t activeInWarpPrefixSum = 0;
        if(threadIdx.x < blockDim.x/32)
            activeInWarpPrefixSum = temp[threadIdx.x];

        // 2nd in-warp reduction
        // calculates exclusive prefix sums of activeInWarp
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            uint32_t n = __shfl_up_sync(0xffffffff, activeInWarpPrefixSum, offset);
            if (laneId >= offset) activeInWarpPrefixSum += n;
        }
        temp[laneId] = activeInWarpPrefixSum;
    }
    
    

    // add amount of active voxels in block to total
    // store write offset for block to shared memory
    // only last thread in first warp has total sum
    if(threadIdx.x == 31)
        atomicAdd(&voxel_data[0], temp[31]);

    // sync to ensure temp[0] is valid
    __syncthreads();
    
    // add block write offset to writeIdx, giving final writeIdx
    // add 1 as offset because 1st value is amount of active voxels
    // results in final writeIdx
    uint32_t writeIdx = inWarpOffset + (temp[warpId] - activeInWarp)  + 1;

    if(isActive)
    {
        voxel_data[writeIdx] = morton;
        ((uint32_t*)data)[morton] = writeIdx;
    }
}

__global__ void gen_mesh(uint32_t size, uint32_t activeVoxelN, uint32_t *activeVoxels, uint32_t *elements, vec3_t *vertices, vec3_t *normals)
{
    uint32_t morton = blockDim.x * blockIdx.x + threadIdx.x;


}

struct testDensityFunctor
{
    density_t __device__ operator()(vec3_t v, density_t prev)
    {
        return (v == vec3_t{4,4,4}) ? 1.f : 0.f;
        //return (density_t)((uint32_t)(v[0] + v[1] + v[2]) % 2);
    }
};

void Test()
{
    uint32_t grid_size = 512;
    uint32_t grid_size_cubed = grid_size * grid_size * grid_size;
    density_t *density_data_device;
    cudaMalloc(&density_data_device, grid_size_cubed * sizeof(density_t));

    unsigned long long num_threads_1 = 512;
    unsigned long long num_blocks_1 = (grid_size_cubed + num_threads_1 - 1) / num_threads_1 / 64;

    printf("Launching density kernel: %d blocks, %d threads\n", num_blocks_1, num_threads_1);

    gen_density_map<<<num_blocks_1, num_threads_1>>>(grid_size, vec3_t{0,0,0}, (float)grid_size, testDensityFunctor{}, density_data_device);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(density_data_device);
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    }

    density_t *density_data_host = new density_t[grid_size_cubed];
    
    cudaMemcpy(density_data_host, density_data_device, sizeof(density_t) * grid_size_cubed, cudaMemcpyDeviceToHost);

    bool right_vals = true;
    for(unsigned int i = 0; i < grid_size_cubed; i++)
    {
        vec_t<uint32_t, 3> v = from_morton(i);
        right_vals = right_vals && density_data_host[i] == ((v[0]+v[1]+v[2])%2);
    }
    /*for(int k= 0; k < grid_size; k++)
    {
        for(int j= 0; j < grid_size; j++)
        {
            for(int i= 0; i < grid_size; i++)
            {
                printf("%f ", density_data_host[to_morton({i,j,k})]);
            }   
            putchar('\n');
        }   
        putchar('\n');
    }*/
    puts(right_vals ? "density_gen_test: SUCC" : "density_gen_test: FAIL");

    uint32_t *active_voxels_device;
    uint32_t active_voxels_count = (grid_size - 1) * (grid_size - 1) * (grid_size - 1) + 1;
    cudaMalloc(&active_voxels_device, sizeof(uint32_t) * active_voxels_count);
    unsigned long long num_threads_2 = 1024;
    unsigned long long num_blocks_2 = (grid_size_cubed + num_threads_2 - 1) / num_threads_2;
    unsigned long long shared_mem_size = 32*sizeof(float);

    printf("Launching active voxel kernel: %d blocks, %d threads\n", num_blocks_2, num_threads_2);
    gen_voxel_data<<<num_blocks_2, num_threads_2, shared_mem_size>>>(grid_size, density_data_device, active_voxels_device);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(density_data_device);
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
    uint32_t *active_voxels_host = new uint32_t[active_voxels_count];
    cudaMemcpy(active_voxels_host, active_voxels_device, sizeof(density_t) * active_voxels_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(density_data_host, density_data_device, sizeof(density_t) * grid_size_cubed, cudaMemcpyDeviceToHost);
    printf("Active voxels: %u\n", active_voxels_host[0]);
    for(unsigned int i = 1; i - 1 < active_voxels_host[0]; i++)
    {
        vec_t<uint32_t, 3> voxelId = from_morton(active_voxels_host[i]);
        printf("%u, %u: (%u, %u, %u) ", i, ((uint32_t*)density_data_host)[active_voxels_host[i]], voxelId[0], voxelId[1], voxelId[2]);
    }
    putchar('\n');

}