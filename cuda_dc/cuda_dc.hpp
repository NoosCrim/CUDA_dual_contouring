#pragma once
#include "cuda_dc/types.hpp"

namespace cuda_dc{
// stage 1 kernel calculating density values from a given function
// 
// calculates densities for voxel from formula "offset + scale * ID / size"
// writes in z order
template<DensityFunctor F>
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

struct ActiveVoxel
{
    uint32_t edgeOwners[3];
    uint32_t morton;
    struct aaa
    {
        vec3_t point;
        gradient_t gradient;
    } hermite_data[3];
    uint32_t edge_idx;
    uint32_t vertex_idx;  // Index into vertex array (UINT32_MAX if no vertex)
    uint8_t corner_case; // bit 1 for corner within model, 0 for outside
};
struct ActiveData
{
    uint32_t active_voxel_n, geometry_edge_n, active_edge_n, vertex_n;
    ActiveVoxel data[];
};

// stage 2 kernel generating active voxels list
__global__ void gen_active_data(uint32_t size, const density_t* __restrict__ density_data, ActiveData *active_data, uint32_t *active_index_map);

// mapping of cube_cases to edge_cases
// edges in order mapped to bits (least significant first)
// {0,0,0} -> {1,0,0}
// {0,0,0} -> {0,1,0}
// {0,0,0} -> {0,0,1}

// {1,0,0} -> {1,1,0}
// {1,0,0} -> {1,0,1}

// {0,1,0} -> {1,1,0}
// {0,1,0} -> {0,0,1}

// {0,0,1} -> {1,0,1}
// {0,0,1} -> {0,1,1}

// {1,1,0} -> {1,1,1}
// {1,0,1} -> {1,1,1}
// {0,1,1} -> {1,1,1}
static constexpr __constant__ __device__ uint16_t edge_case_map[256] = 
{
    0x000, 0x007, 0x019, 0x01E, 0x062, 0x065, 0x07B, 0x07C, 
    0x228, 0x22F, 0x231, 0x236, 0x24A, 0x24D, 0x253, 0x254, 
    0x184, 0x183, 0x19D, 0x19A, 0x1E6, 0x1E1, 0x1FF, 0x1F8, 
    0x3AC, 0x3AB, 0x3B5, 0x3B2, 0x3CE, 0x3C9, 0x3D7, 0x3D0, 
    0x490, 0x497, 0x489, 0x48E, 0x4F2, 0x4F5, 0x4EB, 0x4EC, 
    0x6B8, 0x6BF, 0x6A1, 0x6A6, 0x6DA, 0x6DD, 0x6C3, 0x6C4, 
    0x514, 0x513, 0x50D, 0x50A, 0x576, 0x571, 0x56F, 0x568, 
    0x73C, 0x73B, 0x725, 0x722, 0x75E, 0x759, 0x747, 0x740, 
    0x940, 0x947, 0x959, 0x95E, 0x922, 0x925, 0x93B, 0x93C, 
    0xB68, 0xB6F, 0xB71, 0xB76, 0xB0A, 0xB0D, 0xB13, 0xB14, 
    0x8C4, 0x8C3, 0x8DD, 0x8DA, 0x8A6, 0x8A1, 0x8BF, 0x8B8, 
    0xAEC, 0xAEB, 0xAF5, 0xAF2, 0xA8E, 0xA89, 0xA97, 0xA90, 
    0xDD0, 0xDD7, 0xDC9, 0xDCE, 0xDB2, 0xDB5, 0xDAB, 0xDAC, 
    0xFF8, 0xFFF, 0xFE1, 0xFE6, 0xF9A, 0xF9D, 0xF83, 0xF84, 
    0xC54, 0xC53, 0xC4D, 0xC4A, 0xC36, 0xC31, 0xC2F, 0xC28, 
    0xE7C, 0xE7B, 0xE65, 0xE62, 0xE1E, 0xE19, 0xE07, 0xE00, 
    0xE00, 0xE07, 0xE19, 0xE1E, 0xE62, 0xE65, 0xE7B, 0xE7C, 
    0xC28, 0xC2F, 0xC31, 0xC36, 0xC4A, 0xC4D, 0xC53, 0xC54, 
    0xF84, 0xF83, 0xF9D, 0xF9A, 0xFE6, 0xFE1, 0xFFF, 0xFF8, 
    0xDAC, 0xDAB, 0xDB5, 0xDB2, 0xDCE, 0xDC9, 0xDD7, 0xDD0, 
    0xA90, 0xA97, 0xA89, 0xA8E, 0xAF2, 0xAF5, 0xAEB, 0xAEC, 
    0x8B8, 0x8BF, 0x8A1, 0x8A6, 0x8DA, 0x8DD, 0x8C3, 0x8C4, 
    0xB14, 0xB13, 0xB0D, 0xB0A, 0xB76, 0xB71, 0xB6F, 0xB68, 
    0x93C, 0x93B, 0x925, 0x922, 0x95E, 0x959, 0x947, 0x940, 
    0x740, 0x747, 0x759, 0x75E, 0x722, 0x725, 0x73B, 0x73C, 
    0x568, 0x56F, 0x571, 0x576, 0x50A, 0x50D, 0x513, 0x514, 
    0x6C4, 0x6C3, 0x6DD, 0x6DA, 0x6A6, 0x6A1, 0x6BF, 0x6B8, 
    0x4EC, 0x4EB, 0x4F5, 0x4F2, 0x48E, 0x489, 0x497, 0x490, 
    0x3D0, 0x3D7, 0x3C9, 0x3CE, 0x3B2, 0x3B5, 0x3AB, 0x3AC, 
    0x1F8, 0x1FF, 0x1E1, 0x1E6, 0x19A, 0x19D, 0x183, 0x184, 
    0x254, 0x253, 0x24D, 0x24A, 0x236, 0x231, 0x22F, 0x228, 
    0x07C, 0x07B, 0x065, 0x062, 0x01E, 0x019, 0x007, 0x000
};
// stage 3 generating hermite data with optimized parallel reduction for edge counting
template<GradientFunctor F>
__global__ void gen_hermite_data(uint32_t size, float voxel_size, density_t* density_data, ActiveData *active_data, uint32_t *active_index_map, F gradient_function)
{
    // Shared memory layout: 33 words (same as gen_active_data)
    // Reused for each reduction sequentially
    extern __shared__ uint32_t temp[];
    
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t laneId = threadIdx.x % 32;
    const uint32_t warpId = threadIdx.x / 32;
    
    uint32_t local_hermite_count = 0;
    uint32_t local_geometry_count = 0;
    uint32_t local_vertex_count = 0;
    
    // Only process if within active voxel range
    if(tid < active_data->active_voxel_n)
    {
        uint32_t morton = active_data->data[tid].morton;
        uint8_t corner_case = active_data->data[tid].corner_case;
        uint16_t edge_case = edge_case_map[corner_case];
        vec_t<uint32_t, 3> voxelID = from_morton(morton);

        density_t samples[4];
        samples[0] = density_data[morton];
        samples[0] = (samples[0] < 0) ? -(samples[0]) : samples[0];

        vec3_t ref_point = voxel_size * (vec3_t)voxelID;
        float inv_size = 1.0f / size;  // For normalizing to <0,1> range
        vec3_t point;
        gradient_t gradient;
        
        // Edge 0: X-direction edge, quad needs voxels at Y-1 and Z-1
        if((edge_case & (1u << 0)) && voxelID[0] < (size - 1))
        {
            samples[1] = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{1,0,0})];
            samples[1] = (samples[1] < 0) ? -(samples[1]) : samples[1];
            point = ref_point + vec3_t{voxel_size * samples[0]/(samples[0] + samples[1]), 0, 0};
            // Gradient functor expects normalized <0,1> coordinates
            gradient = gradient_function(vec3_t{voxelID[0] + samples[0]/(samples[0] + samples[1]), (float)voxelID[1], (float)voxelID[2]} * inv_size);
            active_data->data[tid].hermite_data[0] = {point, gradient};
            ++local_hermite_count;
            // Geometry count: all 4 voxels must have valid vertices (all coords < size-1)
            if(voxelID[1] > 0 && voxelID[1] < (size - 1) && voxelID[2] > 0 && voxelID[2] < (size - 1))
                ++local_geometry_count;
        }
        // Edge 1: Y-direction edge, quad needs voxels at X-1 and Z-1
        if((edge_case & (1u << 1)) && voxelID[1] < (size - 1))
        {
            samples[2] = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{0,1,0})];
            samples[2] = (samples[2] < 0) ? -(samples[2]) : samples[2];
            point = ref_point + vec3_t{0, voxel_size * samples[0]/(samples[0] + samples[2]), 0};
            // Gradient functor expects normalized <0,1> coordinates
            gradient = gradient_function(vec3_t{(float)voxelID[0], voxelID[1] + samples[0]/(samples[0] + samples[2]), (float)voxelID[2]} * inv_size);
            active_data->data[tid].hermite_data[1] = {point, gradient};
            ++local_hermite_count;
            // Geometry count: all 4 voxels must have valid vertices (all coords < size-1)
            if(voxelID[0] > 0 && voxelID[0] < (size - 1) && voxelID[2] > 0 && voxelID[2] < (size - 1))
                ++local_geometry_count;
        }
        // Edge 2: Z-direction edge, quad needs voxels at X-1 and Y-1
        if((edge_case & (1u << 2)) && voxelID[2] < (size - 1))
        {
            samples[3] = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{0,0,1})];
            samples[3] = (samples[3] < 0) ? -(samples[3]) : samples[3];
            point = ref_point + vec3_t{0, 0, voxel_size * samples[0]/(samples[0] + samples[3])};
            // Gradient functor expects normalized <0,1> coordinates
            gradient = gradient_function(vec3_t{(float)voxelID[0], (float)voxelID[1], voxelID[2] + samples[0]/(samples[0] + samples[3])} * inv_size);
            active_data->data[tid].hermite_data[2] = {point, gradient};
            ++local_hermite_count;
            // Geometry count: all 4 voxels must have valid vertices (all coords < size-1)
            if(voxelID[0] > 0 && voxelID[0] < (size - 1) && voxelID[1] > 0 && voxelID[1] < (size - 1))
                ++local_geometry_count;
        }
        
        // Voxel needs a vertex if it can be referenced by any quad
        // This happens when all coordinates are < size-1
        if(voxelID[0] < (size - 1) && voxelID[1] < (size - 1) && voxelID[2] < (size - 1))
            local_vertex_count = 1;
    }
    
    // === Hermite count reduction (XOR, no indices needed) ===
    // Step 1: warp-level reduction
    uint32_t warp_hermite_sum = local_hermite_count;
    #pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
        warp_hermite_sum += __shfl_xor_sync(0xFFFFFFFF, warp_hermite_sum, mask);
    
    // Step 2: copy to shared memory
    if(laneId == 0) temp[warpId] = warp_hermite_sum;
    __syncthreads();
    
    // Step 3-4: load into first warp and do second reduction
    if(warpId == 0)
    {
        uint32_t hermiteWarpSum = (laneId < blockDim.x/32) ? temp[laneId] : 0;
        #pragma unroll
        for(int mask = 16; mask > 0; mask >>= 1)
            hermiteWarpSum += __shfl_xor_sync(0xFFFFFFFF, hermiteWarpSum, mask);
        
        // Step 5: single thread adds to global counter
        if(laneId == 0)
            atomicAdd(&active_data->active_edge_n, hermiteWarpSum);
    }

    // === Geometry edge reduction (prefix sum, need indices) ===
    // Note: local_geometry_count can be 0-3, so we need actual sum, not ballot
    // Step 1: warp-level prefix sum using shuffle
    uint32_t geomWarpPrefixSum = local_geometry_count;
    #pragma unroll
    for(int offset = 1; offset < 32; offset <<= 1)
    {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, geomWarpPrefixSum, offset);
        if(laneId >= offset) geomWarpPrefixSum += n;
    }
    uint32_t geomInWarp = __shfl_sync(0xFFFFFFFF, geomWarpPrefixSum, 31); // Total for this warp
    uint32_t geomInWarpOffset = geomWarpPrefixSum - local_geometry_count; // Exclusive prefix sum
    
    // Step 2: copy warp totals to shared memory
    if(laneId == 31) temp[warpId] = geomInWarp;
    __syncthreads();
    
    // Step 3-4: first warp does prefix sum across warp totals
    if(warpId == 0)
    {
        uint32_t geomBlockPrefixSum = (laneId < blockDim.x/32) ? temp[laneId] : 0;
        #pragma unroll
        for(int offset = 1; offset < 32; offset <<= 1)
        {
            uint32_t n = __shfl_up_sync(0xFFFFFFFF, geomBlockPrefixSum, offset);
            if(laneId >= offset) geomBlockPrefixSum += n;
        }
        temp[laneId] = geomBlockPrefixSum;
        
        // Step 5: single thread adds to global counter and stores block offset
        if(laneId == 31)
            temp[32] = atomicAdd(&active_data->geometry_edge_n, geomBlockPrefixSum);
    }
    __syncthreads();
    
    // Calculate final edge index: block_offset + warp_offset + thread_offset
    uint32_t warpOffset = (warpId > 0) ? temp[warpId - 1] : 0;
    uint32_t final_edge_idx = temp[32] + warpOffset + geomInWarpOffset;
    
    // Need sync before reusing temp[] for vertex reduction
    __syncthreads();
    
    // === Vertex count reduction (prefix sum, need indices) ===
    // Step 1: warp-level prefix sum (using ballot since vertex count is 0 or 1)
    uint32_t vertInWarpMap = __ballot_sync(0xFFFFFFFF, local_vertex_count > 0);
    uint32_t vertInWarpOffset = __popc(vertInWarpMap << (32 - laneId));
    uint32_t vertInWarp = __popc(vertInWarpMap);
    
    // Step 2: copy to shared memory
    if(laneId == 0) temp[warpId] = vertInWarp;
    __syncthreads();
    
    // Step 3-4: load into first warp and do second reduction (prefix sum)
    if(warpId == 0)
    {
        uint32_t vertWarpPrefixSum = (laneId < blockDim.x/32) ? temp[laneId] : 0;
        #pragma unroll
        for(int offset = 1; offset < 32; offset <<= 1)
        {
            uint32_t n = __shfl_up_sync(0xFFFFFFFF, vertWarpPrefixSum, offset);
            if(laneId >= offset) vertWarpPrefixSum += n;
        }
        temp[laneId] = vertWarpPrefixSum;
        
        // Step 5: single thread adds to global counter and stores block offset
        if(laneId == 31)
            temp[32] = atomicAdd(&active_data->vertex_n, vertWarpPrefixSum);
    }
    __syncthreads();
    
    // Calculate final vertex index
    uint32_t vertWarpOffset = (warpId > 0) ? temp[warpId - 1] : 0;
    uint32_t final_vertex_idx = temp[32] + vertWarpOffset + vertInWarpOffset;
    
    // Write indices to active data
    if(tid < active_data->active_voxel_n)
    {
        if(local_geometry_count > 0)
            active_data->data[tid].edge_idx = final_edge_idx;
        
        if(local_vertex_count > 0)
            active_data->data[tid].vertex_idx = final_vertex_idx;
        else
            active_data->data[tid].vertex_idx = UINT32_MAX;
    }
}

// stage 4 generating vertices and indices
__global__ void gen_mesh(uint32_t size, float voxel_size, const density_t* __restrict__ density_data, const ActiveData* __restrict__ active_data, const uint32_t* __restrict__ active_index_map, vert_t* __restrict__ vertices, uint32_t* __restrict__ indices);

template<DensityFunctor F1, GradientFunctor F2>
Mesh RunDualContouring(F1 density_functor, F2 gradient_functor, uint32_t grid_size, float mesh_size = 2.0f)
{
    float milli = 0.0f, milli_total = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint32_t grid_size_cubed = grid_size * grid_size * grid_size;
    density_t *density_data_device;
    checkCudaErrors(cudaMalloc(&density_data_device, grid_size_cubed * sizeof(density_t)));

    unsigned long long num_threads_1 = 512;
    unsigned long long num_blocks_1 = (grid_size_cubed + num_threads_1 - 1) / num_threads_1;
    printf("Launching density kernel: %llu blocks, %llu threads\n", num_blocks_1, num_threads_1);
    cudaEventRecord(start);
    
    gen_density_map<<<num_blocks_1, num_threads_1>>>(grid_size, vec3_t{0,0,0}, 1.0f, density_functor, density_data_device);
    
    cudaEventRecord(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(density_data_device);
        return {};
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return {};
    }

    cudaEventElapsedTime(&milli, start, stop);
    milli_total += milli;
    printf("Density kernel execution time: %f ms\n", milli);


    ActiveData *active_voxels_device;
    uint32_t *active_index_map;
    checkCudaErrors(cudaMalloc(&active_voxels_device, sizeof(ActiveData) + sizeof(ActiveVoxel) * grid_size_cubed));
    checkCudaErrors(cudaMemset(active_voxels_device, 0, sizeof(ActiveData)));  // Initialize counters to 0
    checkCudaErrors(cudaMalloc(&active_index_map, sizeof(uint32_t) * grid_size_cubed));
    unsigned long long num_threads_2 = 256;
    unsigned long long num_blocks_2 = (grid_size_cubed + num_threads_2 - 1) / num_threads_2;
    unsigned long long shared_mem_size = 33*sizeof(uint32_t);

    printf("\nLaunching active data kernel: %d blocks, %d threads\n", num_blocks_2, num_threads_2);
    cudaEventRecord(start);

    gen_active_data<<<num_blocks_2, num_threads_2, shared_mem_size>>>(grid_size, density_data_device, active_voxels_device, active_index_map);
    
    cudaEventRecord(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(density_data_device);
        return {};
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return {};
    }

    cudaEventElapsedTime(&milli, start, stop);
    milli_total += milli;
    printf("Active data kernel execution time: %f ms\n", milli);

    ActiveData active_voxels_host;
    cudaMemcpy(&active_voxels_host, active_voxels_device, sizeof(ActiveData), cudaMemcpyDeviceToHost);
    printf("Active voxels: %u\n", active_voxels_host.active_voxel_n);

    unsigned long long num_threads_3 = 512;
    unsigned long long num_blocks_3 = (active_voxels_host.active_voxel_n + num_threads_3 - 1) / num_threads_3;
    if(num_blocks_3 == 0) num_blocks_3 = 1;
    // Shared memory: 33 words (same as gen_active_data)
    unsigned long long shared_mem_size_3 = 33 * sizeof(uint32_t);
    printf("\nLaunching gradient kernel: %llu blocks, %llu threads\n", num_blocks_3, num_threads_3);
    
    cudaEventRecord(start);

    gen_hermite_data<<<num_blocks_3, num_threads_3, shared_mem_size_3>>>(grid_size, 1.f, density_data_device, active_voxels_device, active_index_map, gradient_functor);

    cudaEventRecord(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(density_data_device);
        return {};
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    }

    cudaEventElapsedTime(&milli, start, stop);
    milli_total += milli;
    printf("Gradient kernel execution time: %f ms\n", milli);

    cudaMemcpy(&active_voxels_host, active_voxels_device, sizeof(ActiveData), cudaMemcpyDeviceToHost);
    unsigned long long vertexCount = active_voxels_host.vertex_n;
    unsigned long long indexCount = active_voxels_host.geometry_edge_n * 6llu;
    unsigned long long hermiteCount = active_voxels_host.active_edge_n;

    printf("vertexCount: %llu, indexCount: %llu, hermiteCount: %llu\n", vertexCount, indexCount, hermiteCount);


    vert_t *verts_device;
    uint32_t *idxs_device;
    checkCudaErrors(cudaMalloc(&verts_device, vertexCount * sizeof(vert_t)));
    checkCudaErrors(cudaMalloc(&idxs_device, indexCount * sizeof(uint32_t)));

    unsigned long long num_threads_4 = 1024;
    unsigned long long num_blocks_4 = (active_voxels_host.active_voxel_n + num_threads_4 - 1) / num_threads_4;
    if(num_blocks_4 == 0) num_blocks_4 = 1;
    printf("\nLaunching mesh kernel: %llu blocks, %llu threads\n", num_blocks_4, num_threads_4);

    cudaEventRecord(start);

    gen_mesh<<<num_blocks_4, num_threads_4>>>(grid_size, mesh_size / grid_size, density_data_device, active_voxels_device, active_index_map, verts_device, idxs_device);

    cudaEventRecord(stop);


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(density_data_device);
        return{};
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return {};
    }

    cudaEventElapsedTime(&milli, start, stop);
    milli_total += milli;
    printf("Mesh kernel execution time: %f ms\n", milli);

    puts("\nMesh generation successful!\n\n");
    printf("Total compute time: %f ms\n\n", milli_total);
    Mesh outMesh{vertexCount, indexCount};
    checkCudaErrors(cudaMemcpy(outMesh.verts, verts_device, vertexCount * sizeof(vert_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(outMesh.indices, idxs_device, indexCount * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    cudaFree(density_data_device);
    cudaFree(active_voxels_device);
    cudaFree(active_index_map);
    cudaFree(verts_device);
    cudaFree(idxs_device);
    cudaDeviceSynchronize();
    return outMesh;
}

} // namespace cuda_dc