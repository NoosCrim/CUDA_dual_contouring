#include <helper_cuda.h>
#include "cuda_dc/cuda_dc.hpp"
#include <cstdio>
namespace cuda_dc
{
// stage 2 kernel generating active voxels list
__global__ void gen_active_data(uint32_t size, const density_t* __restrict__ density_data, ActiveData *active_data, uint32_t *active_index_map)
{
    extern  __shared__  uint32_t temp[];
    const uint32_t morton = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t laneId = threadIdx.x % 32;
    const uint32_t warpId = threadIdx.x / 32;

    vec_t<uint32_t, 3> voxelID = from_morton(morton);
    bool hasUnder0 = false, hasOver0 = false;
    uint8_t corner_case = 0u;

    density_t original_density = density_data[morton];

    if(morton < size * size * size)
    {
        density_t sample = original_density;
        // we store density at [morton] because we replace it later
        // and we want to put it back later
        hasUnder0 = hasUnder0 || sample <= 0;
        hasOver0 = hasOver0 || sample > 0;
        corner_case = sample <= 0;
        if(voxelID[0] < size - 1)
        {
            sample = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{1,0,0})];
            hasUnder0 = hasUnder0 || sample <= 0;
            hasOver0 = hasOver0 || sample > 0;
            corner_case = corner_case | ((sample <= 0) << 1);
        }
        if(voxelID[1] < size - 1)
        {
            sample = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{0,1,0})];
            hasUnder0 = hasUnder0 || sample <= 0;
            hasOver0 = hasOver0 || sample > 0;
            corner_case = corner_case | ((sample <= 0) << 2);

            if(voxelID[0] < size - 1)
            {
                sample = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{1,1,0})];
                hasUnder0 = hasUnder0 || sample <= 0;
                hasOver0 = hasOver0 || sample > 0;
                corner_case = corner_case | ((sample <= 0) << 3);
            }
        }
        
        if(voxelID[2] + 1 < size)
        {
            sample = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{0,0,1})];
            hasUnder0 = hasUnder0 || sample <= 0;
            hasOver0 = hasOver0 || sample > 0;
            corner_case = corner_case | ((sample <= 0) << 4);

            if(voxelID[0] + 1 < size)
            {
                sample = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{1,0,1})];
                hasUnder0 = hasUnder0 || sample <= 0;
                hasOver0 = hasOver0 || sample > 0;
                corner_case = corner_case | ((sample <= 0) << 5);
            }
            if(voxelID[1] + 1 < size)
            {
                sample = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{0,1,1})];
                hasUnder0 = hasUnder0 || sample <= 0;
                hasOver0 = hasOver0 || sample > 0;
                corner_case = corner_case | ((sample <= 0) << 6);

                if(voxelID[0] + 1 < size)
                {
                    sample = density_data[to_morton(voxelID + vec_t<uint32_t, 3>{1,1,1})];
                    hasUnder0 = hasUnder0 || sample <= 0;
                    hasOver0 = hasOver0 || sample > 0;
                    corner_case = corner_case | ((sample <= 0) << 7);
                }
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
    __syncthreads();
    

    // add amount of active voxels in block to total
    // store write offset for block to shared memory
    // only last thread in first warp has total sum
    if(threadIdx.x == 31)
        temp[32] = atomicAdd(&active_data->active_voxel_n, temp[31]);

    // sync to ensure temp[0] is valid
    __syncthreads();
    
    // add block write offset to writeIdx, giving final writeIdx
    // add 1 as offset because 1st value is amount of active voxels
    // results in final writeIdx
    uint32_t writeIdx = temp[32] + inWarpOffset + (temp[warpId] - activeInWarp);

    if(isActive)
    {
        active_data->data[writeIdx].morton = morton;
        active_data->data[writeIdx].corner_case = corner_case;
        active_index_map[morton] = writeIdx;
    }
}

// stage 4 generating vertices and indices

// QEF solver for 3x3 symmetric matrix with adaptive regularization toward mass point
// Solves (AᵀA + λI)x = Aᵀb + λ*mass_point where λ is adaptively increased if needed
// Works in grid space (coordinates 0 to size)
__device__ vec3_t solve_qef_biased(
    float ata[6],      // Upper triangular of AᵀA: [a00, a01, a02, a11, a12, a22]
    vec3_t atb,        // Aᵀb vector
    vec3_t mass_point, // Bias point (centroid of hermite points) in grid space
    vec3_t voxel_min,  // Voxel bounds min in grid space
    vec3_t voxel_max,  // Voxel bounds max in grid space
    float bias         // Initial regularization strength
)
{
    constexpr uint32_t MAX_ATTEMPTS = 4;
    // Estimate matrix scale (trace of AᵀA) for adaptive regularization
    float trace = ata[0] + ata[3] + ata[5];
    if(trace < 1e-10f)
    {
        // No hermite data at all, return mass point
        return mass_point;
    }
    
    // Scale bias relative to matrix magnitude (ensures regularization is meaningful)
    float scaled_bias = bias * trace;
    
    // Try to solve with increasing regularization if needed
    vec3_t result;
    for(int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt)
    {
        // Add regularization: (AᵀA + λI)
        float a00 = ata[0] + scaled_bias, a01 = ata[1], a02 = ata[2];
        float a11 = ata[3] + scaled_bias, a12 = ata[4];
        float a22 = ata[5] + scaled_bias;
        
        // Bias Aᵀb toward mass_point: Aᵀb + λ*mass_point
        vec3_t b;
        b[0] = atb[0] + scaled_bias * mass_point[0];
        b[1] = atb[1] + scaled_bias * mass_point[1];
        b[2] = atb[2] + scaled_bias * mass_point[2];
        
        // Calculate determinant using cofactor expansion
        float det = a00 * (a11 * a22 - a12 * a12)
                  - a01 * (a01 * a22 - a12 * a02)
                  + a02 * (a01 * a12 - a11 * a02);
        
        // Check if determinant is large enough (relative to matrix scale)
        float det_threshold = 1e-6f * scaled_bias * scaled_bias * scaled_bias;
        if(fabsf(det) < det_threshold)
        {
            // Matrix is ill-conditioned, increase regularization and retry
            scaled_bias *= 10.0f;
            continue;
        }
        
        float inv_det = 1.0f / det;
        
        // Calculate adjugate matrix elements (symmetric), then multiply by inv_det
        float inv00 = (a11 * a22 - a12 * a12) * inv_det;
        float inv01 = (a02 * a12 - a01 * a22) * inv_det;
        float inv02 = (a01 * a12 - a02 * a11) * inv_det;
        float inv11 = (a00 * a22 - a02 * a02) * inv_det;
        float inv12 = (a02 * a01 - a00 * a12) * inv_det;
        float inv22 = (a00 * a11 - a01 * a01) * inv_det;
        
        // Solve: x = (AᵀA + λI)⁻¹ * (Aᵀb + λ*mass_point)
        result[0] = inv00 * b[0] + inv01 * b[1] + inv02 * b[2];
        result[1] = inv01 * b[0] + inv11 * b[1] + inv12 * b[2];
        result[2] = inv02 * b[0] + inv12 * b[1] + inv22 * b[2];
        
        // Check if result is reasonable (within a few voxels of mass point)
        vec3_t diff = result - mass_point;
        float dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
        if(dist_sq > 4.0f) // More than 2 voxels away is suspicious
        {
            // Solution is unstable, increase regularization
            scaled_bias *= 10.0f;
            continue;
        }
        
        // Solution looks good, break out
        break;
    }
    
    // Clamp to voxel bounds (in grid space)
    result[0] = fmaxf(voxel_min[0], fminf(voxel_max[0], result[0]));
    result[1] = fmaxf(voxel_min[1], fminf(voxel_max[1], result[1]));
    result[2] = fmaxf(voxel_min[2], fminf(voxel_max[2], result[2]));
    
    return result;
}

// Helper to add hermite data to QEF accumulators (all in grid space)
inline __device__ void add_hermite_to_qef(
    vec3_t p,          // Hermite point in grid space
    vec3_t n,          // Gradient (will be normalized)
    float ata[6], 
    vec3_t& atb, 
    vec3_t& mass_point, 
    int& count
)
{
    float len = n.len();
    if(len > 1e-6f)
    {
        n = n / len;
        
        // Add to AᵀA (outer product of normal)
        ata[0] += n[0] * n[0];
        ata[1] += n[0] * n[1];
        ata[2] += n[0] * n[2];
        ata[3] += n[1] * n[1];
        ata[4] += n[1] * n[2];
        ata[5] += n[2] * n[2];
        
        // Add to Aᵀb: n * (n · p)
        float d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];
        atb[0] += n[0] * d;
        atb[1] += n[1] * d;
        atb[2] += n[2] * d;
        
        mass_point += p;
        count++;
    }
}

__global__ void gen_mesh(uint32_t size, float voxel_size, const density_t* __restrict__ density_data, const ActiveData* __restrict__ active_data, const uint32_t* __restrict__ active_index_map, vert_t* __restrict__ vertices, uint32_t* __restrict__ indices)
{ 
    // gen vertices using QEF minimization
    // All QEF math is done in GRID SPACE (0 to size), then transformed to world space at the end
    float half_world_size = voxel_size * size * 0.5f;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for(uint32_t active_index = blockDim.x * blockIdx.x + threadIdx.x; 
        active_index < active_data->active_voxel_n; 
        active_index += stride)
    {
        uint32_t morton = active_data->data[active_index].morton;
        uint8_t corner_case = active_data->data[active_index].corner_case;
        uint16_t edge_case = edge_case_map[corner_case];

        vec_t<uint32_t, 3> voxelID = from_morton(morton);
        
        // Voxel bounds in GRID SPACE
        vec3_t voxel_min_grid = vec3_t(voxelID);
        vec3_t voxel_max_grid = voxel_min_grid + vec3_t{1.0f, 1.0f, 1.0f};
        
        // Build QEF from 12 edges of this voxel (in grid space)
        float ata[6] = {0, 0, 0, 0, 0, 0};
        vec3_t atb = {0, 0, 0};
        vec3_t mass_point = {0, 0, 0};
        int hermite_count = 0;
        
        // Edge 0: {0,0,0} -> {1,0,0} - owned by this voxel
        if(edge_case & (1u << 0))
        {
            add_hermite_to_qef(
                active_data->data[active_index].hermite_data[0].point,
                active_data->data[active_index].hermite_data[0].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 1: {0,0,0} -> {0,1,0} - owned by this voxel
        if(edge_case & (1u << 1))
        {
            add_hermite_to_qef(
                active_data->data[active_index].hermite_data[1].point,
                active_data->data[active_index].hermite_data[1].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 2: {0,0,0} -> {0,0,1} - owned by this voxel
        if(edge_case & (1u << 2))
        {
            add_hermite_to_qef(
                active_data->data[active_index].hermite_data[2].point,
                active_data->data[active_index].hermite_data[2].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 3: {1,0,0} -> {1,1,0} - owned by neighbor at +X (their edge 1)
        if((edge_case & (1u << 3)) && voxelID[0] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[to_morton(voxelID + vec_t<uint32_t, 3>{1,0,0})];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[1].point,
                active_data->data[neighbor_idx].hermite_data[1].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 4: {1,0,0} -> {1,0,1} - owned by neighbor at +X (their edge 2)
        if((edge_case & (1u << 4)) && voxelID[0] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[to_morton(voxelID + vec_t<uint32_t, 3>{1,0,0})];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[2].point,
                active_data->data[neighbor_idx].hermite_data[2].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 5: {0,1,0} -> {1,1,0} - owned by neighbor at +Y (their edge 0)
        if((edge_case & (1u << 5)) && voxelID[1] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[to_morton(voxelID + vec_t<uint32_t, 3>{0,1,0})];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[0].point,
                active_data->data[neighbor_idx].hermite_data[0].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 6: {0,1,0} -> {0,1,1} - owned by neighbor at +Y (their edge 2)
        if((edge_case & (1u << 6)) && voxelID[1] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[to_morton(voxelID + vec_t<uint32_t, 3>{0,1,0})];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[2].point,
                active_data->data[neighbor_idx].hermite_data[2].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 7: {0,0,1} -> {1,0,1} - owned by neighbor at +Z (their edge 0)
        if((edge_case & (1u << 7)) && voxelID[2] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[to_morton(voxelID + vec_t<uint32_t, 3>{0,0,1})];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[0].point,
                active_data->data[neighbor_idx].hermite_data[0].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 8: {0,0,1} -> {0,1,1} - owned by neighbor at +Z (their edge 1)
        if((edge_case & (1u << 8)) && voxelID[2] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[to_morton(voxelID + vec_t<uint32_t, 3>{0,0,1})];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[1].point,
                active_data->data[neighbor_idx].hermite_data[1].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 9: {1,1,0} -> {1,1,1} - owned by neighbor at +X+Y (their edge 2)
        if((edge_case & (1u << 9)) && voxelID[0] < size - 1 && voxelID[1] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[to_morton(voxelID + vec_t<uint32_t, 3>{1,1,0})];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[2].point,
                active_data->data[neighbor_idx].hermite_data[2].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 10: {1,0,1} -> {1,1,1} - owned by neighbor at +X+Z (their edge 1)
        if((edge_case & (1u << 10)) && voxelID[0] < size - 1 && voxelID[2] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[to_morton(voxelID + vec_t<uint32_t, 3>{1,0,1})];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[1].point,
                active_data->data[neighbor_idx].hermite_data[1].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 11: {0,1,1} -> {1,1,1} - owned by neighbor at +Y+Z (their edge 0)
        if((edge_case & (1u << 11)) && voxelID[1] < size - 1 && voxelID[2] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[to_morton(voxelID + vec_t<uint32_t, 3>{0,1,1})];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[0].point,
                active_data->data[neighbor_idx].hermite_data[0].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Compute vertex position in grid space
        vec3_t grid_pos;
        if(hermite_count > 0)
        {
            mass_point = mass_point / (float)hermite_count;
            // Use small constant bias for regularization
            float bias = 0.01f;
            grid_pos = solve_qef_biased(ata, atb, mass_point, voxel_min_grid, voxel_max_grid, bias);
        }
        else
        {
            // Fallback: voxel center in grid space
            grid_pos = (voxel_min_grid + voxel_max_grid) * 0.5f;
        }
        
        // Transform from grid space to world space and write vertex
        uint32_t vertex_idx = active_data->data[active_index].vertex_idx;
        if(vertex_idx != UINT32_MAX)
        {
            vertices[vertex_idx].pos = grid_pos * voxel_size - vec3_t{half_world_size, half_world_size, half_world_size};
        }
    }
    
    // generate indices (grid-stride loop)
    for(uint32_t active_index = blockDim.x * blockIdx.x + threadIdx.x; 
        active_index < active_data->active_voxel_n; 
        active_index += stride)
    {
        uint32_t morton = active_data->data[active_index].morton;
        uint8_t corner_case = active_data->data[active_index].corner_case;
        uint16_t edge_case = edge_case_map[corner_case];

        uint64_t index_idx = active_data->data[active_index].edge_idx * 6;
        uint32_t this_vertex_idx = active_data->data[active_index].vertex_idx;

        vec_t<uint32_t, 3> voxelID = from_morton(morton);
        
        // Edge 0: X-direction edge creates quad in YZ plane
        // The 4 voxels sharing this edge are at: {0,0,0}, {0,-1,0}, {0,-1,-1}, {0,0,-1}
        // So we need voxelID[1] > 0 and voxelID[2] > 0
        // Also need all 4 voxels to have valid vertices (all coords < size-1)
        if(voxelID[0] < size - 1 && voxelID[1] > 0 && voxelID[1] < size - 1 && voxelID[2] > 0 && voxelID[2] < size - 1 && (edge_case & (1u << 0)))
        {
            vec_t<uint32_t, 3> v1 = {voxelID[0], voxelID[1] - 1, voxelID[2]};
            vec_t<uint32_t, 3> v2 = {voxelID[0], voxelID[1] - 1, voxelID[2] - 1};
            vec_t<uint32_t, 3> v3 = {voxelID[0], voxelID[1], voxelID[2] - 1};
            
            uint32_t vi1 = active_data->data[active_index_map[to_morton(v1)]].vertex_idx;
            uint32_t vi2 = active_data->data[active_index_map[to_morton(v2)]].vertex_idx;
            uint32_t vi3 = active_data->data[active_index_map[to_morton(v3)]].vertex_idx;
            
            if(density_data[morton] < 0)
            {
                indices[index_idx] = this_vertex_idx;
                indices[index_idx+1] = vi1;
                indices[index_idx+2] = vi2;

                indices[index_idx+3] = this_vertex_idx;
                indices[index_idx+4] = vi2;
                indices[index_idx+5] = vi3;
            }
            else
            {
                indices[index_idx] = this_vertex_idx;
                indices[index_idx+1] = vi2;
                indices[index_idx+2] = vi1;

                indices[index_idx+3] = this_vertex_idx;
                indices[index_idx+4] = vi3;
                indices[index_idx+5] = vi2;
            }
            index_idx += 6;
        }
        // Edge 1: Y-direction edge creates quad in XZ plane
        // The 4 voxels sharing this edge are at: {0,0,0}, {-1,0,0}, {-1,0,-1}, {0,0,-1}
        // So we need voxelID[0] > 0 and voxelID[2] > 0
        // Also need all 4 voxels to have valid vertices (all coords < size-1)
        if(voxelID[0] > 0 && voxelID[0] < size - 1 && voxelID[1] < size - 1 && voxelID[2] > 0 && voxelID[2] < size - 1 && (edge_case & (1u << 1)))
        {
            vec_t<uint32_t, 3> v1 = {voxelID[0], voxelID[1], voxelID[2] - 1};
            vec_t<uint32_t, 3> v2 = {voxelID[0] - 1, voxelID[1], voxelID[2] - 1};
            vec_t<uint32_t, 3> v3 = {voxelID[0] - 1, voxelID[1], voxelID[2]};
            
            uint32_t vi1 = active_data->data[active_index_map[to_morton(v1)]].vertex_idx;
            uint32_t vi2 = active_data->data[active_index_map[to_morton(v2)]].vertex_idx;
            uint32_t vi3 = active_data->data[active_index_map[to_morton(v3)]].vertex_idx;
            
            if(density_data[morton] < 0)
            {
                indices[index_idx] = this_vertex_idx;
                indices[index_idx+1] = vi1;
                indices[index_idx+2] = vi2;

                indices[index_idx+3] = this_vertex_idx;
                indices[index_idx+4] = vi2;
                indices[index_idx+5] = vi3;
            }
            else
            {
                indices[index_idx] = this_vertex_idx;
                indices[index_idx+1] = vi2;
                indices[index_idx+2] = vi1;

                indices[index_idx+3] = this_vertex_idx;
                indices[index_idx+4] = vi3;
                indices[index_idx+5] = vi2;
            }
            index_idx += 6;
        }
        // Edge 2: Z-direction edge creates quad in XY plane
        // The 4 voxels sharing this edge are at: {0,0,0}, {-1,0,0}, {-1,-1,0}, {0,-1,0}
        // So we need voxelID[0] > 0 and voxelID[1] > 0
        // Also need all 4 voxels to have valid vertices (all coords < size-1)
        if(voxelID[0] > 0 && voxelID[0] < size - 1 && voxelID[1] > 0 && voxelID[1] < size - 1 && voxelID[2] < size - 1 && (edge_case & (1u << 2)))
        {
            vec_t<uint32_t, 3> v1 = {voxelID[0] - 1, voxelID[1], voxelID[2]};
            vec_t<uint32_t, 3> v2 = {voxelID[0] - 1, voxelID[1] - 1, voxelID[2]};
            vec_t<uint32_t, 3> v3 = {voxelID[0], voxelID[1] - 1, voxelID[2]};
            
            uint32_t vi1 = active_data->data[active_index_map[to_morton(v1)]].vertex_idx;
            uint32_t vi2 = active_data->data[active_index_map[to_morton(v2)]].vertex_idx;
            uint32_t vi3 = active_data->data[active_index_map[to_morton(v3)]].vertex_idx;
            
            if(density_data[morton] < 0)
            {
                indices[index_idx] = this_vertex_idx;
                indices[index_idx+1] = vi1;
                indices[index_idx+2] = vi2;

                indices[index_idx+3] = this_vertex_idx;
                indices[index_idx+4] = vi2;
                indices[index_idx+5] = vi3;
            }
            else
            {
                indices[index_idx] = this_vertex_idx;
                indices[index_idx+1] = vi2;
                indices[index_idx+2] = vi1;

                indices[index_idx+3] = this_vertex_idx;
                indices[index_idx+4] = vi3;
                indices[index_idx+5] = vi2;
            }
            index_idx += 6;
        }
    }
    
    
}

} // namespace cuda_dc