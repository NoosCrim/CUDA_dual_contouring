#include <helper_cuda.h>
#include "cuda_dc/cuda_dc.hpp"
#include <cstdio>
namespace cuda_dc
{

// stage 4 generating vertices and indices

// QEF solver for 3x3 symmetric matrix with adaptive regularization toward mass point.
// Solves (A'A + lambda*I)x = A'b + lambda*mass_point, increasing lambda if needed.
// Works in grid space (coordinates 0 to size).
__device__ vec3_t solve_qef_biased(
    float ata[6],      // Upper triangular of A'A: [a00, a01, a02, a11, a12, a22]
    vec3_t atb,        // A'b vector
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
        // Add regularization: (A'A + lambda*I)
        float a00 = ata[0] + scaled_bias, a01 = ata[1], a02 = ata[2];
        float a11 = ata[3] + scaled_bias, a12 = ata[4];
        float a22 = ata[5] + scaled_bias;
        
        // Bias A'b toward mass_point: A'b + lambda*mass_point
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
        
        // Solve: x = (A'A + lambda*I)^-1 * (A'b + lambda*mass_point)
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
        
        // Add to A'A (outer product of normal)
        ata[0] += n[0] * n[0];
        ata[1] += n[0] * n[1];
        ata[2] += n[0] * n[2];
        ata[3] += n[1] * n[1];
        ata[4] += n[1] * n[2];
        ata[5] += n[2] * n[2];
        
        // Add to A'b: n * dot(n, p)
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
        
        // Precompute neighbor Morton codes using fast increments
        uint32_t m_x = morton_inc_x(morton);      // +X
        uint32_t m_y = morton_inc_y(morton);      // +Y
        uint32_t m_z = morton_inc_z(morton);      // +Z
        uint32_t m_xy = morton_inc_y(m_x);        // +X+Y
        uint32_t m_xz = morton_inc_z(m_x);        // +X+Z
        uint32_t m_yz = morton_inc_z(m_y);        // +Y+Z
        
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
            uint32_t neighbor_idx = active_index_map[m_x];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[1].point,
                active_data->data[neighbor_idx].hermite_data[1].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 4: {1,0,0} -> {1,0,1} - owned by neighbor at +X (their edge 2)
        if((edge_case & (1u << 4)) && voxelID[0] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[m_x];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[2].point,
                active_data->data[neighbor_idx].hermite_data[2].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 5: {0,1,0} -> {1,1,0} - owned by neighbor at +Y (their edge 0)
        if((edge_case & (1u << 5)) && voxelID[1] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[m_y];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[0].point,
                active_data->data[neighbor_idx].hermite_data[0].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 6: {0,1,0} -> {0,1,1} - owned by neighbor at +Y (their edge 2)
        if((edge_case & (1u << 6)) && voxelID[1] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[m_y];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[2].point,
                active_data->data[neighbor_idx].hermite_data[2].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 7: {0,0,1} -> {1,0,1} - owned by neighbor at +Z (their edge 0)
        if((edge_case & (1u << 7)) && voxelID[2] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[m_z];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[0].point,
                active_data->data[neighbor_idx].hermite_data[0].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 8: {0,0,1} -> {0,1,1} - owned by neighbor at +Z (their edge 1)
        if((edge_case & (1u << 8)) && voxelID[2] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[m_z];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[1].point,
                active_data->data[neighbor_idx].hermite_data[1].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 9: {1,1,0} -> {1,1,1} - owned by neighbor at +X+Y (their edge 2)
        if((edge_case & (1u << 9)) && voxelID[0] < size - 1 && voxelID[1] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[m_xy];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[2].point,
                active_data->data[neighbor_idx].hermite_data[2].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 10: {1,0,1} -> {1,1,1} - owned by neighbor at +X+Z (their edge 1)
        if((edge_case & (1u << 10)) && voxelID[0] < size - 1 && voxelID[2] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[m_xz];
            add_hermite_to_qef(
                active_data->data[neighbor_idx].hermite_data[1].point,
                active_data->data[neighbor_idx].hermite_data[1].gradient,
                ata, atb, mass_point, hermite_count
            );
        }
        
        // Edge 11: {0,1,1} -> {1,1,1} - owned by neighbor at +Y+Z (their edge 0)
        if((edge_case & (1u << 11)) && voxelID[1] < size - 1 && voxelID[2] < size - 1)
        {
            uint32_t neighbor_idx = active_index_map[m_yz];
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
        
        // Precompute neighbor Morton codes for -1 offsets
        uint32_t m_ny = morton_dec_y(morton);     // -Y
        uint32_t m_nz = morton_dec_z(morton);     // -Z
        uint32_t m_nx = morton_dec_x(morton);     // -X
        uint32_t m_nynz = morton_dec_z(m_ny);     // -Y-Z
        uint32_t m_nxnz = morton_dec_z(m_nx);     // -X-Z
        uint32_t m_nxny = morton_dec_y(m_nx);     // -X-Y
        
        // Edge 0: X-direction edge creates quad in YZ plane
        // The 4 voxels sharing this edge are at: {0,0,0}, {0,-1,0}, {0,-1,-1}, {0,0,-1}
        // So we need voxelID[1] > 0 and voxelID[2] > 0
        // Also need all 4 voxels to have valid vertices (all coords < size-1)
        if(voxelID[0] < size - 1 && voxelID[1] > 0 && voxelID[1] < size - 1 && voxelID[2] > 0 && voxelID[2] < size - 1 && (edge_case & (1u << 0)))
        {
            uint32_t vi1 = active_data->data[active_index_map[m_ny]].vertex_idx;      // {0,-1,0}
            uint32_t vi2 = active_data->data[active_index_map[m_nynz]].vertex_idx;    // {0,-1,-1}
            uint32_t vi3 = active_data->data[active_index_map[m_nz]].vertex_idx;      // {0,0,-1}
            
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
            uint32_t vi1 = active_data->data[active_index_map[m_nz]].vertex_idx;      // {0,0,-1}
            uint32_t vi2 = active_data->data[active_index_map[m_nxnz]].vertex_idx;    // {-1,0,-1}
            uint32_t vi3 = active_data->data[active_index_map[m_nx]].vertex_idx;      // {-1,0,0}
            
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
            uint32_t vi1 = active_data->data[active_index_map[m_nx]].vertex_idx;      // {-1,0,0}
            uint32_t vi2 = active_data->data[active_index_map[m_nxny]].vertex_idx;    // {-1,-1,0}
            uint32_t vi3 = active_data->data[active_index_map[m_ny]].vertex_idx;      // {0,-1,0}
            
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

// Thread-coarsened version: each thread processes a 2x2x2 block of voxels.
// This reduces memory traffic by reusing density values at shared corners.
// A 2x2x2 block needs 3x3x3 = 27 density values (instead of 8*8 = 64 without reuse).
__global__ void gen_active_data(uint32_t size, const density_t* __restrict__ density_data, ActiveData *active_data, uint32_t *active_index_map)
{
    extern __shared__ uint32_t temp[];
    
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t laneId = threadIdx.x % 32;
    const uint32_t warpId = threadIdx.x / 32;
    
    // Each thread processes a 2x2x2 block of voxels
    // Block base position in grid coordinates (multiply by 2)
    const uint32_t half_size = size / 2;
    const uint32_t half_size_cubed = half_size * half_size * half_size;
    
    // Results for up to 8 voxels per thread
    uint8_t corner_cases[8] = {0};
    uint32_t mortons[8];
    uint8_t active_mask = 0;  // Bitmask of which voxels are active
    int active_count = 0;
    
    if(tid < half_size_cubed)
    {
        // Get block base position from Morton code at half resolution
        vec_t<uint32_t, 3> blockBase = from_morton(tid);
        blockBase[0] *= 2;
        blockBase[1] *= 2;
        blockBase[2] *= 2;
        
        // Check if entire 2x2x2 block is interior (no boundary voxels)
        bool blockInterior = (blockBase[0] + 2 <= size - 1) && 
                             (blockBase[1] + 2 <= size - 1) && 
                             (blockBase[2] + 2 <= size - 1);
        
        // Load 3x3x3 = 27 density values for the block (if interior)
        // Layout: d[z][y][x] where each dimension is 0,1,2
        density_t d[3][3][3];
        bool signs[3][3][3];
        
        if(blockInterior)
        {
            // Compute base Morton code for the block
            uint32_t m_base = to_morton(blockBase);
            
            // Load all 27 densities using Morton increments.
            // We build Morton codes for each row start (x=0), then walk along x.
            // row_start[z][y] holds the Morton code for position (0, y, z) relative to blockBase.
            
            uint32_t row_start[3][3];
            
            // Build row starts for z=0 plane
            row_start[0][0] = m_base;
            row_start[0][1] = morton_inc_y(m_base);
            row_start[0][2] = morton_inc_y(row_start[0][1]);
            
            // Build row starts for z=1 and z=2 planes
            #pragma unroll
            for(int y = 0; y < 3; y++)
            {
                row_start[1][y] = morton_inc_z(row_start[0][y]);
                row_start[2][y] = morton_inc_z(row_start[1][y]);
            }
            
            // Load densities by walking along x for each row
            #pragma unroll
            for(int z = 0; z < 3; z++)
            {
                #pragma unroll
                for(int y = 0; y < 3; y++)
                {
                    uint32_t m = row_start[z][y];
                    d[z][y][0] = density_data[m];
                    m = morton_inc_x(m);
                    d[z][y][1] = density_data[m];
                    d[z][y][2] = density_data[morton_inc_x(m)];
                }
            }
            
            // Compute signs
            #pragma unroll
            for(int z = 0; z < 3; z++)
                #pragma unroll
                for(int y = 0; y < 3; y++)
                    #pragma unroll
                    for(int x = 0; x < 3; x++)
                        signs[z][y][x] = d[z][y][x] <= 0;
            
            // Process 8 voxels in the 2x2x2 block
            // Voxel (0,0,0) at blockBase
            #pragma unroll
            for(int vz = 0; vz < 2; vz++)
            {
                #pragma unroll
                for(int vy = 0; vy < 2; vy++)
                {
                    #pragma unroll
                    for(int vx = 0; vx < 2; vx++)
                    {
                        int vidx = vx + vy * 2 + vz * 4;
                        vec_t<uint32_t, 3> voxelPos = blockBase + vec_t<uint32_t, 3>{(uint32_t)vx, (uint32_t)vy, (uint32_t)vz};
                        mortons[vidx] = to_morton(voxelPos);
                        
                        // Build corner case from 8 corners of this voxel
                        uint8_t cc = 0;
                        cc |= signs[vz  ][vy  ][vx  ] << 0;  // corner 0
                        cc |= signs[vz  ][vy  ][vx+1] << 1;  // corner 1
                        cc |= signs[vz  ][vy+1][vx  ] << 2;  // corner 2
                        cc |= signs[vz  ][vy+1][vx+1] << 3;  // corner 3
                        cc |= signs[vz+1][vy  ][vx  ] << 4;  // corner 4
                        cc |= signs[vz+1][vy  ][vx+1] << 5;  // corner 5
                        cc |= signs[vz+1][vy+1][vx  ] << 6;  // corner 6
                        cc |= signs[vz+1][vy+1][vx+1] << 7;  // corner 7
                        
                        corner_cases[vidx] = cc;
                        
                        // Active if not all same sign
                        bool isActive = (cc != 0) && (cc != 255);
                        if(isActive)
                        {
                            active_mask |= (1u << vidx);
                            active_count++;
                        }
                    }
                }
            }
        }
        else
        {
            // Handle boundary blocks - fall back to per-voxel processing
            #pragma unroll
            for(int vz = 0; vz < 2; vz++)
            {
                #pragma unroll
                for(int vy = 0; vy < 2; vy++)
                {
                    #pragma unroll
                    for(int vx = 0; vx < 2; vx++)
                    {
                        int vidx = vx + vy * 2 + vz * 4;
                        vec_t<uint32_t, 3> voxelPos = blockBase + vec_t<uint32_t, 3>{(uint32_t)vx, (uint32_t)vy, (uint32_t)vz};
                        
                        // Skip if outside grid
                        if(voxelPos[0] >= size || voxelPos[1] >= size || voxelPos[2] >= size)
                            continue;
                        
                        uint32_t morton = to_morton(voxelPos);
                        mortons[vidx] = morton;
                        
                        bool atBoundary = (voxelPos[0] >= size - 1) || (voxelPos[1] >= size - 1) || (voxelPos[2] >= size - 1);
                        
                        density_t s0 = density_data[morton];
                        bool sign0 = s0 <= 0;
                        uint8_t cc = sign0;
                        
                        if(!atBoundary)
                        {
                            // Can load all 8 corners
                            uint32_t m100 = morton_inc_x(morton);
                            uint32_t m010 = morton_inc_y(morton);
                            uint32_t m110 = morton_inc_x(m010);
                            uint32_t m001 = morton_inc_z(morton);
                            uint32_t m101 = morton_inc_x(m001);
                            uint32_t m011 = morton_inc_y(m001);
                            uint32_t m111 = morton_inc_x(m011);
                            
                            cc |= (density_data[m100] <= 0) << 1;
                            cc |= (density_data[m010] <= 0) << 2;
                            cc |= (density_data[m110] <= 0) << 3;
                            cc |= (density_data[m001] <= 0) << 4;
                            cc |= (density_data[m101] <= 0) << 5;
                            cc |= (density_data[m011] <= 0) << 6;
                            cc |= (density_data[m111] <= 0) << 7;
                            
                            corner_cases[vidx] = cc;
                            bool isActive = (cc != 0) && (cc != 255);
                            if(isActive)
                            {
                                active_mask |= (1u << vidx);
                                active_count++;
                            }
                        }
                        else
                        {
                            // Boundary voxel - check each neighbor
                            bool hasUnder0 = sign0, hasOver0 = !sign0;
                            
                            if(voxelPos[0] < size - 1)
                            {
                                bool s = density_data[morton_inc_x(morton)] <= 0;
                                hasUnder0 |= s; hasOver0 |= !s;
                                cc |= (s << 1);
                            }
                            if(voxelPos[1] < size - 1)
                            {
                                uint32_t m010 = morton_inc_y(morton);
                                bool s = density_data[m010] <= 0;
                                hasUnder0 |= s; hasOver0 |= !s;
                                cc |= (s << 2);
                                
                                if(voxelPos[0] < size - 1)
                                {
                                    s = density_data[morton_inc_x(m010)] <= 0;
                                    hasUnder0 |= s; hasOver0 |= !s;
                                    cc |= (s << 3);
                                }
                            }
                            if(voxelPos[2] < size - 1)
                            {
                                uint32_t m001 = morton_inc_z(morton);
                                bool s = density_data[m001] <= 0;
                                hasUnder0 |= s; hasOver0 |= !s;
                                cc |= (s << 4);
                                
                                if(voxelPos[0] < size - 1)
                                {
                                    s = density_data[morton_inc_x(m001)] <= 0;
                                    hasUnder0 |= s; hasOver0 |= !s;
                                    cc |= (s << 5);
                                }
                                if(voxelPos[1] < size - 1)
                                {
                                    uint32_t m011 = morton_inc_y(m001);
                                    s = density_data[m011] <= 0;
                                    hasUnder0 |= s; hasOver0 |= !s;
                                    cc |= (s << 6);
                                    
                                    if(voxelPos[0] < size - 1)
                                    {
                                        s = density_data[morton_inc_x(m011)] <= 0;
                                        hasUnder0 |= s; hasOver0 |= !s;
                                        cc |= (s << 7);
                                    }
                                }
                            }
                            
                            corner_cases[vidx] = cc;
                            bool isActive = hasUnder0 && hasOver0;
                            if(isActive)
                            {
                                active_mask |= (1u << vidx);
                                active_count++;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Warp-level reduction: sum active counts across warp
    uint32_t warp_total = active_count;
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
        warp_total += __shfl_xor_sync(0xFFFFFFFF, warp_total, offset);
    
    // Exclusive prefix sum within warp
    uint32_t warp_offset = active_count;
    #pragma unroll
    for(int offset = 1; offset < 32; offset <<= 1)
    {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, warp_offset, offset);
        if(laneId >= offset) warp_offset += n;
    }
    warp_offset -= active_count;  // Convert to exclusive
    
    // Store warp total in shared memory
    if(laneId == 31) temp[warpId] = warp_total;
    __syncthreads();
    
    // First warp computes prefix sum of warp totals
    if(warpId == 0)
    {
        uint32_t warpPrefixSum = (laneId < blockDim.x/32) ? temp[laneId] : 0;
        #pragma unroll
        for(int offset = 1; offset < 32; offset <<= 1)
        {
            uint32_t n = __shfl_up_sync(0xFFFFFFFF, warpPrefixSum, offset);
            if(laneId >= offset) warpPrefixSum += n;
        }
        temp[laneId] = warpPrefixSum;
    }
    __syncthreads();
    
    // Get block offset from atomic add
    if(threadIdx.x == blockDim.x - 1)
        temp[32] = atomicAdd(&active_data->active_voxel_n, temp[blockDim.x/32 - 1]);
    __syncthreads();
    
    // Compute final write position
    uint32_t block_offset = temp[32];
    uint32_t warp_base = (warpId > 0) ? temp[warpId - 1] : 0;
    uint32_t writeIdx = block_offset + warp_base + warp_offset;
    
    // Write active voxels
    for(int v = 0; v < 8; v++)
    {
        if(active_mask & (1u << v))
        {
            active_data->data[writeIdx].morton = mortons[v];
            active_data->data[writeIdx].corner_case = corner_cases[v];
            active_index_map[mortons[v]] = writeIdx;
            writeIdx++;
        }
    }
}

} // namespace cuda_dc