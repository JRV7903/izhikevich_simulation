// C++ includes
#include <algorithm>
#include <fstream>

// ------------------------------------------------------------------------
// Helper macro for error-checking CUDA calls
#define CHECK_CUDA_ERRORS(call) {\
    cudaError_t error = call;\
    if (error != cudaSuccess) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": cuda error " + std::to_string(error) + ": " + cudaGetErrorString(error));\
    }\
}

namespace
{
unsigned int* glbSpkCntNeurons;
unsigned int* d_glbSpkCntNeurons;
unsigned int* glbSpkNeurons;
unsigned int* d_glbSpkNeurons;

float* VNeurons;
float* d_VNeurons;
float* UNeurons;
float* d_UNeurons;
float* aNeurons;
float* d_aNeurons;
float* bNeurons;
float* d_bNeurons;
float* cNeurons;
float* d_cNeurons;
float* dNeurons;
float* d_dNeurons;

__global__ void preNeuronResetKernel(unsigned int* d_glbSpkCntNeurons) {
    unsigned int id = 32 * blockIdx.x + threadIdx.x;
    if(id == 0) {
        d_glbSpkCntNeurons[0] = 0;
    }
}

__global__ void updateNeuronsKernel(unsigned int *d_glbSpkCntNeurons, unsigned int *d_glbSpkNeurons, float *d_VNeurons, float *d_UNeurons, 
                                    const float *d_aNeurons, const float *d_bNeurons, const float *d_cNeurons, const float *d_dNeurons)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[32];
    __shared__ unsigned int shPosSpk;
    __shared__ unsigned int shSpkCount;
    if (threadIdx.x == 0); {
        shSpkCount = 0;
    }
    
    __syncthreads();
    // Neurons
    if(id < 32) {
        if(id < 4) {
            float lV = d_VNeurons[id];
            float lU = d_UNeurons[id];
            const float la = d_aNeurons[id];
            const float lb = d_bNeurons[id];
            const float lc = d_cNeurons[id];
            const float ld = d_dNeurons[id];
            
            float Isyn = 0;
            
            // calculate membrane potential
            if (lV >= 30.0f){
                lV=lc;
                lU+=ld;
            } 
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn+(1.00000000000000000e+01f))*1.0f; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn+(1.00000000000000000e+01f))*1.0f;
            lU+=la*(lb*lV-lU)*1.0f;
            
            if(lV > 30.0){   //keep this to not confuse users with unrealistiv voltage values
              lV=30.0;
            }
            
            // test for and register a true spike
            if (lV >= 29.99f) {
                const unsigned int spkIdx = atomicAdd((unsigned int *) &shSpkCount, 1);
                shSpk[spkIdx] = id;
            }
            d_VNeurons[id] = lV;
            d_UNeurons[id] = lU;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd((unsigned int*)&d_glbSpkCntNeurons[0], shSpkCount);
            }
        }
        __syncthreads();
        if (threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            d_glbSpkNeurons[shPosSpk + threadIdx.x] = n;
        }
    }
    
}
}

int main()
{
    CHECK_CUDA_ERRORS(cudaSetDevice(0));
    
    // Allocate memory
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntNeurons, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntNeurons, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkNeurons, 4 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkNeurons, 4 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&VNeurons, 4 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_VNeurons, 4 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&UNeurons, 4 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_UNeurons, 4 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&aNeurons, 4 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_aNeurons, 4 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&bNeurons, 4 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_bNeurons, 4 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&cNeurons, 4 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_cNeurons, 4 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dNeurons, 4 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dNeurons, 4 * sizeof(float)));
    
    // Initialise
    glbSpkCntNeurons[0] = 0;
    std::fill_n(VNeurons, 4, -6.50000000000000000e+01f);
    std::fill_n(UNeurons, 4, -2.00000000000000000e+01f);
    
    // Regular
    aNeurons[0] = 0.02; bNeurons[0] = 0.2;  cNeurons[0] = -65.0;    dNeurons[0] = 8.0;

    // Fast
    aNeurons[1] = 0.1;  bNeurons[1] = 0.2;  cNeurons[1] = -65.0;    dNeurons[1] = 2.0;

    // Chattering
    aNeurons[2] = 0.02; bNeurons[2] = 0.2;  cNeurons[2] = -50.0;    dNeurons[2] = 2.0;

    // Bursting
    aNeurons[3] = 0.02; bNeurons[3] = 0.2;  cNeurons[3] = -55.0;    dNeurons[3] = 4.0;
    
    // Copy to device
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntNeurons, glbSpkCntNeurons, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VNeurons, VNeurons, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UNeurons, UNeurons, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aNeurons, aNeurons, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_bNeurons, bNeurons, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_cNeurons, cNeurons, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_dNeurons, dNeurons, 4 * sizeof(float), cudaMemcpyHostToDevice));
    
    std::ofstream spikes("spikes.csv");
    std::ofstream voltages("voltages.csv");
    
    for(unsigned int i = 0; i < 200; i++) {
        // Launch kernels
        const dim3 threads(32, 1);
        const dim3 grid(1, 1);
        preNeuronResetKernel<<<grid, threads>>>(d_glbSpkCntNeurons);
        updateNeuronsKernel<<<grid, threads>>>(d_glbSpkCntNeurons, d_glbSpkNeurons, d_VNeurons, d_UNeurons, 
                                            d_aNeurons, d_bNeurons, d_cNeurons, d_dNeurons);
        
        // Copy voltages back from device and write to file
        CHECK_CUDA_ERRORS(cudaMemcpy(VNeurons, d_VNeurons, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        voltages << i << "," << VNeurons[0] << "," << VNeurons[1] << "," << VNeurons[2] << "," << VNeurons[3] << std::endl;
        
        // Copy spikes back from device
        CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntNeurons, d_glbSpkCntNeurons, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkNeurons, d_glbSpkNeurons, 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        for(unsigned int s = 0; s < glbSpkCntNeurons[0]; s++) {
            spikes << i << "," << glbSpkNeurons[s] << std::endl;
        }

    }
    return EXIT_SUCCESS;
}