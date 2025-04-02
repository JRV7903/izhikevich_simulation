#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cstdlib>

#include "izhikevich_ispc.h"

int main() {
    unsigned int* glbSpkCntNeurons = new unsigned int[1];
    unsigned int* glbSpkNeurons = new unsigned int[4];
    float* VNeurons = new float[4];
    float* UNeurons = new float[4];
    float* aNeurons = new float[4];
    float* bNeurons = new float[4];
    float* cNeurons = new float[4];
    float* dNeurons = new float[4];
    
    glbSpkCntNeurons[0] = 0;
    std::fill_n(VNeurons, 4, -65.0f);
    std::fill_n(UNeurons, 4, -20.0f);
    
    aNeurons[0] = 0.02f; bNeurons[0] = 0.2f;  cNeurons[0] = -65.0f; dNeurons[0] = 8.0f;
    aNeurons[1] = 0.1f;  bNeurons[1] = 0.2f;  cNeurons[1] = -65.0f; dNeurons[1] = 2.0f;
    aNeurons[2] = 0.02f; bNeurons[2] = 0.2f;  cNeurons[2] = -50.0f; dNeurons[2] = 2.0f;
    aNeurons[3] = 0.02f; bNeurons[3] = 0.2f;  cNeurons[3] = -55.0f; dNeurons[3] = 4.0f;
    
    std::ofstream spikes("spikes.csv");
    std::ofstream voltages("voltages.csv");
    
    for(unsigned int i = 0; i < 200; i++) {
        ispc::preNeuronReset(glbSpkCntNeurons);
        ispc::updateNeurons(glbSpkCntNeurons, glbSpkNeurons, 
                            VNeurons, UNeurons, 
                            aNeurons, bNeurons, cNeurons, dNeurons);
        
        voltages << i << "," << VNeurons[0] << "," << VNeurons[1] << "," 
                 << VNeurons[2] << "," << VNeurons[3] << std::endl;
        
        for(unsigned int s = 0; s < glbSpkCntNeurons[0]; s++) {
            spikes << i << "," << glbSpkNeurons[s] << std::endl;
        }
    }
    
    delete[] glbSpkCntNeurons;
    delete[] glbSpkNeurons;
    delete[] VNeurons;
    delete[] UNeurons;
    delete[] aNeurons;
    delete[] bNeurons;
    delete[] cNeurons;
    delete[] dNeurons;
    
    return EXIT_SUCCESS;
} 