export uniform int preNeuronReset(uniform unsigned int* uniform glbSpkCntNeurons) {
    glbSpkCntNeurons[0] = 0;
    return 0;
}

export uniform int updateNeurons(uniform unsigned int* uniform glbSpkCntNeurons, 
                        uniform unsigned int* uniform glbSpkNeurons,
                        uniform float* uniform VNeurons, 
                        uniform float* uniform UNeurons,
                        uniform const float* uniform aNeurons, 
                        uniform const float* uniform bNeurons, 
                        uniform const float* uniform cNeurons, 
                        uniform const float* uniform dNeurons) {
    
    uniform unsigned int spkCount = 0;
    
    // Process neurons
    foreach (id = 0 ... 4) {
        float lV = VNeurons[id];
        float lU = UNeurons[id];
        const float la = aNeurons[id];
        const float lb = bNeurons[id];
        const float lc = cNeurons[id];
        const float ld = dNeurons[id];
        
        float Isyn = 0;
        
        // Calculate membrane potential
        if (lV >= 30.0f) {
            lV = lc;
            lU += ld;
        }
        
        // Update at two times for numerical stability
        lV += 0.5f * (0.04f * lV * lV + 5.0f * lV + 140.0f - lU + Isyn + 10.0f) * 1.0f;
        lV += 0.5f * (0.04f * lV * lV + 5.0f * lV + 140.0f - lU + Isyn + 10.0f) * 1.0f;
        lU += la * (lb * lV - lU) * 1.0f;
        
        // Keep voltage capped at 30.0 to not confuse users with unrealistic values
        if (lV > 30.0f) {
            lV = 30.0f;
        }
        
        // Test for and register a spike
        if (lV >= 29.99f) {
            unsigned int spkIdx = atomic_add_local(&spkCount, 1);
            if (spkIdx < 4) { // Safety check
                glbSpkNeurons[spkIdx] = id;
            }
        }
        
        VNeurons[id] = lV;
        UNeurons[id] = lU;
    }
    
    // Update spike count
    glbSpkCntNeurons[0] = spkCount;
    return 0;
} 