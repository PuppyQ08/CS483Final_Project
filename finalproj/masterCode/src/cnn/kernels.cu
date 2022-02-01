
#include <cuda.h>
#include <cublas_v2.h>

const static float dt = 1.0E-01f;

// Utility CUDA kernel functions
__device__ float activation_function(float x){
    return 1 / (1 + exp(-x));
}

__global__ void apply_activation_function(float *input, float *output, const int N){
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx < N) {
        output[tx] = activation_function(input[tx]);
    }
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N){
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx < N) {
        err[tx] = ((Y == tx ? 1.0f : 0.0f) - output[tx]);
    }
}

__global__ void apply_grad(float *output, float *grad, const int N){
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx < N) {
        output[tx] += dt * grad[tx];
    }
}

// Forward propagation kernels
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*24*24*5*5; //86,400
    const int tix = threadIdx.x;
    __shared__ float weight_s[6][5][5];
    if (tix < 6*5*5){
        weight_s[tix/5/5][tix%25/5][tix%5] = weight[tix/5/5][tix%25/5][tix%5];
    }
    __syncthreads();

    if (tx < N) { 
            atomicAdd(&preact[tx/5/5%6][tx/5/5/6%24][tx/5/5/6/24%24],
                weight_s[tx/5/5%6][tx%5][tx/5%5] * input[tx/5/5/6%24 + tx%5][tx/5/5/6/24%24 + tx/5%5]);
    }
}

__global__ void fp_bias_c1(float preact[6][24][24], float bias[6]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*24*24;
    if(tx < N){
        preact[tx/24/24][tx/24%24][tx%24] += bias[tx/24/24];
    }
}

__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*6*6;
    if (tx < N) {
        for (int i = 0; i < 4*4; ++i) {
            atomicAdd(&preact[tx/6/6][tx/6%6][tx%6],
                weight[0][i/4][i%4] * input[tx/6/6][tx/6%6*4 + i/4][tx%6*4 + i%4]);
        }
    }  
}

__global__ void fp_bias_s1(float preact[6][6][6], float bias[1]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*6*6;
    if (tx < N){
        preact[tx/6/6][tx/6%6][tx%6] += bias[0];
    }
}

__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6]){

    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 10*6*6*6;
    if (tx < N) {
        atomicAdd(&preact[tx%10], 
                  weight[tx%10][tx/10%6][tx/10/6%6][tx/10/6/6%6] * input[tx/10%6][tx/10/6%6][tx/10/6/6%6]);
                  
    }
}

__global__ void fp_bias_f(float preact[10], float bias[10]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tx < 10){
        preact[tx] += bias[tx];
    }
}


// Back propagation kernels
__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 10*6*6*6;
    if (tx < N) {
        d_weight[tx%10][tx/10%6][tx/10/6%6][tx/10/6/6%6] = 
            (d_preact[tx%10] * p_output[tx/10%6][tx/10/6%6][tx/10/6/6%6]);
    }
}

__global__ void bp_bias_f(float bias[10], float d_preact[10]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tx < 10){
        bias[tx] += dt * d_preact[tx];
    }
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]){

    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*6*6*10;
    
    if(tx < N){
        const int i1 = tx % 10;
        const int i2 = (tx / 10) % 6;
        const int i3 = (tx / 10/ 6) % 6;
        const int i4 = ((tx / 10 / 6 / 6) % 6);
        
        atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
    
    }

}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*6*6;

    if (tx < N) {
        const int i1 = tx%6;
        const int i2 = tx/6%6;
        const int i3 = tx/6/6%6;
        const float output = activation_function(preact[i1][i2][i3]);
        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * output * (1-output);
    }
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]){
    
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 4*4*6*6*6;
    if (tx < N) {
        atomicAdd(&d_weight[0][tx%4][tx/4%4],
                   d_preact[tx/4/4%6][tx/4/4/6%6][tx/4/4/6/6%6] 
                 * p_output[tx/4/4%6][tx/4/4/6%6 * 4 + tx%4][tx/4/4/6/6%6 * 4 + tx/4%4]);
    }

}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*6*6;
    const float d = pow(6.0f, 3.0f);

    if (tx < N) {
        atomicAdd(&bias[0], dt * d_preact[tx%6][tx/6%6][tx/6/6%6] / d);
    }
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*24*24;
    if(tx < N){
        d_output[tx/4/4%6][tx/4/4/6%6 * 4 + tx%4][tx/4/4/6/6%6 * 4 + tx/4%4] =
            n_weight[0][tx%4][tx/4%4] * nd_preact[tx/4/4%6][tx/4/4/6%6][tx/4/4/6/6%6];
    }
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]){

    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*24*24;
    if(tx < N){
        const float output = activation_function(preact[tx/24/24][tx/24%24][tx%24]);
        d_preact[tx/24/24][tx/24%24][tx%24] = d_output[tx/24/24][tx/24%24][tx%24] * output * (1 - output);
    }

}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*5*5*24*24;//86,400
    const float d = pow(24.0f, 2.0f);

    if (tx < N) {
        atomicAdd(&d_weight[tx%6][tx/6%5][tx/6/5%5],
                  d_preact[tx%6][tx/6/5/5%24][tx/6/5/5/24%24] 
                  * p_output[tx/6/5/5%24 + tx/6%5][tx/6/5/5/24%24 + tx/6/5%5] / d);
    }
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]){
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 6*24*24;
    const float d = pow(24.0f, 2.0f);
    if (tx < N) {
        atomicAdd(&bias[tx%6], dt * d_preact[tx%6][tx/6/24][tx/6%24] / d);
    }
}
