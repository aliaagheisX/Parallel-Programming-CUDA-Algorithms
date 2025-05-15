#include <iostream>
#include <cstring>
using namespace std;

#define N 20
#define M 30


/*
    vec [in]: 1xN
    Mat [in]: NxM
    res [out]: 1xM
*/
void multVecMatrix(float * vec, float *Mat, float *res) {
    for(int j = 0;j < M;j++) {
        res[j] = 0;
        for(int i = 0; i < N; i++) {
            res[j] += Mat[j + i * M] * vec[i];
        }
    }
}


int main() {
    
    float *vec, *Mat, *res;

    vec = (float*)malloc(N * sizeof(float));        // 1xN
    Mat = (float*)malloc(N * M * sizeof(float));    // NxM
    res = (float*)malloc(M * sizeof(float));        // 1xM

    for(int i = 0; i < N; i++) vec[i] = i + 1;

    for(int i = 0; i < N; i++) {
        for(int j = 0;j < M;j++) {
            Mat[j + i * M] = j + i * M + 1;
        }
    }

    
    multVecMatrix(vec, Mat, res);
    
    for(int i = 0; i < M; i++) cout << res[i] << " ";
    
    free(vec);
    free(Mat);
    free(res);

    return 0;
}
