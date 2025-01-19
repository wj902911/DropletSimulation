#ifndef CUDACLASS_H
#define CUDACLASS_H

class CudaClass {
public:
    CudaClass(int size);
    ~CudaClass();

    void addVectors();

private:
    int size;
    int* h_a, * h_b, * h_c; // Host vectors
    int* d_a, * d_b, * d_c; // Device vectors

    void allocateHostMemory();
    void allocateDeviceMemory();
    void freeMemory();
};

#endif // CUDACLASS_H