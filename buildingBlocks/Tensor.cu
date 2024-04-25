#include "Tensor.cuh"
#include "matrixOperations.cuh"
#include <iostream>
#include <string>
#include <sstream> 
#include <vector>
#include <numeric>


using namespace std;
using namespace MatrixOperations;

// Full Tensor Constructor
Tensor::Tensor(const vector<float>& vals, const vector<int>& dims, string dev) 
    : dimensions(dims), totalVals(vals.size()), nDimensions(dims.size()), device(dev), values(vals) {
    if("cuda" == dev){
        int size2 = totalVals*sizeof(float);
        cudaMalloc((void**)&valuesCuda, size2);
        cudaMemcpy(valuesCuda, vals.data(), size2, cudaMemcpyHostToDevice);
    }
}

Tensor::Tensor(float* c_device, const vector<int>& dims, string dev)
    : dimensions(dims), totalVals(accumulate(dims.begin(), dims.end(), 1, multiplies<int>())), nDimensions(dims.size()), device(dev) {
    values = vector<float>(totalVals);
    cudaError_t status = cudaMalloc((void**)&valuesCuda, totalVals * sizeof(float));
    if (status != cudaSuccess) {
        throw runtime_error("Failed to allocate device memory: " + string(cudaGetErrorString(status)));
    }

    status = cudaMemcpyAsync(valuesCuda, c_device, totalVals * sizeof(float), cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
        cudaFree(valuesCuda);
        throw runtime_error("Failed to copy data to device memory: " + string(cudaGetErrorString(status)));
    }

    cudaDeviceSynchronize();
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cudaFree(valuesCuda); 
        throw runtime_error("CUDA error after copying data: " + string(cudaGetErrorString(status)));
    }
}

Tensor::Tensor() {

}

// Copy Constructor
Tensor::Tensor(const Tensor& other) 
    : totalVals(other.totalVals), nDimensions(other.nDimensions), device(other.device), dimensions(other.dimensions), values(other.values) {
    if (device.compare("cuda") == 0) {
        cudaMalloc(&valuesCuda, totalVals * sizeof(float));
        cudaMemcpy(valuesCuda, other.valuesCuda, totalVals * sizeof(float), cudaMemcpyDeviceToDevice);
    }   
}

// Destructor
Tensor::~Tensor() {
    if (device.compare("cuda") == 0) {
        cudaFree(this->valuesCuda);
    }
}

// Assignment Operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    if (device.compare("cuda") == 0) {
        cudaFree(valuesCuda);
    }

    totalVals = other.totalVals;
    nDimensions = other.nDimensions;
    device = other.device;
    dimensions = other.dimensions;
    values = other.values;

    if (device.compare("cuda") == 0) {
        cudaMalloc(&valuesCuda, totalVals * sizeof(float));
        cudaMemcpy(valuesCuda, other.valuesCuda, totalVals * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return *this;
}

// Equality Operator
bool Tensor::operator==(const Tensor& other) {
    if (this == &other) {
        return true;
    }
    if (totalVals != other.totalVals || nDimensions != other.nDimensions || device != other.device) {
        return false;
    }
    if (dimensions != other.dimensions) {
        return false;
    }
    if (values != other.values) {
        return false;
    }
    if ("cuda" == device && other.device == "cuda"){
        vector<float> valuesCudaCopy(totalVals);
        cudaMemcpy(valuesCudaCopy.data(), other.valuesCuda, other.totalVals * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(values.data(), valuesCuda, totalVals * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < totalVals; i++) {
            if (valuesCuda[i] != valuesCudaCopy[i]) {
            return false;
            }
        }
    }
    return true;
}

// reshape function
void Tensor::reshape(const vector<int>& newDims) {
    int totalNewVals = accumulate(newDims.begin(), newDims.end(), 1, std::multiplies<int>());
    
    if(totalNewVals != totalVals) {
        throw std::invalid_argument("New dimensions do not match total number of elements.");
    }

    dimensions = newDims;
    nDimensions = newDims.size();
}

// Accessor functions
int Tensor::getTotalValues() const {
    return totalVals;
}

int Tensor::getNumDimensions() const {
    return nDimensions;
}

vector<int> Tensor::getDimensions() const{
    return dimensions;
}

string Tensor::getDevice() const {
    return device;
}

void Tensor::uploadFromCuda() {
    cudaMemcpy(values.data(), valuesCuda, totalVals * sizeof(float), cudaMemcpyDeviceToHost);
}

vector<float> Tensor::getValues() {
    if (device == "cuda") {
        uploadFromCuda();
    }
    return values;
}

//Operator Overloading
// Indexing Method
float Tensor::operator[](const vector<int>& indices) const {
    int calculatedIndex = 0;
    for (int i = 0; i < nDimensions; i++) {
        if (indices[i] < 0 || indices[i] >= dimensions[i]) {
            std::ostringstream msg;
            msg << "Index out of range For Dimensions for axis " << i << ".";
            throw std::out_of_range(msg.str());
        }
        calculatedIndex = calculatedIndex * dimensions[i] + indices[i];
    }
    if (calculatedIndex < 0 || calculatedIndex >= totalVals) {
        throw std::out_of_range("Index out of range.");
    }
    return values[calculatedIndex];
}

// String for Dimensions 
string Tensor::getDimensionsString() const {
    string dimensionsString = "(";
    for (int i = 0; i < nDimensions; i++) {
        dimensionsString += to_string(dimensions[i]);
        if (i != nDimensions - 1) {
            dimensionsString += ", ";
        }
    }
    dimensionsString += ")";
    return dimensionsString;
}

// toString function
string Tensor::toString() {
    ostringstream oss;
    if (nDimensions == 1) {
        oss << "[";
        for (int i = 0; i < totalVals; ++i) {
            oss << getValues()[i];
            if (i < totalVals - 1) oss << ", ";
        }
        oss << "]";
    } else if (nDimensions == 2) {
        oss << "[";
        for (int i = 0; i < dimensions[0]; ++i) {
            oss << "[";
            for (int j = 0; j < dimensions[1]; ++j) {
                oss << getValues()[i * dimensions[1] + j];
                if (j < dimensions[1] - 1) oss << ", ";
            }
            oss << "]";
            if (i < dimensions[0] - 1) oss << ",\n ";
        }
        oss << "]";
    } else if (nDimensions == 3) {
        oss << "[";
        for (int i = 0; i < dimensions[0]; ++i) {
            oss << "[";
            for (int j = 0; j < dimensions[1]; ++j) {
                oss << "[";
                for (int k = 0; k < dimensions[2]; ++k) {
                    oss << getValues()[i * dimensions[1] * dimensions[2] + j * dimensions[2] + k];
                    if (k < dimensions[2] - 1) oss << ", ";
                }
                oss << "]";
                if (j < dimensions[1] - 1) oss << ",\n  ";
            }
            oss << "]";
            if (i < dimensions[0] - 1) oss << ",\n ";
        }
        oss << "]";
    } else {
        oss << "Tensor of shape " << getDimensionsString() << " with " << totalVals << " values";
    }
    return oss.str();
}
// Function to check if two tensors are broadcastable
// Check numpy documentation for more info --> https://numpy.org/doc/stable/user/basics.broadcasting.html
bool broadcastable(Tensor a, Tensor b) {
    for(int i = 0; i < min(a.nDimensions, b.nDimensions); i++) {
        int indexA = a.nDimensions - i - 1;
        int indexB = b.nDimensions - i - 1;
        if((a.dimensions[indexA] != b.dimensions[indexB]) && a.dimensions[indexA] != 1 && b.dimensions[indexB] != 1 ) {
            return false;
        }
    }
    return true;
}



// Check if both tensor shapes are exactly equal
bool checkShape(Tensor a, Tensor b){
    if (a.nDimensions != b.nDimensions) {
        return false;
    }
    for(int i = 0; i < a.nDimensions; i++) {
        if(a.dimensions[i] != b.dimensions[i]) {
            return false;
        }
    }
    return true;
}


// Return the broadcasting shape for the new array
vector<int> getShapeBroadcasting(Tensor a, Tensor b){
    vector<int> dimensionsArray(max(a.nDimensions, b.nDimensions));
    for(int i = 0; i < max(a.nDimensions, b.nDimensions); i++) {
        if (i < a.nDimensions && i < b.nDimensions) {
            if (a.dimensions[i] == b.dimensions[i]) {
                dimensionsArray[i] = a.dimensions[i];
            } else if (a.dimensions[i] == 1) {
                dimensionsArray[i] = b.dimensions[i];
            } else if (b.dimensions[i] == 1) {
                dimensionsArray[i] = a.dimensions[i];
            } else {
                dimensionsArray[i] = -1;
            }
        } else if (i < a.nDimensions) {
            dimensionsArray[i] = a.dimensions[i];
        } else {
            dimensionsArray[i] = b.dimensions[i];
        }
    }
    return dimensionsArray;
}




Tensor operator+(Tensor a, Tensor b) {
    if(checkShape(a, b)) {
        if(a.nDimensions == 1 && b.nDimensions == 1) {
            if(a.device == "cuda" && b.device == "cuda") {
                float* results;
                cudaMalloc((void**)&results, a.getTotalValues() * sizeof(float));
                cudaMemset(results, 0, a.getTotalValues() * sizeof(float));
                vector_addition(a.valuesCuda, b.valuesCuda, results, a.totalVals);
                vector<float> host_values(a.getTotalValues());
                cudaMemcpy(host_values.data(), results, a.getTotalValues() * sizeof(float), cudaMemcpyDeviceToHost);
                vector<int> dims = {a.getTotalValues()};
                return Tensor(host_values, dims, string("cuda")); 
            } else {
                vector<float> results(a.totalVals);
                for(int i = 0; i < a.totalVals; i ++) {
                    results[i] = a.values[i] + b.values[i];
                }
                vector<int> dims = {a.getTotalValues()};
                return Tensor(results, dims);
            }
        }

        vector<int> broadcastingShape = getShapeBroadcasting(a, b);
        int numDims = max(a.nDimensions, b.nDimensions);
        int totalValues = broadcastingShape.size();

        vector<float> resultValues(totalValues);

        // n Dimensional Matrix Addition:
        if(a.device == "cuda" && b.device == "cuda") {
            vector<int> resultingDims = a.dimensions;

            resultingDims[a.nDimensions - 1] = b.dimensions[b.nDimensions - 1];
            int totalOutputVals = 1;
            for (int i = 0; i < a.nDimensions; i++) {
                totalOutputVals *= resultingDims[i];
            }
            vector<float> outputValues(totalOutputVals);
            int n = a.dimensions[a.nDimensions - 2];
            int m = a.dimensions[a.nDimensions - 1]; 
            int batchSize = totalOutputVals / (n*m);
            float* zeros_device;
            cudaMalloc((void**)&zeros_device, batchSize * n * m * sizeof(float));
            cudaMemset(zeros_device, 0, batchSize * m * n * sizeof(float));
            for (int batch = 0; batch < batchSize; batch++) {
                float* a_batch = a.valuesCuda + batch * (n * m);
                float* b_batch = b.valuesCuda + batch * (n * m);
                float* c_batch = zeros_device + batch * (m * n);
                matrix_addition(a_batch, b_batch, c_batch, n, m);

            }
            return Tensor(zeros_device, resultingDims, string("cuda"));
        } else {
            for(int i = 0; i < totalValues; i++) {
                int idxA = 0;
                int idxB = 0;
                int multiplierA = 1;
                int multiplierB = 1;
                for(int dim = numDims - 1; dim >= 0; dim--) {
                    int dimA = dim - (numDims - a.nDimensions);
                    int dimB = dim - (numDims - b.nDimensions);
                    int dimIndex = (i / multiplierA) % broadcastingShape[dim];

                    if (dimA >= 0 && a.dimensions[dimA] != 1) {
                        idxA += (dimIndex * multiplierA);
                    }
                    if (dimB >= 0 && b.dimensions[dimB] != 1) {
                        idxB += (dimIndex * multiplierB);
                    }

                    if (dimA >= 0) multiplierA *= a.dimensions[dimA];
                    if (dimB >= 0) multiplierB *= b.dimensions[dimB];
                }
                resultValues[i] = a.values[idxA] + b.values[idxB];
            }
            return Tensor(resultValues, broadcastingShape);
        }
    } else {
        vector<int> shapeBroadcast = getShapeBroadcasting(a, b);
        int maxDim = max(a.nDimensions, b.nDimensions);
        int minDim = min(a.nDimensions, b.nDimensions);
        vector<float> a_vals = a.getValues();
        vector<float> b_vals = b.getValues();
        int a_sizeFactor = 1;
        int b_sizeFactor = 1;

        for(int i = 0; i < maxDim; i++) {
            if (i < minDim) {
                int indexA = a.nDimensions - i - 1;
                int indexB = b.nDimensions - i - 1;
                if((a.dimensions[indexA] != b.dimensions[indexB]) && a.dimensions[indexA] != 1 && b.dimensions[indexB] != 1 ) {
                    std::ostringstream msg;
                    msg << "Tensor of shape" << a.getDimensionsString() << "is not broadcastable with tensor of shape" << b.getDimensionsString() << ".";
                    throw std::invalid_argument(msg.str());
                }
                if (a.dimensions[indexA] == 0 || a.dimensions[indexA] == 1) {
                    a_sizeFactor *= shapeBroadcast[maxDim - i - 1];
                }
                if (b.dimensions[indexB] == 0 || b.dimensions[indexB] == 1) {
                    b_sizeFactor *= shapeBroadcast[maxDim - i - 1];
                }
            } else {
                if (a.nDimensions > b.nDimensions) {
                    b_sizeFactor *= shapeBroadcast[maxDim - i - 1];
                } else {
                    a_sizeFactor *= shapeBroadcast[maxDim - i - 1];
                }
            }
        }
        
        vector<float> a_result(a_sizeFactor * a.totalVals);
        vector<float> b_result(b_sizeFactor * b.totalVals);
        
        for (int i = 0; i < a_sizeFactor; i++) {
            for (int j = 0; j < a.totalVals; j++) {
                a_result[i * a.totalVals + j] = a_vals[j];
            }
        }

        for (int i = 0; i < b_sizeFactor; i++) {
            for (int j = 0; j < b.totalVals; j++) {
                b_result[i * b.totalVals + j] = b_vals[j];
            }
        }

        Tensor a_tensor = Tensor(a_result, shapeBroadcast);
        Tensor b_tensor = Tensor(b_result, shapeBroadcast);
        return a_tensor + b_tensor;
    }
}


// Matrix Multiplication By Scalar
Tensor operator*(float x, Tensor a) {
    if (a.device == "cuda") {
        float* device_arr;
        cudaMalloc((void**)&device_arr, a.totalVals * sizeof(float));
        cudaMemset(device_arr, 0, a.totalVals * sizeof(float));
        matrix_scaling(a.valuesCuda, device_arr, x, a.getDimensions()[0], a.getDimensions()[1]);
        if (cudaPeekAtLastError() != cudaSuccess) {
            cudaFree(device_arr);  // Free memory on error
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(cudaGetLastError())));
        }
        //float* host_values = (float *)malloc(m * p * batchSize * sizeof(float));
        //cudaMemcpy(host_values, zeros_device, m* p * batchSize  * sizeof(float), cudaMemcpyDeviceToHost);
        return Tensor(device_arr, a.getDimensions(), string("cuda"));
    } else {
        vector<float> valuesCopy(a.values);
        for(int i = 0; i < a.totalVals; i++) {
            valuesCopy[i] = valuesCopy[i] * x; 
        }
        return Tensor(valuesCopy, a.dimensions);
    }
}



// Matrix Multiplication
Tensor operator*(Tensor a, Tensor b) {
    if (a.nDimensions >= 2 && b.nDimensions >= 2) {
        int lastDimA = a.dimensions[a.nDimensions - 1];
        int secondLastDimA = a.dimensions[a.nDimensions - 2];
        int lastDimB = b.dimensions[b.nDimensions - 1];
        int secondLastDimB = b.dimensions[b.nDimensions - 2];
        
        if (lastDimA != secondLastDimB && (lastDimA != 1 && lastDimB != 1)) {
            std::ostringstream msg;
            msg << "Tensor of shape" << a.getDimensionsString() << "is not possible to matrix multiply with tensor of shape" << b.getDimensionsString() << ".";
            throw std::invalid_argument(msg.str());
        }
    }

    if((a.nDimensions == 1 && b.nDimensions == 1) || (a.dimensions[a.nDimensions - 1] ==  1 && b.dimensions[b.nDimensions - 1] == 1)) {
        if(a.device == "cuda" && b.device == "cuda") {
            //cout << "Used Dot Product" << endl;
            vector<float> result(1);
            result[0] = 0;
            for(int i = 0; i < a.totalVals; i ++) {
                result[0] = result[0] + (a.getValues()[i] * b.getValues()[i]);
                
            }
            vector<int> resultDims = {1};
            return Tensor(result, resultDims, string("cuda"));

        } else {
            vector<float> result(1);
            result[0] = 0;
            for(int i = 0; i < a.totalVals; i ++) {
                result[0] = result[0] + (a.getValues()[i] * b.getValues()[i]);
                
            }
            vector<int> resultDims = {1};
            return Tensor(result, resultDims);
        }
    } else {
        vector<int> resultingDims = a.dimensions;
        resultingDims[a.nDimensions - 1] = b.dimensions[b.nDimensions - 1];
        int totalOutputVals = 1;
        for (int i = 0; i < a.nDimensions; i++) {
            totalOutputVals *= resultingDims[i];
        }
        vector<float> outputValues(totalOutputVals);
        int n = a.dimensions[a.nDimensions - 2];
        int m = a.dimensions[a.nDimensions - 1]; 
        int p = b.dimensions[b.nDimensions - 1];
        int batchSize = totalOutputVals / (m*p);
        if(a.device == "cuda" && b.device == "cuda") {
            float* zeros_device;
            cudaMalloc((void**)&zeros_device,batchSize * m * p * sizeof(float));
            cudaMemset(zeros_device, 0, batchSize * m * p * sizeof(float));
            //c_device[0] = 1;
            for (int batch = 0; batch < batchSize; batch++) {
                float* a_batch = a.valuesCuda + batch * (n * m);
                float* b_batch = b.valuesCuda + batch * (n * p);
                float* c_batch = zeros_device + batch * (m * p);

                matrix_multiplication(a_batch, b_batch, c_batch, n, m, p);
            }
            float* host_values = (float *)malloc(m * p * batchSize * sizeof(float));
            cudaMemcpy(host_values, zeros_device, m * p * batchSize * sizeof(float), cudaMemcpyDeviceToHost);
            vector<float> newVals(m* p * batchSize, 0);
            for (int i = 0; i <  m* p * batchSize; i++) {
                newVals[i] = host_values[i];
            }
            Tensor x(newVals, resultingDims, string("cuda"));
            return x;


        } else {
            for (int batch = 0; batch < batchSize; batch++) {
                for (int i = 0; i < resultingDims[a.nDimensions - 2]; i++) {
                    for (int j = 0; j < resultingDims[a.nDimensions - 1]; j++) {
                        float sum = 0.0;
                        for (int k = 0; k < a.dimensions[a.nDimensions - 1]; k++) {
                            int aIndex = batch * (a.dimensions[a.nDimensions - 2] * a.dimensions[a.nDimensions - 1]) + i * a.dimensions[a.nDimensions - 1] + k;
                            int bIndex = batch * (b.dimensions[b.nDimensions - 2] * b.dimensions[b.nDimensions - 1]) + k * b.dimensions[b.nDimensions - 1] + j;
                            sum += a.values[aIndex] * b.values[bIndex];
                        }
                        int outputIndex = batch * (a.dimensions[a.nDimensions - 2] * b.dimensions[b.nDimensions - 1]) + i * b.dimensions[b.nDimensions - 1] + j;
                        outputValues[outputIndex] = sum;
                    }
                }
            }

            // Return resulting dims of multiplied tensor
            return Tensor(outputValues, resultingDims);

        }
    }
}


float mean(Tensor a) {
    vector<float> values = a.getValues();
    float sum = 0;
    for (int i = 0; i < a.getTotalValues(); i++) {
        sum += values[i];
    }
    return sum / a.getTotalValues();
}

Tensor multiply(Tensor a, Tensor b) {
    if(checkShape(a, b)) {
        if(a.nDimensions == 1 && b.nDimensions == 1) {
            if(a.device == "cuda" && b.device == "cuda") {
                float* results;
                cudaMalloc((void**)&results, a.getTotalValues() * sizeof(float));
                cudaMemset(results, 0, a.getTotalValues() * sizeof(float));
                vector_addition(a.valuesCuda, b.valuesCuda, results, a.totalVals);
                vector<float> host_values(a.getTotalValues());
                cudaMemcpy(host_values.data(), results, a.getTotalValues() * sizeof(float), cudaMemcpyDeviceToHost);
                vector<int> dims = {a.getTotalValues()};
                return Tensor(host_values, dims, string("cuda")); 
            } else {
                vector<float> results(a.totalVals);
                for(int i = 0; i < a.totalVals; i ++) {
                    results[i] = a.values[i] + b.values[i];
                }
                vector<int> dims = {a.getTotalValues()};
                return Tensor(results, dims);
            }
        }

        vector<int> broadcastingShape = getShapeBroadcasting(a, b);
        int numDims = max(a.nDimensions, b.nDimensions);
        int totalValues = broadcastingShape.size();

        vector<float> resultValues(totalValues);

        // n Dimensional Matrix Addition:
        if(a.device == "cuda" && b.device == "cuda") {
            vector<int> resultingDims = a.dimensions;

            resultingDims[a.nDimensions - 1] = b.dimensions[b.nDimensions - 1];
            int totalOutputVals = 1;
            for (int i = 0; i < a.nDimensions; i++) {
                totalOutputVals *= resultingDims[i];
            }
            vector<float> outputValues(totalOutputVals);
            int n = a.dimensions[a.nDimensions - 2];
            int m = a.dimensions[a.nDimensions - 1]; 
            int batchSize = totalOutputVals / (n*m);
            float* zeros_device;
            cudaMalloc((void**)&zeros_device, batchSize * n * m * sizeof(float));
            cudaMemset(zeros_device, 0, batchSize * m * n * sizeof(float));
            for (int batch = 0; batch < batchSize; batch++) {
                float* a_batch = a.valuesCuda + batch * (n * m);
                float* b_batch = b.valuesCuda + batch * (n * m);
                float* c_batch = zeros_device + batch * (m * n);
                multiply_matrix_elements(a_batch, b_batch, c_batch, n, m);

            }
            return Tensor(zeros_device, resultingDims, string("cuda"));
        } else {
            for(int i = 0; i < totalValues; i++) {
                int idxA = 0;
                int idxB = 0;
                int multiplierA = 1;
                int multiplierB = 1;
                for(int dim = numDims - 1; dim >= 0; dim--) {
                    int dimA = dim - (numDims - a.nDimensions);
                    int dimB = dim - (numDims - b.nDimensions);
                    int dimIndex = (i / multiplierA) % broadcastingShape[dim];

                    if (dimA >= 0 && a.dimensions[dimA] != 1) {
                        idxA += (dimIndex * multiplierA);
                    }
                    if (dimB >= 0 && b.dimensions[dimB] != 1) {
                        idxB += (dimIndex * multiplierB);
                    }

                    if (dimA >= 0) multiplierA *= a.dimensions[dimA];
                    if (dimB >= 0) multiplierB *= b.dimensions[dimB];
                }
                resultValues[i] = a.values[idxA] * b.values[idxB];
            }
            return Tensor(resultValues, broadcastingShape);
        }
    } else {
        vector<int> shapeBroadcast = getShapeBroadcasting(a, b);
        int maxDim = max(a.nDimensions, b.nDimensions);
        int minDim = min(a.nDimensions, b.nDimensions);
        vector<float> a_vals = a.getValues();
        vector<float> b_vals = b.getValues();
        int a_sizeFactor = 1;
        int b_sizeFactor = 1;

        for(int i = 0; i < maxDim; i++) {
            if (i < minDim) {
                int indexA = a.nDimensions - i - 1;
                int indexB = b.nDimensions - i - 1;
                if((a.dimensions[indexA] != b.dimensions[indexB]) && a.dimensions[indexA] != 1 && b.dimensions[indexB] != 1 ) {
                    std::ostringstream msg;
                    msg << "Tensor of shape" << a.getDimensionsString() << "is not broadcastable with tensor of shape" << b.getDimensionsString() << ".";
                    throw std::invalid_argument(msg.str());
                }
                if (a.dimensions[indexA] == 0 || a.dimensions[indexA] == 1) {
                    a_sizeFactor *= shapeBroadcast[maxDim - i - 1];
                }
                if (b.dimensions[indexB] == 0 || b.dimensions[indexB] == 1) {
                    b_sizeFactor *= shapeBroadcast[maxDim - i - 1];
                }
            } else {
                if (a.nDimensions > b.nDimensions) {
                    b_sizeFactor *= shapeBroadcast[maxDim - i - 1];
                } else {
                    a_sizeFactor *= shapeBroadcast[maxDim - i - 1];
                }
            }
        }
        
        vector<float> a_result(a_sizeFactor * a.totalVals);
        vector<float> b_result(b_sizeFactor * b.totalVals);
        
        for (int i = 0; i < a_sizeFactor; i++) {
            for (int j = 0; j < a.totalVals; j++) {
                a_result[i * a.totalVals + j] = a_vals[j];
            }
        }

        for (int i = 0; i < b_sizeFactor; i++) {
            for (int j = 0; j < b.totalVals; j++) {
                b_result[i * b.totalVals + j] = b_vals[j];
            }
        }

        Tensor a_tensor = Tensor(a_result, shapeBroadcast);
        Tensor b_tensor = Tensor(b_result, shapeBroadcast);
        return a_tensor + b_tensor;
    }
}

Tensor mean(Tensor a, int dim) {
    vector<float> values = a.getValues();
    vector<int> dims = a.getDimensions();
    if (dim < 0) {
        std::ostringstream msg;
        msg << "Dimension cannot be less than 0.";
        throw std::invalid_argument(msg.str());
    }
    if (dim == 0) {
        vector<float> valuesDimZero(1);
        valuesDimZero[0] = mean(a);
        vector<int> dims = {1};
        Tensor result(valuesDimZero, dims);
        return result;
    }

    vector<int> cleanedDims(dim);
    int divisionDimFactor = 1;
    for (int i = 0; i < a.getNumDimensions(); i++) {
        if (i < dim) {
            cleanedDims[i] = dims[i];
        } else {
            divisionDimFactor *= dims[i];
        }
    }

    vector<float> means(a.getTotalValues() / divisionDimFactor);

    float temp = 0;
    for (int i = 0; i < a.getTotalValues(); i++) {
        temp += values[i];
        if ((i+1) % divisionDimFactor == 0) {
            means[((i+1)/divisionDimFactor) - 1] = temp / divisionDimFactor;
            temp = 0;
        }
    }


    Tensor result(means, cleanedDims);
    return result;
}


float standardDev(Tensor a) { 
    float aMean = mean(a);
    vector<float> values = a.getValues();
    float sumDeviances = 0;
    for (int i = 0; i < a.getTotalValues(); i++) {
        sumDeviances += abs(values[i] - aMean);
    }
    return sqrt(sumDeviances / a.getTotalValues());
}


// Tensor standardize(Tensor a) {
//     float sdv = standardDev(a);
//     float mn = mean(a);
//     vector<float> values = a.getValues();
//     vector<float> resValues(a.getTotalValues());
    
//     for (int i = 0; i < a.getTotalValues(); i++) {
//         resValues[i] = (values[i] - mn) / sdv;
//     }
//     vector<int> resDimensions = {1};
//     return Tensor result(resValues, resDimensions);
// }
