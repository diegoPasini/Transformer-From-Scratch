#include "Tensor.cuh"
#include "matrixOperations.cuh"
#include <iostream>
#include <string>
#include <sstream> 

using namespace std;
using namespace MatrixOperations;

// Full Tensor Constructor
Tensor::Tensor(float* vals, int* dims, int numDims, string dev) {
    dimensions = (int *)malloc(numDims*sizeof(int));
    int totalVals = 1;
    for(int i = 0; i < numDims; i++) {
        totalVals = totalVals * dims[i];
        dimensions[i] = dims[i];
    }
    this->totalVals = totalVals;
    this->nDimensions = numDims;
    this->values = (float *)malloc(totalVals*sizeof(float));
    for(int j = 0; j < totalVals; j++) {
        values[j] = vals[j];
    }

    if(dev.compare("cuda") == 0){
        this->device = dev;
        cudaMalloc(&valuesCuda, totalVals*sizeof(float));
        cudaMemcpy(valuesCuda, values, totalVals*sizeof(float), cudaMemcpyHostToDevice);
    }
}


// Copy Constructor
Tensor::Tensor(const Tensor& other) {
    this->totalVals = other.totalVals;
    this->nDimensions = other.nDimensions;
    this->device = other.device;

    this->dimensions = (int *)malloc(nDimensions * sizeof(int));
    for(int i = 0; i < nDimensions; i++) {
        this->dimensions[i] = other.dimensions[i];
    }

    this->values = (float *)malloc(totalVals * sizeof(float));
    for(int i = 0; i < totalVals; i++) {
        this->values[i] = other.values[i];
    }

    if (device.compare("cuda") == 0) {
        cudaMalloc(&valuesCuda, totalVals * sizeof(float));
        cudaMemcpy(valuesCuda, other.valuesCuda, totalVals * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}


// Cuda Memory Reference constructor:
Tensor::Tensor(float** c_device, int* dims, int numDims, string dev){
    dimensions = (int *)malloc(numDims * sizeof(int));
    int totalVals = 1;
    for(int i = 0; i < numDims; i++) {
        totalVals *= dims[i];
        dimensions[i] = dims[i];
    }
    this->totalVals = totalVals;
    this->nDimensions = numDims;
    this->device = dev;
    
    if (dev == "cuda") {
        valuesCuda = *c_device;
    } else {
        std::cerr << "Error: Non-CUDA device specified for CUDA memory constructor." << std::endl;
    }
}


// Destructor
Tensor::~Tensor() {
    free(dimensions);
    free(values);
    if (device.compare("cuda") == 0) {
        cudaFree(valuesCuda);
    }
}


// Assignment Operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    free(dimensions);
    free(values);
    if (device.compare("cuda") == 0) {
        cudaFree(valuesCuda);
    }

    this->totalVals = other.totalVals;
    this->nDimensions = other.nDimensions;
    this->device = other.device;

    this->dimensions = (int *)malloc(nDimensions * sizeof(int));
    for(int i = 0; i < nDimensions; i++) {
        this->dimensions[i] = other.dimensions[i];
    }

    this->values = (float *)malloc(totalVals * sizeof(float));
    for(int i = 0; i < totalVals; i++) {
        this->values[i] = other.values[i];
    }

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
    if (this->totalVals != other.totalVals || this->nDimensions != other.nDimensions || this->device != other.device) {
        return false;
    }
    for (int i = 0; i < this->nDimensions; i++) {
        if (this->dimensions[i] != other.dimensions[i]) {
            return false;
        }
    }
    for (int i = 0; i < this->totalVals; i++) {
        if (this->values[i] != other.values[i]) {
            return false;
        }
    }
    if (this->device.compare("cuda") == 0 && other.device.compare("cuda") == 0) {
        float* valuesCudaCopy = (float *)malloc(totalVals * sizeof(float));
        cudaMemcpy(valuesCudaCopy, other.valuesCuda, totalVals * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < this->totalVals; i++) {
            if (this->valuesCuda[i] != valuesCudaCopy[i]) {
                free(valuesCudaCopy);
                return false;
            }
        }   
          free(valuesCudaCopy);
    }
    return true;
}

// reshape function
void Tensor::reshape(int* newDims, int newNumDims) {
    int totalNewVals = 1;
    for(int i = 0; i < newNumDims; i++) {
        totalNewVals *= newDims[i];
    }
    
    if(totalNewVals != this->totalVals) {
        throw std::invalid_argument("New dimensions do not match total number of elements.");
    }

    free(this->dimensions);
    this->dimensions = (int *)malloc(newNumDims * sizeof(int));
    for(int i = 0; i < newNumDims; i++) {
        this->dimensions[i] = newDims[i];
    }
    this->nDimensions = newNumDims;
}


// Accessor functions
int Tensor::getTotalValues() {
    return totalVals;
}

int Tensor::getNumDimensions() {
    return nDimensions;
}

int* Tensor::getDimensions() {
    int* dimensionsCopy = (int *)malloc(nDimensions * sizeof(int));
    for(int i = 0; i < nDimensions; i++) {
        dimensionsCopy[i] = dimensions[i];
    }
    return dimensionsCopy;
}

string Tensor::getDevice() {
    return device;
}

// Get Values - Returns CUDA values if enabled
float* Tensor::getValues() {
    float* valuesCopy = (float *)malloc(totalVals * sizeof(float));
    if (device.compare("cuda") != 0) {
        for(int i = 0; i < totalVals; i++) {
            valuesCopy[i] = values[i];
        }
    } else {
        cudaMemcpy(valuesCopy, valuesCuda, totalVals * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return valuesCopy;
}


//Operator Overloading
// Indexing Method
float Tensor::operator[](int* indices) {
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
string Tensor::getDimensionsString() {
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
int* getShapeBroadcasting(Tensor a, Tensor b) {
    int* dimensionsArray = (int *)malloc(max(a.nDimensions, b.nDimensions) * sizeof(int));
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
            float* results = (float *)malloc(a.totalVals * sizeof(float));
            if(a.device == "cuda" && b.device == "cuda") {
                vector_addition(a.getValues(), b.getValues(), results, a.totalVals);
                return Tensor(results, &(a.totalVals), 1, "cuda"); 
            } else {
                for(int i = 0; i < a.totalVals; i ++) {
                    results[i] = a.values[i] + b.values[i];
                }
                return Tensor(results, &(a.totalVals), 1);
            }
        }

        int* broadcastingShape = getShapeBroadcasting(a, b);
        int numDims = max(a.nDimensions, b.nDimensions);
        int totalValues = 1;
        for (int i = 0; i < numDims; i++) {
            totalValues *= broadcastingShape[i];
        }

        float* resultValues = (float *)malloc(totalValues * sizeof(float));

        // n Dimensional Matrix Addition:
        if(a.device == "cuda" && b.device == "cuda") {
            return Tensor(resultValues, broadcastingShape, numDims);
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
            return Tensor(resultValues, broadcastingShape, numDims);
        }

    } else {
        int* shapeBroadcast = getShapeBroadcasting(a, b);
        int maxDim = max(a.nDimensions, b.nDimensions);
        int minDim = min(a.nDimensions, b.nDimensions);
        float* a_vals = a.getValues();
        float* b_vals = b.getValues();
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
        
        float* a_result = new float[a_sizeFactor * a.totalVals];
        float* b_result = new float[b_sizeFactor * b.totalVals];
        
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

        Tensor a_tensor = Tensor(a_result, shapeBroadcast, maxDim);
        Tensor b_tensor = Tensor(b_result, shapeBroadcast, maxDim);
        return a_tensor + b_tensor;
    }
}


// Matrix Multiplication
Tensor operator*(Tensor a, Tensor b) {
    if (!(checkShape(a, b))) {
        std::ostringstream msg;
        msg << "Tensor of shape" << a.getDimensionsString() << "does not have same shape as tensor with shape" << b.getDimensionsString() << ".";
        throw std::invalid_argument(msg.str());
    }

    if (a.nDimensions >= 2 && b.nDimensions >= 2) {
        int lastDimA = a.dimensions[a.nDimensions - 1];
        int secondLastDimA = a.dimensions[a.nDimensions - 2];
        int lastDimB = b.dimensions[b.nDimensions - 1];
        int secondLastDimB = b.dimensions[b.nDimensions - 2];
        
        if (lastDimA != secondLastDimB || lastDimB != secondLastDimA) {
            std::ostringstream msg;
            msg << "Tensor of shape" << a.getDimensionsString() << "is not possible to matrix multiply with tensor of shape" << b.getDimensionsString() << ".";
            throw std::invalid_argument(msg.str());
        }
    }

    if(a.nDimensions == 1 && b.nDimensions == 1) {
        if(a.device == "cuda" && b.device == "cuda") {
            float* c_device;
            cudaMalloc((void**)&c_device, sizeof(float));
            dot_product(a.getValues(), b.getValues(), c_device, a.totalVals);
            
            
            int* resultDims = new int[1];
            resultDims[0] = 1;
            return Tensor(&c_device, resultDims, 1, "cuda"); 

            
        } else {
            float* result = (float *)malloc(sizeof(float));
            for(int i = 0; i < a.totalVals; i ++) {
                result[0] = result[0] + (a.values[i] * b.values[i]);
            }
            int* resultDims = new int[1];
            resultDims[0] = 1;
            return Tensor(result, resultDims, 1);
        }
    } else {
        int* resultingDims = new int[a.nDimensions];
        for(int i = 0; i < a.nDimensions - 1; i++) {
            resultingDims[i] = a.dimensions[i];
        }
        resultingDims[a.nDimensions - 1] = b.dimensions[b.nDimensions - 1];

        int totalOutputVals = 1;
        for (int i = 0; i < a.nDimensions; i++) {
            totalOutputVals *= resultingDims[i];
        }
        float* outputValues = new float[totalOutputVals];
        int n = totalOutputVals / (a.dimensions[a.nDimensions - 2] * b.dimensions[b.nDimensions - 1]);
        int m = a.dimensions[a.nDimensions - 1]; 
        int p = b.dimensions[b.nDimensions - 1];
        int batchSize = totalOutputVals / (resultingDims[a.nDimensions - 2] * resultingDims[a.nDimensions - 1]);
        if(a.device == "cuda" && b.device == "cuda") {
            float* c_device;
            cudaMalloc((void**)&c_device, n * p * sizeof(float));
            for (int batch = 0; batch < batchSize; batch++) {
                float* a_batch = a.getValues() + batch * (a.dimensions[a.nDimensions - 2] * m);
                float* b_batch = b.getValues() + batch * (b.dimensions[b.nDimensions - 2] * p);
                float* c_batch = c_device + batch * (a.dimensions[a.nDimensions - 2] * p);
                matrix_multiplication(a_batch, b_batch, c_batch, n, m, p);
            }
            return Tensor(&c_device, resultingDims, a.nDimensions, "cuda");
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

            return Tensor(outputValues, resultingDims, a.nDimensions);

        }
    }
}