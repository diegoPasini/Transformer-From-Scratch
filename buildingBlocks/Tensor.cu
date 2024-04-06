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

// Cuda Tensor Constructor
Tensor::Tensor(float* valsCuda, int* dims, int numDims, string dev) { // Need to differentiate in some way
    dimensions = (int *)malloc(numDims * sizeof(int));
    int totalVals = 1;
    for (int i = 0; i < numDims; i++) {
        totalVals = totalVals * dims[i];
        dimensions[i] = dims[i];
    }
    this->totalVals = totalVals;
    this->nDimensions = numDims;
    this->values = (float *)malloc(totalVals * sizeof(float));

    if (dev.compare("cuda") == 0) {
        this->device = dev;
        cudaMalloc(&valuesCuda, totalVals * sizeof(float));
        cudaMemcpy(valuesCuda, valsCuda, totalVals * sizeof(float), cudaMemcpyDeviceToDevice);
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
            oss << values[i];
            if (i < totalVals - 1) oss << ", ";
        }
        oss << "]";
    } else if (nDimensions == 2) {
        oss << "[";
        for (int i = 0; i < dimensions[0]; ++i) {
            oss << "[";
            for (int j = 0; j < dimensions[1]; ++j) {
                oss << values[i * dimensions[1] + j];
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
                    oss << values[i * dimensions[1] * dimensions[2] + j * dimensions[2] + k];
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
        if((a.dimensions[indexA] != b.dimensions[indexB]) || a.dimensions[indexA] != 1 || b.dimensions[indexB] != 1 ) {
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


// Tensor Addition
Tensor operator+(Tensor a, Tensor b) {
    if (!(checkShape(a,b)) && !(broadcastable(a, b))) {
        std::ostringstream msg;
        msg << "Tensor of shape" << a.getDimensionsString() << "is not broadcastable with tensor of shape" << b.getDimensionsString() << ".";
        throw std::invalid_argument(msg.str());
    }

    if(a.nDimensions == 1 && b.nDimensions == 1) {
        float* results = (float *)malloc(a.totalVals * sizeof(float));
        if(a.device == "cuda" && b.device == "cuda") {
            vector_addition(a.getValues(), b.getValues(), results, a.totalVals); // Need to keep this on cuda and create tensor
            return; // Create new Tensor
        } else {
            for(int i = 0; i < a.totalVals; i ++) {
                results[i] = a.values[i] + b.values[i];
            }
            return; // Create new tensor
        }
    }

    // n Dimensional Matrix Addition:
    if(a.nDimensions == b.nDimensions && a.nDimensions != 1) {
        if(a.device == "cuda" && b.device == "cuda") {
            vector_addition(a.getValues(), b.getValues(), results, a.totalVals); // Need to keep this on cuda and create tensor
            return; // Create new Tensor
        } else {
            
        }
    }  
    else if 
}

// Matrix Scaling With Float
Tensor operator*(float a, Tensor b) {
    float* newValues = new float[b.totalVals];
    if (b.device.compare("cuda") == 0) {
        
    } else {
        for(int i  = 0; i < b.totalVals; i++) {
            newValues[i] = b.values[i] * a;
        }
    }
    Tensor result(newValues, b.dimensions, b.nDimensions, b.device);
    delete[] newValues;
    return result;
}


// Tensor operator*(const Tensor& a, const Tensor& b) {
//     if (!(broadcastable(a, b))) {
//         std::ostringstream msg;
//         msg << "Tensor of shape" << a.getDimensionsString() << "is not broadcastable with tensor of shape" << b.getDimensionsString() << ".";
//         throw std::invalid_argument(msg.str());
//     }
    
    
// }