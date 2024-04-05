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


// Destructor
Tensor::~Tensor() {
    free(dimensions);
    free(values);
    if (device.compare("cuda") == 0) {
        cudaFree(valuesCuda);
    }
}

// Accessor functions
// Get the total number of values in the tensor
int Tensor::getTotalValues() {
    return totalVals;
}

// Get the number of dimensions in the tensor
int Tensor::getNumDimensions() {
    return nDimensions;
}

// Get the dimensions of the tensor
int* Tensor::getDimensions() {
    int* dimensionsCopy = (int *)malloc(nDimensions * sizeof(int));
    for(int i = 0; i < nDimensions; i++) {
        dimensionsCopy[i] = dimensions[i];
    }
    return dimensionsCopy;
}

// Get the device type of the tensor
string Tensor::getDevice() {
    return device;
}

// Get the values of the tensor
float* Tensor::getValues() {
    float* valuesCopy = (float *)malloc(totalVals * sizeof(float));
    for(int i = 0; i < totalVals; i++) {
        valuesCopy[i] = values[i];
    }
    return valuesCopy;
}

// Get the CUDA device values of the tensor
// Maybe make this the return in the original values for simplicity.
float* Tensor::getValuesCuda() {
    if (device.compare("cuda") != 0) {
        std::invalid_argument("CUDA not enabled on tensor, so values cannot be accessed");
    }
    float* valuesCudaCopy = (float *)malloc(totalVals * sizeof(float));
    cudaMemcpy(valuesCudaCopy, valuesCuda, totalVals * sizeof(float), cudaMemcpyDeviceToHost);
    return valuesCudaCopy;
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


//Operator Overloading
// Accessing Method
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

// Non CUDA Tensor Operations

// Tensor::Tensor {
//     private:
//         // The values will be stored using a pointer. This helps with efficiency for retrieval
//         float* values;
//         int totalVals;
//         float* valuesCuda;
//         // Dimensions are also stored with a pointer and with an n value for teh number of dimensions
//         int* dimensions;
//         int nDimensions;

//         string device = "";
//         // Constructors

        

        

// };




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

// Tensor Addition
Tensor operator+(Tensor a, Tensor b) {
    if (!(checkShape(a,b)) && !(broadcastable(a, b))) {
        std::ostringstream msg;
        msg << "Tensor of shape" << a.getDimensionsString() << "is not broadcastable with tensor of shape" << b.getDimensionsString() << ".";
        throw std::invalid_argument(msg.str());
    }

    

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