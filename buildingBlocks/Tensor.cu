#include <matrixOperations.cuh>;
#include <iostream>;
#include <string>
#include <sstream> 

using namespace std;
using namespace MatrixOperations;


class Tensor {
    private:
        // The values will be stored using a pointer. This helps with efficiency for retrieval
        float* values;
        int totalVals;
        float* valuesCuda;
        // Dimensions are also stored with a pointer and with an n value for teh number of dimensions
        int* dimensions;
        int nDimensions;

        string device = "";
        // Constructors

        // Full Tensor Constructor
        Tensor(float* vals, int* dims, int numDims, string dev = "") {
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

        // Accessor functions
        // Get the total number of values in the tensor
        int getTotalValues() const {
            return totalVals;
        }

        // Get the number of dimensions in the tensor
        int getNumDimensions() const {
            return nDimensions;
        }

        // Get the dimensions of the tensor
        const int* getDimensions() const {
            return dimensions;
        }

        // Get the device type of the tensor
        string getDevice() const {
            return device;
        }

        // Get the values of the tensor
        const float* getValues() const {
            return values;
        }

        // Get the CUDA device values of the tensor
        const float* getValuesCuda() const {
            if (device.compare("cuda") != 0) {
                std::invalid_argument("CUDA not enabled on tensor, so values cannot be accessed");
            }
            return valuesCuda;
        }

        // reshape function
        void reshape(int* newDims, int newNumDims) {
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
        float& operator[](const int* indices) {
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

        // Tensor Multiplication
        friend Tensor operator*(const Tensor& a, const Tensor& b);

        // Tensor Addition

        // Tensor Subtraction

        // Tensor toString

        // String for Dimensions 
        string getDimensions() {
            string dimensionsString = "(";
            for (int i = 0; i < nDimensions; i++) {
                dimensionsString += to_string(dimensions[i]);
                if (i != nDimensions - 1) {
                    dimensionsString += ", ";
                }
            }
            dimensionsString += to_string(dimensions[nDimensions - 1]);
            dimensionsString += ")";
            return dimensionsString;
        }


        // Broadcastable Check
        friend bool broadcastable(Tensor a, Tensor b);

        // Non CUDA Tensor Operations

        

};

// Function to check if two tensors are broadcastable
// Check numpy documentation for more info --> https://numpy.org/doc/stable/user/basics.broadcasting.html
bool broadcastable(Tensor a, Tensor b) {
    for(int i = 0; i < min(a.nDimensions, b.nDimensions); i--) {
        int indexA = a.nDimensions - i - 1;
        int indexB = b.nDimensions - i - 1;
        if((indexA != indexB) || indexA != 1 || indexB != 1 ) {
            return false;
        }
    }
    return true;
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


Tensor operator*(const Tensor& a, const Tensor& b) {
    if (!(broadcastable(a, b))) {
        std::ostringstream msg;
        msg << "Tensor of shape" << a.getDimensions() << "is not broadcastable with tensor of shape" << b.getDimensions() << ".";
        throw std::invalid_argument(msg.str());
    }


    if 
    float* resultValues = new float[a.totalVals];
    for (int i = 0; i < a.totalVals; ++i) {
        resultValues[i] = a.values[i] * b.values[i];
    }

    Tensor result(resultValues, a.dimensions, a.nDimensions, a.device); 

    delete[] resultValues; 

    return result;
}