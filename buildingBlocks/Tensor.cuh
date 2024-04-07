#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <string>
using namespace std;


class Tensor {
    public:
        // Constructor: Initializes a tensor with given values, dimensions, and optionally specifies the device
        Tensor(float* vals, int* dims, int numDims, string dev = "");

        Tensor(float** c_device, int* dims, int numDims, string dev = "");

        // Destructor: Responsible for freeing allocated resources
        ~Tensor();

        // Copy constructor and copy assignment operator
        Tensor(const Tensor& other);

        // Assignment Operator
        Tensor& operator=(const Tensor& other);

        // Equality Operator
        bool operator==(const Tensor& other);

        // Accessor functions
        int getTotalValues();
        int getNumDimensions();
        int* getDimensions();
        string getDevice();
        float* getValues();

        // Operation to reshape the tensor
        void reshape(int* newDims, int newNumDims);

        
        // Indexing Operator
        float operator[](int* indices);

        // Matrix Addition
        friend Tensor operator+(Tensor a, Tensor b);

        // Matrix Scaling
        friend Tensor operator*(float a, Tensor b);

        // Matrix Multiplication
        friend Tensor operator*(Tensor a, Tensor b);



        // toString Dimensions
        string getDimensionsString();

        // toString function
        string toString();

        // TO - DO: FINISH BROADCASTABLE IMPLEMENTATION:
        // // Checks if two tensors are broadcastable
        friend bool broadcastable(Tensor a, Tensor b);

        // Check if two tensors have an equal shape
        friend bool checkShape(Tensor a, Tensor b);

        // // Get Broadcastable Shape
        friend int* getShapeBroadcasting(Tensor a, Tensor b);


        // Get mean
        friend float mean(Tensor a);

        // Get mean with dimension
        friend Tensor mean(Tensor a, int dim);

        // Get mean
        friend float standardDev(Tensor a);

        // Standardize, treated as 1D
        friend Tensor standardize(Tensor a);

    private:
        float* values;     // Pointer to the tensor's values
        int totalVals;     // Total number of values
        float* valuesCuda; // Pointer to the tensor's values on CUDA device
        int* dimensions;   // Pointer to the tensor's dimensions
        int nDimensions;   // Number of dimensions
        string device = ""; // The computing device, e.g., "cuda" for CUDA devices
};

#endif
