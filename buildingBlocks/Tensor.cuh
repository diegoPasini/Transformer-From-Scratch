#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <string>
#include <vector>
using namespace std;


class Tensor {
    public:
        // Constructor: Initializes a tensor with given values, dimensions, and optionally specifies the device
        Tensor();
        Tensor(const vector<float>& vals, const vector<int>& dims, string dev = "");
        //Tensor(float& c_device, const vector<int>& dims, string dev = "cuda");

        Tensor(float* c_device, const vector<int>& dims, string dev = "");


        // Destructor: Responsible for freeing allocated resources
        ~Tensor();

        // Copy constructor and copy assignment operator
        Tensor(const Tensor& other);

        // Assignment Operator
        Tensor& operator=(const Tensor& other);

        // Equality Operator
        bool operator==(const Tensor& other);

        // Accessor functions
        int getTotalValues() const;
        int getNumDimensions() const;
        vector<int> getDimensions() const;
        string getDevice() const;
        vector<float> getValues();
        void transpose();


        // Operation to reshape the tensor
        void reshape(const vector<int>& newDims);

        void uploadFromCuda();
        
        // Indexing Operator
        float operator[](const vector<int>& indices) const;

        // Matrix Addition
        friend Tensor operator+(Tensor a, Tensor b);

        // Matrix Scaling
        friend Tensor operator*(float a, Tensor b);

        // Matrix Multiplication
        friend Tensor operator*(Tensor a, Tensor b);

        

        // toString Dimensions
        string getDimensionsString() const;

        // toString function
        string toString();

        // TO - DO: FINISH BROADCASTABLE IMPLEMENTATION:
        // // Checks if two tensors are broadcastable
        friend bool broadcastable(Tensor a, Tensor b);

        // Check if two tensors have an equal shape
        friend bool checkShape(Tensor a, Tensor b);

        // // Get Broadcastable Shape
        friend vector<int> getShapeBroadcasting(Tensor a, Tensor b);

        friend Tensor multiply(Tensor a, Tensor b); 


        // Get mean
        friend float mean(Tensor a);

        // Get mean with dimension
        friend Tensor mean(Tensor a, int dim);

        // Get mean
        friend float standardDev(Tensor a);

        // Standardize, treated as 1D
        //friend Tensor standardize(Tensor a);

    private:
        vector<float> values;     // Pointer to the tensor's values
        int totalVals;     // Total number of values
        float* valuesCuda; // Pointer to the tensor's values on CUDA device
        vector<int> dimensions;   // Pointer to the tensor's dimensions
        int nDimensions;   // Number of dimensions
        string device = ""; // The computing device, e.g., "cuda" for CUDA devices
};

#endif
