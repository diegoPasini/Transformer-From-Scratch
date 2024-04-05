#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <string>
using namespace std;


class Tensor {
    public:
        // Constructor: Initializes a tensor with given values, dimensions, and optionally specifies the device
        Tensor(float* vals, int* dims, int numDims, std::string dev = "");

        // Destructor: Responsible for freeing allocated resources
        ~Tensor();

        // Copy constructor and copy assignment operator
        // If you are managing resources like dynamically allocated memory, you need to define them
        Tensor(const Tensor& other);

        

        //Tensor& operator=(const Tensor& other);

        // Move constructor and move assignment operator
        // For efficiently transferring resources
        //Tensor(Tensor&& other) noexcept;
        //Tensor& operator=(Tensor&& other) noexcept;

        // Accessor functions
        int getTotalValues();
        int getNumDimensions();
        int* getDimensions();
        string getDevice();
        float* getValues();
        float* getValuesCuda();

        // Operation to reshape the tensor
        void reshape(int* newDims, int newNumDims);

        // Operator overloads for accessing elements and tensor operations
        float operator[](int* indices);
        friend Tensor operator+(Tensor a, Tensor b);
        friend Tensor operator*(float a, Tensor b);
        friend Tensor operator*(Tensor a, Tensor b);

        // Utility functions
        string getDimensionsString();

        // Checks if two tensors are broadcastable
        friend bool broadcastable(Tensor a, Tensor b);
        friend bool checkShape(Tensor a, Tensor b);

    private:
        float* values;     // Pointer to the tensor's values
        int totalVals;     // Total number of values
        float* valuesCuda; // Pointer to the tensor's values on CUDA device
        int* dimensions;   // Pointer to the tensor's dimensions
        int nDimensions;   // Number of dimensions
        string device = ""; // The computing device, e.g., "cuda" for CUDA devices
};

#endif // TENSOR_CUH_
