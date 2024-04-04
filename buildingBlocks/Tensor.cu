#include <matrixOperations.cuh>;
#include <iostream>;
#include <string>
using namespace std;

class Tensor {
    private:
        // The values will be stored using a pointer. This helps with efficiency for retrieval
        float* values;
        float* valuesCuda;
        // Dimensions are also stored with a pointer and with an n value for teh number of dimensions
        int* dimensions;
        int n;

        string device = "";
    
    public: 

        // Full Constructor
        Tensor(float* vals, int* dims, int numDims, string dev = "") {
            dimensions = (int *)malloc(numDims*sizeof(int));
            int totalVals = 1;
            for(int i = 0; i < numDims; i++) {
                totalVals = totalVals * dims[i];
                dimensions[i] = dims[i];
            }
            
            values = (float *)malloc(totalVals*sizeof(float));
            for(int j = 0; j < totalVals; j++) {
                values[j] = vals[j];
            }

            if(dev.compare("cuda") == 0){
                device = dev;
                cudaMalloc(&valuesCuda, totalVals*sizeof(float));
                cudaMemcpy(valuesCuda, values, totalVals*sizeof(float), cudaMemcpyHostToDevice);
            }
        }

        int* getDimensions() {
            return dimensions;
        }

        int numDimensions() {
            return n;
        }

        //Operator Overloading
        

};