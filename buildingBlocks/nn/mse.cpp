#include <vector>
#include "loss.h"
#include <cmath>

class MSE : public Loss {
public:
    int n = 0;
    MSE(int n) {
        this-> n = n;
    }

    ~MSE(){}

    float forward_loss(vector<float> predicted, vector<float> actual) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += pow(( predicted[i] - actual[i]), 2);
        }
        sum /= n;
        return sum;
    }

    float backward_loss(float predicted, float actual) {
        return predicted - actual;
    } 
};
