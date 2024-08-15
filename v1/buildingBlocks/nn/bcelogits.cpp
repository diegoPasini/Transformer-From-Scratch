#include <vector>
#include "loss.h"
#include <cmath>

class BCEWithLogitsLoss : public Loss {
public:
    int n = 0;
    BCEWithLogitsLoss(int n) {
        this->n = n;
    }

    ~BCEWithLogitsLoss() {}

    float forward_loss(vector<float> logits, vector<float> targets) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            float sigmoid_output = 1 / (1 + exp(-logits[i]));
            sum += -targets[i] * log(sigmoid_output) - (1 - targets[i]) * log(1 - sigmoid_output);
        }
        return sum / n;
    }

    float backward_loss(float logit, float target) {
        float sigmoid_output = 1 / (1 + exp(-logit));
        return sigmoid_output - target;
    }
};
