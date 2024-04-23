#include <vector>

class Loss {
public:
    virtual ~Loss() {}

    virtual double forward_loss(vector<float> predicted, vector<float> actual);
    virtual double backward_loss(vector<float> output);
};
