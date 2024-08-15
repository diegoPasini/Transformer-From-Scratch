#include <vector>
#ifndef LOSS_H  
#define LOSS_H 
class Loss {
public:
    virtual ~Loss() {}

    virtual float forward_loss(vector<float> predicted, vector<float> actual) = 0;
    virtual float backward_loss(float predicted, float actual) = 0;
};

#endif
