#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>

// Mean Squared Error
float mean_squared_error(vector<float> predicted, vector<float> actual){
    float summation = 0;
    for(int i = 0; i < actual.size(); i++){
        summation += (predicted[i] - actual[i])**2;
    }
    summation /= actual.size();
    return summation;
}

// Cross Entropy Loss
// THIS ASSUMES SOFTMAX HAS ALREADY BEEN APPLIED TO THE PREDICTIONS
float categorical_cross_entropy(vector<float> predicted, vector<float> actual) {
    loss = 0;
    for(int i = 0; i < predicted.size(); i++) {
        loss = loss + (-1 * actual[i] * log(predicted[i]));
    }
    return loss;
}
