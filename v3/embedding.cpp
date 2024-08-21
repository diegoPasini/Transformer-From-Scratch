#include <vector>
#include <random>
#include <cmath>

using namespace std;

class Embedding {
private:
    int vocab_size;
    int embedding_dim;
    vector<vector<float>> embeddings;

    void initialize_embeddings() {
        float k = sqrt(1.0f / embedding_dim);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> distr(-k, k);

        embeddings.resize(vocab_size, vector<float>(embedding_dim));
        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < embedding_dim; ++j) {
                embeddings[i][j] = distr(gen);
            }
        }
    }

public:
    Embedding(int vocab_size, int embedding_dim)
        : vocab_size(vocab_size), embedding_dim(embedding_dim) {
        initialize_embeddings();
    }

    vector<float> forward(int token) {
        if (token < 0 || token >= vocab_size) {
            throw out_of_range("Token index out of range");
        }
        return embeddings[token];
    }

    vector<vector<float>> forward(const vector<int>& tokens) {
        vector<vector<float>> output;
        for (int token : tokens) {
            output.push_back(forward(token));
        }
        return output;
    }

    vector<vector<vector<float>>> forward(const vector<vector<int>>& batch_tokens) {
        vector<vector<vector<float>>> batch_output;
        for (const auto& tokens : batch_tokens) {
            batch_output.push_back(forward(tokens));
        }
        return batch_output;
    }

    vector<vector<float>> backward(const vector<vector<float>>& d_outputs, const vector<int>& tokens, float learning_rate) {
        vector<vector<float>> d_embeddings(vocab_size, vector<float>(embedding_dim, 0.0f));

        for (size_t i = 0; i < tokens.size(); ++i) {
            int token = tokens[i];
            if (token < 0 || token >= vocab_size) {
                throw out_of_range("Token index out of range");
            }
            for (int j = 0; j < embedding_dim; ++j) {
                d_embeddings[token][j] += d_outputs[i][j];
            }
        }

        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < embedding_dim; ++j) {
                embeddings[i][j] -= learning_rate * d_embeddings[i][j];
            }
        }

        return d_embeddings;
    }

    vector<vector<vector<float>>> backward(const vector<vector<vector<float>>>& d_outputs, const vector<vector<int>>& batch_tokens, float learning_rate) {
        vector<vector<vector<float>>> batch_d_embeddings(batch_tokens.size());

        for (size_t i = 0; i < batch_tokens.size(); ++i) {
            batch_d_embeddings[i] = backward(d_outputs[i], batch_tokens[i], learning_rate);
        }

        return batch_d_embeddings;
    }
};
