#include <stdio.h>
#include <math.h>

// Take in raw attention scores and convert them into
// a probability distribution of weights.
static void softmax(float *raw_scores, int len) {
    // We first find the largest of the scores
    // as the maximum value
    float max_val = raw_scores[0];
    for (int i = 1; i < len; i++) {
        if (raw_scores[i] > max_val) max_val = raw_scores[i];
    }

    // Then we subtract each score by max, before
    // exponentiating for stability
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        raw_scores[i] = expf(raw_scores[i] - max_val);
        sum += raw_scores[i];
    }
    // Normalization of weights to become a
    // distribution summing to 1
    for (int i = 0; i < len; i++) {
        raw_scores[i] /= sum;
    }
}

// Single attention head for processing a single query token
// against a full sequence of K/V pairs, but keep in mind 
// real transformers stack multiple heads in parallel
static void self_attention_head(float *out, float *q, float *k,
                         float *v, int seq_l, int head_d) {
    // Specifically, q is the token's query vector; k is
    // our key matrix; v is our value matrix; seq_l is
    // the number of tokens in the sequence; head_d is
    // the shared dimension of all Q/K/V vectors; and
    // out is our output vector
    
    float scores[seq_l];

    // We first calculate the attention scores
    for (int i = 0; i < seq_l; i++) {
        // Compute the Q/K^T dot product
        scores[i] = 0.0f;
        for (int j = 0; j < head_d; j++) {
            scores[i] += q[j] * k[(i * head_d) + j];
        }

        // Divide the score by root of d_k so
        // we don't oversaturate softmax
        scores[i] /= sqrtf(head_d);
    }

    // Convert the raw scores into attention weights
    softmax(scores, seq_l);

    // Get our output via the weighted sum of all values
    for (int i = 0; i < head_d; i++) {
        out[i] = 0.0f;
        for(int j = 0; j < seq_l; j++) {
            out[i] += scores[j] * v[(j * head_d) + i];
        }
    }
}

int main(void) {
    // So in a real transformer, we get our Q/K/V matrices by 
    // projecting token embeddings through learned weight
    // matrices W_Q, W_K, and W_V, but here I just added
    // some example vectors to represent what a model
    // produces for "the river bank", where we have
    // 3 tokens of "the", "river", and "bank"
    const int seq_l = 3;
    const int head_d = 4;
    // We're specifically looking at the output for the
    // token "bank", so we'll only have the query of
    // this token
    float q[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    // Our keys
    float k[3 * 4] = {0.5f, 0.5f, 0.5f, 0.1f,   // k_the
                      2.5f, 2.5f, 2.5f, 1.0f,   // k_river
                      1.0f, 1.0f, 1.0f, 0.5f};  // k_bank
    // Our values
    float v[3 * 4] = {0.0f, 1.0f, 0.3f, 0.2f,   // v_the
                      4.0f, 3.0f, 0.5f, 0.8f,   // v_river
                      1.0f, 1.0f, 0.4f, 0.3f};  // v_bank
    // And our output vector
    float out[4] = {0};

    // Now run the head and print the output for "bank"
    self_attention_head(out, q, k, v, seq_l, head_d);
    printf("OUTPUT o_bank = (");
    for (int i = 0; i < head_d - 1; i++) {
        printf("%.4f, ", out[i]);
    }
    printf("%.4f)\n", out[head_d - 1]);
    printf("(You'll notice the first two components, %.1f and %.1f, align with v_river's 4.0 and 3.0)\n", out[0], out[1]);
    
    return 0;
}
