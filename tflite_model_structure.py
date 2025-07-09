// Estrutura didática de um modelo .tflite representada em C++
// Explica como funciona internamente um modelo treinado, com base em rede neural simples
// usada, por exemplo, para controle de prótese com sinais EMG embarcada em ESP32/STM32

#include <vector>
#include <string>
#include <cmath>

// 1. Estrutura da Rede Neural (camadas)
struct LayerDefinition {
    std::string type;         // Tipo da camada: Input, Dense, etc
    int input_size;           // Tamanho da entrada
    int output_size;          // Tamanho da saída
    std::string activation;   // Função de ativação
};

std::vector<LayerDefinition> network_structure = {
    {"Input", 10, 10, "none"},
    {"Dense", 10, 8, "relu"},
    {"Dense", 8, 2, "softmax"}
};

// 2. Pesos e Bias (valores aprendidos no treinamento)
// Camada densa 1: 10 entradas x 8 neurônios
float dense_1_weights[10][8] = {
    {0.12, -0.34, 0.56, -0.21, 0.18, 0.09, -0.07, 0.01},
    {0.05,  0.11, -0.12, 0.17, -0.05, 0.06, -0.03, 0.02},
    {-0.08, 0.14, 0.09, -0.06, 0.03, -0.01, 0.10, -0.04},
    {0.02, -0.07, 0.04, 0.06, 0.08, -0.02, 0.01, 0.05},
    {-0.09, 0.03, 0.07, -0.08, 0.04, 0.12, 0.00, -0.06},
    {0.06, 0.01, -0.05, 0.10, -0.07, 0.02, 0.11, 0.03},
    {-0.04, 0.08, 0.05, -0.01, 0.09, 0.00, -0.02, 0.07},
    {0.03, -0.06, 0.02, 0.01, 0.07, -0.03, 0.04, 0.00},
    {0.00, 0.10, -0.09, 0.03, 0.06, 0.05, -0.08, 0.09},
    {-0.02, 0.07, 0.03, -0.04, 0.11, -0.05, 0.06, -0.01}
};

float dense_1_bias[8] = {0.1, 0.0, -0.1, 0.05, 0.02, 0.03, -0.04, 0.01};

// Camada densa 2: 8 entradas x 2 saídas
float dense_2_weights[8][2] = {
    {0.25, -0.35},
    {0.12, 0.07},
    {-0.18, 0.22},
    {0.05, -0.11},
    {0.14, 0.09},
    {-0.06, 0.03},
    {0.08, -0.02},
    {0.11, 0.10}
};

float dense_2_bias[2] = {0.01, -0.01};

// 3. Funções de ativação
float relu(float x) {
    return x > 0 ? x : 0;
}

void softmax(const float* input, float* output, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

// 4. Informações sobre entrada e saída do modelo
struct ModelIO {
    int input_shape = 10;
    int output_shape = 2;
    std::string input_dtype = "float32";
    std::string output_dtype = "float32";
};

ModelIO model_info;

// Observação:
// O modelo .tflite é convertido para array binário (.h) e interpretado por bibliotecas como
// TensorFlow Lite Micro. Aqui simulamos manualmente para entendimento da estrutura lógica
// envolvida em uma inferência embarcada real.
