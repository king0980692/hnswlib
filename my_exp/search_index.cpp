#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include "../hnswlib/hnswlib.h"

// 讀取多行向量（corpus embedding）的函式，並自動取得向量維度
std::vector<std::vector<float>> read_embeddings(const std::string &filepath, int &embDim) {
    std::ifstream infile(filepath);
    if (!infile) {
        throw std::runtime_error("Error: Cannot open file: " + filepath);
    }
    std::vector<std::vector<float>> embeddings;
    std::string line;
    int lineNum = 0;
    while (std::getline(infile, line)) {
        ++lineNum;
        if (line.empty()) continue;
        std::vector<float> vec;
        std::istringstream iss(line);
        std::string token;
        // 如果包含逗號，則以逗號作分隔；否則以空白分隔
        if (line.find(',') != std::string::npos) {
            while (std::getline(iss, token, ',')) {
                std::istringstream tokenStream(token);
                float value;
                if (!(tokenStream >> value)) {
                    throw std::runtime_error("Error parsing float on line " + std::to_string(lineNum));
                }
                vec.push_back(-value);
            }
        } else {
            float value;
            while (iss >> value) {
                vec.push_back(-value);
            }
        }
        if (embeddings.empty()) {
            embDim = vec.size();
        } else if (vec.size() != static_cast<size_t>(embDim)) {
            throw std::runtime_error("Error: Inconsistent embedding dimension on line " + std::to_string(lineNum));
        }
        embeddings.push_back(vec);
    }
    return embeddings;
}

// 讀取單一行向量（query embedding）的函式
std::vector<float> read_embedding(const std::string &filepath, int expectedDim) {
    std::ifstream infile(filepath);
    if (!infile) {
        throw std::runtime_error("Error: Cannot open file: " + filepath);
    }
    std::string line;
    if (!std::getline(infile, line)) {
        throw std::runtime_error("Error: Empty file: " + filepath);
    }
    std::vector<float> vec;
    std::istringstream iss(line);
    std::string token;
    if (line.find(',') != std::string::npos) {
        while (std::getline(iss, token, ',')) {
            std::istringstream tokenStream(token);
            float value;
            if (!(tokenStream >> value)) {
                throw std::runtime_error("Error parsing float in query file.");
            }
            vec.push_back(value);
        }
    } else {
        float value;
        while (iss >> value) {
            vec.push_back(value);
        }
    }
    if (vec.size() != static_cast<size_t>(expectedDim)) {
        throw std::runtime_error("Error: Query embedding dimension (" + std::to_string(vec.size()) +
                                 ") does not match expected dimension (" + std::to_string(expectedDim) + ").");
    }
    return vec;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <corpus_embedding_file> <query_embedding_file> [topK]\n";
        return 1;
    }

    std::string corpusFile = argv[1];
    std::string queryFile = argv[2];
    int topK = 10;
    if (argc >= 4) {
        topK = std::stoi(argv[3]);
    }

    int embDim = 0;
    std::vector<std::vector<float>> corpusEmbeddings;
    try {
        corpusEmbeddings = read_embeddings(corpusFile, embDim);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    /* std::cout << "Loaded " << corpusEmbeddings.size() << " corpus embeddings (dimension " << embDim << ").\n"; */

    std::vector<float> queryEmbedding;
    try {
        queryEmbedding = read_embedding(queryFile, embDim);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    /* std::cout << "Loaded query embedding.\n"; */

    // 使用 InnerProductSpace (內積) 建立索引
    hnswlib::InnerProductSpace space(embDim);
    size_t max_elements = corpusEmbeddings.size();
    int M = 16;               // HNSW 參數：連接數
    int ef_construction = 200; // HNSW 建立索引時的參數
    hnswlib::HierarchicalNSW<float> index(&space, max_elements, M, ef_construction);

    // 將所有 corpus embedding 加入索引
    for (size_t i = 0; i < max_elements; i++) {
        index.addPoint(corpusEmbeddings[i].data(), i);
    }
    /* std::cout << "HNSW index built.\n"; */

    // 用 query embedding 進行 topK 搜尋
    auto result = index.searchKnn(queryEmbedding.data(), topK);

    // 將結果取出（priority_queue 中的結果由大到小，需反轉）
    std::vector<std::pair<float, hnswlib::labeltype>> results;
    while (!result.empty()) {
        results.push_back(result.top());
        result.pop();
    }
    /* std::reverse(results.begin(), results.end()); */

    /* std::cout << "Top " << topK << " nearest neighbors (inner product similarity):\n"; */
    for (const auto &p : results) {
        /* std::cout << "[" << p.second << "]" << " Similarity: " << -1+p.first << "\n"; */
        std::cout << p.second << std::endl;
    }

    return 0;
}
