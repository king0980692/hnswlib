#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include "../hnswlib/hnswlib.h"

// 讀取單一行向量（query embedding）的函式
std::vector<float> read_embedding(const std::string &filepath) {
    std::ifstream infile(filepath);
    if (!infile) {
        throw std::runtime_error("Error: Cannot open query embedding file: " + filepath);
    }
    std::string line;
    if (!std::getline(infile, line)) {
        throw std::runtime_error("Error: Query embedding file is empty: " + filepath);
    }
    infile.close();

    std::vector<float> vec;
    std::istringstream iss(line);
    std::string token;
    if (line.find(',') != std::string::npos) {
        while (std::getline(iss, token, ',')) {
            std::istringstream tokenStream(token);
            float value;
            if (!(tokenStream >> value)) {
                throw std::runtime_error("Error parsing float value in query file.");
            }
            vec.push_back(value);
        }
    } else {
        float value;
        while (iss >> value) {
            vec.push_back(value);
        }
    }
    return vec;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <index_file> <query_embedding_file> [topK]\n";
        return 1;
    }

    std::string indexFile = argv[1];
    std::string queryFile = argv[2];
    int topK = 10;
    if (argc >= 4) {
        topK = std::stoi(argv[3]);
    }

    // 讀取 query embedding
    std::vector<float> query;
    try {
        query = read_embedding(queryFile);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    if (query.empty()) {
        std::cerr << "Error: Query embedding is empty.\n";
        return 1;
    }
    int embDim = query.size();
    /* std::cout << "Loaded query embedding (dimension " << embDim << ").\n"; */

    // 使用 InnerProductSpace 建立相同維度的空間
    hnswlib::InnerProductSpace space(embDim);

    // 反序列化索引，讀取已儲存的 index 檔案
    hnswlib::HierarchicalNSW<float>* index;
    try {
        index = new hnswlib::HierarchicalNSW<float>(&space, indexFile);
    } catch (const std::exception &e) {
        std::cerr << "Error loading index: " << e.what() << "\n";
        return 1;
    }
    /* std::cout << "Index loaded from " << indexFile << ".\n"; */

    // 執行 topK 搜尋
    auto result = index->searchKnn(query.data(), topK);

    // 從 priority_queue 中取出結果（結果排序從高到低，需反轉）
    std::vector<std::pair<float, hnswlib::labeltype>> results;
    while (!result.empty()) {
        /* results.push_back(result.top()); */
        const auto &elements = result.top();
        /* std::cout << "Label: " << elements.second << ", Similarity: " << elements.first - 1 << "\n"; */
        std::cout << elements.second << "\n";
        result.pop();
    }
    /* std::reverse(results.begin(), results.end()); */

    // 輸出結果：每筆結果顯示 label 與內積相似度
    /* std::cout << "Top " << topK << " nearest neighbors (inner product similarity):\n"; */
    /* for (const auto &p : results) { */
    /*     std::cout << "Label: " << p.second << ", Similarity: " << p.first - 1 << "\n"; */
    /* } */

    delete index;
    return 0;
}

