#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "../hnswlib/hnswlib.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <corpus_embedding_file> [index_output_file]\n";
        return 1;
    }

    std::string embedFilePath = argv[1];
    std::string indexOutputPath = (argc >= 3) ? argv[2] : "hnsw_innerproduct_index.bin";

    // 開啟 embeddings 檔案
    std::ifstream infile(embedFilePath);
    if (!infile) {
        std::cerr << "Error: Cannot open embedding file: " << embedFilePath << "\n";
        return 1;
    }

    std::vector<std::vector<float>> embeddings;
    std::string line;
    int lineNum = 0;
    int embDim = 0;  // 自動偵測的向量維度

    while (std::getline(infile, line)) {
        ++lineNum;
        if (line.empty()) continue;  // 略過空行

        std::vector<float> vec;
        std::istringstream iss(line);
        std::string token;
        // 檢查是否包含逗號，若有則以逗號分隔，否則以空白分隔
        if (line.find(',') != std::string::npos) {
            while (std::getline(iss, token, ',')) {
                std::istringstream tokenStream(token);
                float value;
                if (!(tokenStream >> value)) {
                    std::cerr << "Error: Failed to parse a float value in line " << lineNum << "\n";
                    return 1;
                }
                vec.push_back(-value);
            }
        } else {
            float value;
            while (iss >> value) {
                vec.push_back(-value);
            }
        }

        if (vec.empty()) {
            std::cerr << "Warning: Skipping empty line " << lineNum << "\n";
            continue;
        }

        // 第一行自動偵測維度
        if (embeddings.empty()) {
            embDim = vec.size();
            std::cout << "Detected embedding dimension: " << embDim << "\n";
        } else if (vec.size() != static_cast<size_t>(embDim)) {
            std::cerr << "Error: Line " << lineNum << " does not have " << embDim
                      << " elements. Found " << vec.size() << " elements.\n";
            return 1;
        }
        embeddings.push_back(vec);
    }
    infile.close();

    int max_elements = embeddings.size();
    std::cout << "Read " << max_elements << " embeddings from file.\n";

    // 建立 HNSW 索引 (使用 InnerProductSpace，內積作為相似度度量)
    hnswlib::InnerProductSpace space(embDim);
    int M = 16;               // HNSW 連接參數
    int ef_construction = 200; // 建構索引時使用的參數
    hnswlib::HierarchicalNSW<float> index(&space, max_elements, M, ef_construction);

    // 將每個向量加入索引（直接使用原始向量，不做正規化）
    for (int i = 0; i < max_elements; i++) {
        index.addPoint(embeddings[i].data(), i);
    }

    // 儲存索引
    index.saveIndex(indexOutputPath);
    std::cout << "Index saved to " << indexOutputPath << "\n";

    return 0;
}

