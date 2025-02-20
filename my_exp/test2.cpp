#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <cmath>
#include "../hnswlib/hnswlib.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <index_bin_file> <query_embedding_file_path> [topK]\n";
        std::cout << "The query file should contain one line of embedding values separated by commas or spaces.\n";
        return 1;
    }

    // 取得命令列參數
    std::string index_path = argv[1];
    std::string query_file_path = argv[2];
    int topK = 10;
    if (argc >= 4) {
        topK = std::stoi(argv[3]);
    }

    // 開啟並讀取 query embedding 檔案的第一行
    std::ifstream qfile(query_file_path);
    if (!qfile) {
        std::cerr << "Error: Cannot open query embedding file: " << query_file_path << "\n";
        return 1;
    }
    std::string query_line;
    if (!std::getline(qfile, query_line)) {
        std::cerr << "Error: Query embedding file is empty: " << query_file_path << "\n";
        return 1;
    }
    qfile.close();

    // 解析 query_line，假設以逗號分隔；若失敗則以空白分隔
    std::vector<float> query;
    {
        std::istringstream iss(query_line);
        std::string token;
        while (std::getline(iss, token, ',')) {
            std::istringstream tokenStream(token);
            float val;
            if (tokenStream >> val) {
                query.push_back(val);
            }
        }
        if (query.empty()) {
            std::istringstream iss_ws(query_line);
            float val;
            while (iss_ws >> val) {
                query.push_back(val);
            }
        }
    }

    if (query.empty()) {
        std::cerr << "Error: Failed to parse query embedding from file: " << query_file_path << "\n";
        return 1;
    }

    // 為了使用 cosine similarity，先將查詢向量正規化
    /* float norm = 0.0f; */
    /* for (float v : query) { */
    /*     norm += v * v; */
    /* } */
    /* norm = std::sqrt(norm); */
    /* if (norm > 0) { */
    /*     for (auto& v : query) { */
    /*         v /= norm; */
    /*     } */
    /* } else { */
    /*     std::cerr << "Error: Query vector norm is zero.\n"; */
    /*     return 1; */
    /* } */

    // 使用查詢向量的維度（請確認與索引建立時一致）
    int dim = query.size();

    // 使用 inner product 空間來模擬 cosine similarity（前提是向量皆已正規化）
    hnswlib::InnerProductSpace space(dim);

    // 反序列化索引：從 bin 檔案讀入 index
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);

    // 執行 topK 最近鄰搜尋
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(query.data(), topK);

    // 從 priority queue 中取出結果並反轉順序（由最相似到最不相似）
    std::vector<std::pair<float, hnswlib::labeltype>> results;
    while (!result.empty()) {
        results.push_back(result.top());
        result.pop();
    }
    std::reverse(results.begin(), results.end());

    // 輸出結果：每筆結果顯示 label 與 similarity（內積值，正規化後即為 cosine similarity）
    std::cout << "Top " << topK << " nearest neighbors (cosine similarity):\n";
    for (const auto& p : results) {
        std::cout << "Label: " << p.second << ", Similarity: " << p.first << "\n";
    }

    delete alg_hnsw;
    return 0;
}

