#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "autograd/engine.hpp"

void trace(ValuePtr root, std::unordered_set<ValuePtr>& nodes,
           std::vector<std::pair<ValuePtr, ValuePtr>>& edges) {
  if (nodes.find(root) == nodes.end()) {
    nodes.insert(root);
    for (const auto& child : root->_prev) {
      edges.push_back({child, root});
      trace(child, nodes, edges);
    }
  }
}

void drawDot(ValuePtr root, std::string rankdir = "LR") {
  std::unordered_set<ValuePtr> nodes;
  std::vector<std::pair<ValuePtr, ValuePtr>> edges;

  trace(root, nodes, edges);

  std::ofstream dotFile("graph.dot");
  if (!dotFile.is_open()) {
    std::cerr << "Failed to open dot file for writing" << std::endl;
    return;
  }

  dotFile << "digraph G{\n";
  dotFile << " rankdir=" << rankdir << ";\n";

  for (const auto& n : nodes) {
    std::stringstream ss_id;
    ss_id << "node_" << n.get();
    std::string n_id = ss_id.str();

    std::stringstream ss_label;
    ss_label << std::fixed << std::setprecision(4) << "{ data " << n->data
             << " | grad " << n->grad << " }";
    dotFile << "    " << n_id << " [label=\"" << ss_label.str()
            << "\", shape=record];\n";

    if (!n->_op.empty()) {
      std::string op_id = n_id + "_op";
      dotFile << "    " << op_id << " [label=\"" << n->_op << "\"];\n";
      dotFile << "    " << op_id << " -> " << n_id << ";\n";
    }
  }
  for (const auto& edge : edges) {
    std::stringstream ss_n1, ss_n2;
    ss_n1 << "node_" << edge.first.get();
    ss_n2 << "node_" << edge.second.get();

    dotFile << "    " << ss_n1.str() << " -> " << ss_n2.str() << "_op;\n";
  }

  dotFile << "}\n";
  dotFile.close();

  std::string command = "dot -Tsvg graph.dot -o graph.svg";
  std::system(command.c_str());
}