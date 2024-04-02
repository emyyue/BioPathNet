#include <tuple>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>

#include <torch/extension.h>
#include <pybind11/numpy.h>

std::mt19937 seed;

// may have bugs, may hang
std::tuple<py::array_t<int>, py::array_t<int>> get_n_hop_graph(
    const py::array_t<int> &_nodes, const py::array_t<int> &_edge_list,
    const py::array_t<int> &_order_in, const py::array_t<int> &_order_out,
    const py::array_t<int> &_degree_in, const py::array_t<int> &_degree_out,
    const py::array_t<int> &_offset_in, const py::array_t<int> &_offset_out,
    int num_hop, const std::vector<int> &num_neighbors) {
    auto nodes = _nodes.unchecked<1>();
    auto edge_list = _edge_list.unchecked<2>();
    auto order_in = _order_in.unchecked<1>();
    auto order_out = _order_out.unchecked<1>();
    auto degree_in = _degree_in.unchecked<1>();
    auto degree_out = _degree_out.unchecked<1>();
    auto offset_in = _offset_in.unchecked<1>();
    auto offset_out = _offset_out.unchecked<1>();

    std::queue<int> q;
    std::unordered_map<int, int> distance;
    std::vector<int> edge_index;
    for (int i = 0; i < nodes.shape(0); i++) {
        int node = nodes(i);
        distance[node] = 0;
        q.push(node);
    }

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        if (distance[u] == num_hop)
            continue;

        if (degree_in(u) > 0) {
            std::uniform_int_distribution<int> rand(0, degree_in(u) - 1);
            int start = rand(seed);
            int offset = offset_in(u);
            int degree = degree_in(u);
            int num_neighbor = num_neighbors[distance[u]];
            int end = start + std::min(num_neighbor, degree);
            for (int i = start; i < end; i++) {
                int index = offset + i % degree;
                index = order_in(index);
                int v = edge_list(index, 1);
                TORCH_CHECK(u == edge_list(index, 0), "Wrong edge list");
                if (distance.find(v) == distance.end()) {
                    distance[v] = distance[u] + 1;
                    q.push(v);
                }
                edge_index.push_back(index);
            }
        }

        if (degree_out(u) > 0) {
            std::uniform_int_distribution<int> rand(0, degree_out(u) - 1);
            int start = rand(seed);
            int offset = offset_out(u);
            int degree = degree_out(u);
            int num_neighbor = num_neighbors[distance[u]];
            int end = start + std::min(num_neighbor, degree);
            for (int i = start; i < end; i++) {
                int index = offset + i % degree;
                index = order_out(index);
                int v = edge_list(index, 0);
                TORCH_CHECK(u == edge_list(index, 1), "Wrong edge list");
                if (distance.find(v) == distance.end()) {
                    distance[v] = distance[u] + 1;
                    q.push(v);
                }
                edge_index.push_back(index);
           }
       }
    }

    std::set<int> edge_index_set(edge_index.begin(), edge_index.end());
    edge_index.assign(edge_index_set.begin(), edge_index_set.end());
    std::unordered_map<int, int> mapping;
    int i = 0;
    for (auto &item : distance)
        mapping[item.first] = i++;

    int num_sampled_edge = edge_index_set.size(); // type cast to int
    py::array_t<int> _sampled_edges({num_sampled_edge, 3});
    auto sampled_edges = _sampled_edges.mutable_unchecked<2>();
    for (int i = 0; i < edge_index.size(); i++) {
        int e = edge_index[i];
        sampled_edges(i, 0) = mapping[edge_list(e, 0)];
        sampled_edges(i, 1) = mapping[edge_list(e, 1)];
        sampled_edges(i, 2) = edge_list(e, 2);
    }

    py::array_t<int> _sampled_nodes(nodes.size());
    auto sampled_nodes = _sampled_nodes.mutable_unchecked<1>();
    for (int i = 0; i < nodes.size(); i++)
        sampled_nodes(i) = mapping[nodes(i)];

    return std::make_tuple(_sampled_edges, _sampled_nodes);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_n_hop_graph", &get_n_hop_graph, "get n hop graph surrounding the given nodes");
}
