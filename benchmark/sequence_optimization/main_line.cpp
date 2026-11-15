#include "lib/optimized.hpp"
#include "lib/nonoptimize.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace benchmark
{
    constexpr std::size_t BranchCount = 4;
    constexpr std::size_t NodesPerBranch = 32;
    constexpr std::size_t NodeCount = BranchCount * NodesPerBranch; // 128 nodes

    constexpr unsigned WorkerCount = 4;
    constexpr std::size_t WarmupCount = 2;
    constexpr std::size_t SampleCount = 30;
    constexpr std::uint64_t WorkIterationsPerNode = 4096;
    constexpr std::uint64_t Seed = 0xC0FFEE1234567890ULL;
    constexpr std::uint64_t Unwritten = std::numeric_limits<std::uint64_t>::max();

    using Clock = std::chrono::steady_clock;

    struct State
    {
        explicit State(std::uint64_t initialSeed) : seed(initialSeed)
        {
            for (auto& value : output)
            {
                value.store(Unwritten, std::memory_order_relaxed);
            }
        }

        const std::uint64_t seed;
        std::array<std::atomic<std::uint64_t>, NodeCount> output;
    };

    // Deterministic seed expansion without using random_device or changing the seed between runs.
    std::uint64_t SplitMix64(std::uint64_t value) noexcept
    {
        value += 0x9E3779B97F4A7C15ULL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    }

    std::string NodeName(std::size_t branch, std::size_t depth)
    {
        return "B" + std::to_string(branch) + "_N" + std::to_string(depth);
    }

    template <class GraphT>
    void BuildGraph(GraphT& graph, std::uint64_t workIterations)
    {
        using NodeFunction = std::function<std::string(std::shared_ptr<State>)>;

        // Add all nodes first, then create edges, following the same pattern as test.cpp.
        for (std::size_t branch = 0; branch < BranchCount; ++branch)
        {
            for (std::size_t depth = 0; depth < NodesPerBranch; ++depth)
            {
                const std::size_t nodeIndex = branch * NodesPerBranch + depth;
                const std::string name = NodeName(branch, depth);

                NodeFunction task = [nodeIndex, workIterations](std::shared_ptr<State> state) -> std::string
                {
                    // Each node performs its own computation and writes the result to a unique slot.
                    std::uint64_t value = SplitMix64(
                        state->seed + static_cast<std::uint64_t>(nodeIndex) * 0x9E3779B97F4A7C15ULL);

                    // Perform deterministic computation with a loop-carried dependency.
                    // The result is stored and verified after graph.wait().
                    for (std::uint64_t i = 0; i < workIterations; ++i)
                    {
                        value ^= value >> 12;
                        value ^= value << 25;
                        value ^= value >> 27;
                        value *= 0x2545F4914F6CDD1DULL;
                    }

                    state->output[nodeIndex].store(value, std::memory_order_relaxed);
                    return {}; // Use static edges only; no conditional or cycle edges.
                };

                graph.addNode(name, std::move(task));
            }
        }

        // Graph structure: START forks into 4 branches, each containing a sequential chain of 32 nodes.
        for (std::size_t branch = 0; branch < BranchCount; ++branch)
        {
            graph.addEdge(graph.START, NodeName(branch, 0));

            for (std::size_t depth = 0; depth + 1 < NodesPerBranch; ++depth)
            {
                graph.addEdge(NodeName(branch, depth), NodeName(branch, depth + 1));
            }

            graph.addEdge(NodeName(branch, NodesPerBranch - 1), graph.END);
        }
    }

    struct RunResult
    {
        double elapsedMicroseconds = 0.0;
        std::uint64_t checksum = 0;
        bool allNodesExecuted = false;
    };

    template <class GraphT>
    RunResult RunOne(std::uint64_t workIterations)
    {
        auto state = std::make_shared<State>(Seed);
        GraphT graph(state);

        BuildGraph(graph, workIterations);
        graph.spawnWorker(WorkerCount);

        const auto start = Clock::now();
        graph.exec();
        graph.wait();
        const auto finish = Clock::now();

        // Stop workers after the benchmark timing has finished.
        graph.destroyWorker();

        RunResult result;
        result.elapsedMicroseconds =
            std::chrono::duration<double, std::micro>(finish - start).count();

        result.allNodesExecuted = true;
        result.checksum = 0x6A09E667F3BCC909ULL;

        for (const auto& slot : state->output)
        {
            const std::uint64_t value = slot.load(std::memory_order_relaxed);
            if (value == Unwritten)
            {
                result.allNodesExecuted = false;
            }

            result.checksum ^= value + 0x9E3779B97F4A7C15ULL
                             + (result.checksum << 6)
                             + (result.checksum >> 2);
        }

        return result;
    }

    double Median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        const std::size_t middle = values.size() / 2;

        if (values.size() % 2 == 0)
        {
            return (values[middle - 1] + values[middle]) / 2.0;
        }
        return values[middle];
    }

    void PrintStats(const char* name, const std::vector<double>& samples)
    {
        const auto [minimum, maximum] = std::minmax_element(samples.begin(), samples.end());
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        const double median = Median(samples);

        std::cout << name << '\n'
                  << "  min    : " << *minimum << " us\n"
                  << "  median : " << median << " us\n"
                  << "  mean   : " << mean << " us\n"
                  << "  max    : " << *maximum << " us\n";
    }
}

int main()
{
    using AriadneGraph = Ariadne::Graph<benchmark::State>;
    using AriadneOpGraph = AriadneOp::Graph<benchmark::State>;

    std::cout << "Ariadne vs AriadneOp benchmark\n"
              << "Graph       : " << benchmark::BranchCount << " branches x "
              << benchmark::NodesPerBranch << " sequential nodes = "
              << benchmark::NodeCount << " nodes\n"
              << "Workers     : " << benchmark::WorkerCount << '\n'
              << "Work/node   : " << benchmark::WorkIterationsPerNode << " iterations\n"
              << "Samples     : " << benchmark::SampleCount << '\n'
              << "Seed        : 0x" << std::hex << benchmark::Seed << std::dec << "\n\n";

    // Warm up both implementations before collecting benchmark results.
    for (std::size_t i = 0; i < benchmark::WarmupCount; ++i)
    {
        if (i % 2 == 0)
        {
            (void)benchmark::RunOne<AriadneGraph>(benchmark::WorkIterationsPerNode);
            (void)benchmark::RunOne<AriadneOpGraph>(benchmark::WorkIterationsPerNode);
        }
        else
        {
            (void)benchmark::RunOne<AriadneOpGraph>(benchmark::WorkIterationsPerNode);
            (void)benchmark::RunOne<AriadneGraph>(benchmark::WorkIterationsPerNode);
        }
    }

    std::vector<double> ariadneSamples;
    std::vector<double> ariadneOpSamples;
    ariadneSamples.reserve(benchmark::SampleCount);
    ariadneOpSamples.reserve(benchmark::SampleCount);

    std::uint64_t expectedChecksum = 0;
    bool hasExpectedChecksum = false;

    // Alternate the execution order in each sample to reduce ordering bias.
    for (std::size_t i = 0; i < benchmark::SampleCount; ++i)
    {
        benchmark::RunResult ariadneResult;
        benchmark::RunResult ariadneOpResult;

        if (i % 2 == 0)
        {
            ariadneResult = benchmark::RunOne<AriadneGraph>(benchmark::WorkIterationsPerNode);
            ariadneOpResult = benchmark::RunOne<AriadneOpGraph>(benchmark::WorkIterationsPerNode);
        }
        else
        {
            ariadneOpResult = benchmark::RunOne<AriadneOpGraph>(benchmark::WorkIterationsPerNode);
            ariadneResult = benchmark::RunOne<AriadneGraph>(benchmark::WorkIterationsPerNode);
        }

        if (!ariadneResult.allNodesExecuted || !ariadneOpResult.allNodesExecuted)
        {
            std::cerr << "ERROR: at least one graph did not execute all "
                      << benchmark::NodeCount << " nodes\n";
            return 2;
        }

        if (ariadneResult.checksum != ariadneOpResult.checksum)
        {
            std::cerr << "ERROR: checksum mismatch between implementations\n"
                      << "Ariadne   : 0x" << std::hex << ariadneResult.checksum << '\n'
                      << "AriadneOp : 0x" << ariadneOpResult.checksum << std::dec << '\n';
            return 3;
        }

        if (!hasExpectedChecksum)
        {
            expectedChecksum = ariadneResult.checksum;
            hasExpectedChecksum = true;
        }
        else if (ariadneResult.checksum != expectedChecksum)
        {
            std::cerr << "ERROR: checksum changed between samples\n";
            return 4;
        }

        ariadneSamples.push_back(ariadneResult.elapsedMicroseconds);
        ariadneOpSamples.push_back(ariadneOpResult.elapsedMicroseconds);
    }

    std::cout << std::fixed << std::setprecision(2);
    benchmark::PrintStats("Ariadne", ariadneSamples);
    std::cout << '\n';
    benchmark::PrintStats("AriadneOp", ariadneOpSamples);

    const double ariadneMedian = benchmark::Median(ariadneSamples);
    const double ariadneOpMedian = benchmark::Median(ariadneOpSamples);

    std::cout << "\nAriadneOp speedup: " << (ariadneMedian / ariadneOpMedian) << "x\n"
              << "Checksum         : 0x" << std::hex << expectedChecksum << std::dec << '\n';

    return 0;
}