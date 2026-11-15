#include "lib/optimized.hpp"
#include "lib/nonoptimize.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
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
    // The graph consists of multiple sequential lanes connected by random forward edges.
    // Every edge points from a lower depth to a higher depth, ensuring the graph is a DAG with no cycles.
    constexpr std::size_t LaneCount = 10;
    constexpr std::size_t NodeCount = 240;
    constexpr std::size_t MinNodesPerLane = 12;

    constexpr std::size_t MaxRandomEdgesPerNode = 4;
    constexpr std::size_t MaxForwardJump = 5;
    constexpr std::uint64_t MinWorkIterations = 512;
    constexpr std::uint64_t MaxWorkIterations = 4096;

    constexpr unsigned WorkerCount = 4;
    constexpr std::size_t WarmupCount = 2;
    constexpr std::size_t SampleCount = 30;
    constexpr std::uint64_t Seed = 0xC0FFEE1234567890ULL;
    constexpr std::uint64_t Unwritten = std::numeric_limits<std::uint64_t>::max();

    using Clock = std::chrono::steady_clock;

    struct State
    {
        State()
        {
            for (auto& value : output)
            {
                value.store(Unwritten, std::memory_order_relaxed);
            }
        }

        std::array<std::atomic<std::uint64_t>, NodeCount> output;
    };

    std::uint64_t SplitMix64(std::uint64_t value) noexcept
    {
        value += 0x9E3779B97F4A7C15ULL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    }

    // A deterministic PRNG implementation that does not depend on the standard library's distribution algorithms.
    // The same seed always produces the same graph structure and workload.
    class DeterministicRng
    {
    public:
        explicit DeterministicRng(std::uint64_t seed) : state(seed) {}

        std::uint64_t next() noexcept
        {
            state += 0x9E3779B97F4A7C15ULL;
            return SplitMix64(state - 0x9E3779B97F4A7C15ULL);
        }

        std::uint64_t bounded(std::uint64_t bound) noexcept
        {
            return bound == 0 ? 0 : next() % bound;
        }

    private:
        std::uint64_t state;
    };

    struct NodeSpec
    {
        std::uint64_t workIterations = 0;
        std::uint64_t inputSeed = 0;
    };

    struct Edge
    {
        std::size_t from;
        std::size_t to;
    };

    struct GraphSpec
    {
        std::array<NodeSpec, NodeCount> nodes{};
        std::array<std::size_t, LaneCount> laneOffsets{};
        std::array<std::size_t, LaneCount> laneLengths{};
        std::vector<std::size_t> startNodes;
        std::vector<std::size_t> endNodes;
        std::vector<Edge> edges;
        std::size_t sequentialEdgeCount = 0;
        std::size_t randomEdgeCount = 0;
    };

    constexpr std::size_t NodeIndex(const GraphSpec& spec,
                                    std::size_t lane,
                                    std::size_t depth) noexcept
    {
        return spec.laneOffsets[lane] + depth;
    }

    std::string NodeName(std::size_t index)
    {
        return "Task" + std::to_string(index);
    }

    void AddUniqueEdge(GraphSpec& spec,
                       std::vector<std::vector<std::size_t>>& outgoing,
                       std::size_t from,
                       std::size_t to,
                       bool randomEdge)
    {
        auto& targets = outgoing[from];
        if (std::find(targets.begin(), targets.end(), to) != targets.end())
        {
            return;
        }

        targets.push_back(to);
        spec.edges.push_back({from, to});
        if (randomEdge)
        {
            ++spec.randomEdgeCount;
        }
        else
        {
            ++spec.sequentialEdgeCount;
        }
    }

    GraphSpec GenerateGraph()
    {
        DeterministicRng rng(Seed);
        GraphSpec spec;
        std::vector<std::vector<std::size_t>> outgoing(NodeCount);

        // Randomize the length of each sequential lane while ensuring that every lane has at least 12 nodes.
        // The total node count remains 240, but each lane can have a different length.
        spec.laneLengths.fill(MinNodesPerLane);
        for (std::size_t i = LaneCount * MinNodesPerLane; i < NodeCount; ++i)
        {
            ++spec.laneLengths[rng.bounded(LaneCount)];
        }

        std::size_t offset = 0;
        for (std::size_t lane = 0; lane < LaneCount; ++lane)
        {
            spec.laneOffsets[lane] = offset;
            offset += spec.laneLengths[lane];
        }

        // Randomize the workload and input seed of each node using the same PRNG seed.
        for (std::size_t i = 0; i < NodeCount; ++i)
        {
            spec.nodes[i].workIterations = MinWorkIterations
                + rng.bounded(MaxWorkIterations - MinWorkIterations + 1);
            spec.nodes[i].inputSeed = rng.next();
        }

        // Build the backbone as 10 sequential lanes with varying lengths.
        // This guarantees that every node is reachable from START and can eventually reach END.
        for (std::size_t lane = 0; lane < LaneCount; ++lane)
        {
            const std::size_t laneLength = spec.laneLengths[lane];
            spec.startNodes.push_back(NodeIndex(spec, lane, 0));
            spec.endNodes.push_back(NodeIndex(spec, lane, laneLength - 1));

            for (std::size_t depth = 0; depth + 1 < laneLength; ++depth)
            {
                AddUniqueEdge(spec, outgoing,
                              NodeIndex(spec, lane, depth),
                              NodeIndex(spec, lane, depth + 1),
                              false);
            }
        }

        // Add random edges to introduce forks, joins, cross-branch dependencies,
        // and edges that skip multiple depth levels. All edges point forward.
        for (std::size_t lane = 0; lane < LaneCount; ++lane)
        {
            const std::size_t laneLength = spec.laneLengths[lane];
            for (std::size_t depth = 0; depth + 1 < laneLength; ++depth)
            {
                const std::size_t from = NodeIndex(spec, lane, depth);
                const std::size_t wanted = rng.bounded(MaxRandomEdgesPerNode + 1);

                std::size_t added = 0;
                const std::size_t maxAttempts = wanted * 8 + 8;
                for (std::size_t attempt = 0; attempt < maxAttempts && added < wanted; ++attempt)
                {
                    const std::size_t targetLane = rng.bounded(LaneCount);
                    const std::size_t targetLength = spec.laneLengths[targetLane];
                    if (targetLength <= depth + 1)
                    {
                        continue;
                    }

                    const std::size_t remainingDepths = targetLength - 1 - depth;
                    const std::size_t maxJump = std::min(MaxForwardJump, remainingDepths);
                    const std::size_t jump = 1 + rng.bounded(maxJump);
                    const std::size_t to = NodeIndex(spec, targetLane, depth + jump);
                    const std::size_t previousCount = spec.edges.size();

                    AddUniqueEdge(spec, outgoing, from, to, true);
                    if (spec.edges.size() != previousCount)
                    {
                        ++added;
                    }
                }
            }
        }

        return spec;
    }

    template <class GraphT>
    void BuildGraph(GraphT& graph, const GraphSpec& spec)
    {
        using NodeFunction = std::function<std::string(std::shared_ptr<State>)>;

        for (std::size_t i = 0; i < NodeCount; ++i)
        {
            const NodeSpec node = spec.nodes[i];

            NodeFunction task = [i, node](std::shared_ptr<State> state) -> std::string
            {
                std::uint64_t value = SplitMix64(node.inputSeed ^
                    (static_cast<std::uint64_t>(i) * 0xD6E8FEB86659FD93ULL));

                // The loop-carried data dependency prevents the compiler from eliminating the computation.
                for (std::uint64_t iteration = 0;
                     iteration < node.workIterations;
                     ++iteration)
                {
                    value ^= value >> 12;
                    value ^= value << 25;
                    value ^= value >> 27;
                    value *= 0x2545F4914F6CDD1DULL;
                }

                // Each node writes to a unique atomic slot; the checksum is calculated after wait().
                if (value == Unwritten)
                {
                    value ^= 1;
                }
                state->output[i].store(value, std::memory_order_relaxed);
                return {};
            };

            graph.addNode(NodeName(i), std::move(task));
        }

        for (const std::size_t node : spec.startNodes)
        {
            graph.addEdge(graph.START, NodeName(node));
        }

        for (const Edge& edge : spec.edges)
        {
            graph.addEdge(NodeName(edge.from), NodeName(edge.to));
        }

        for (const std::size_t node : spec.endNodes)
        {
            graph.addEdge(NodeName(node), graph.END);
        }
    }

    struct RunResult
    {
        double elapsedMicroseconds = 0.0;
        std::uint64_t checksum = 0;
        bool allNodesExecuted = false;
    };

    template <class GraphT>
    RunResult RunOne(const GraphSpec& spec)
    {
        auto state = std::make_shared<State>();
        GraphT graph(state);

        BuildGraph(graph, spec);
        graph.spawnWorker(WorkerCount);

        const auto start = Clock::now();
        graph.exec();
        graph.wait();
        const auto finish = Clock::now();

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

        std::cout << name << '\n'
                  << "  min    : " << *minimum << " us\n"
                  << "  median : " << Median(samples) << " us\n"
                  << "  mean   : " << mean << " us\n"
                  << "  max    : " << *maximum << " us\n";
    }

    bool VerifyResult(const RunResult& result, std::uint64_t expectedChecksum,
                      bool hasExpectedChecksum, const char* implementation)
    {
        if (!result.allNodesExecuted)
        {
            std::cerr << "ERROR: " << implementation << " did not execute all nodes\n";
            return false;
        }

        if (hasExpectedChecksum && result.checksum != expectedChecksum)
        {
            std::cerr << "ERROR: checksum changed for " << implementation << '\n';
            return false;
        }
        return true;
    }
}

int main()
{
    using AriadneGraph = Ariadne::Graph<benchmark::State>;
    using AriadneOpGraph = AriadneOp::Graph<benchmark::State>;

    // Generate the graph only once and use the same GraphSpec for both implementations.
    const benchmark::GraphSpec graphSpec = benchmark::GenerateGraph();

    std::cout << "Ariadne vs AriadneOp benchmark\n"
              << "Nodes             : " << benchmark::NodeCount << '\n'
              << "Sequential lanes : " << benchmark::LaneCount << '\n'
              << "Lane lengths      : ";
    for (std::size_t lane = 0; lane < benchmark::LaneCount; ++lane)
    {
        std::cout << graphSpec.laneLengths[lane]
                  << (lane + 1 == benchmark::LaneCount ? '\n' : ',');
    }
    std::cout << "Sequential edges  : " << graphSpec.sequentialEdgeCount << '\n'
              << "Random edges      : " << graphSpec.randomEdgeCount << '\n'
              << "Total node edges  : " << graphSpec.edges.size() << '\n'
              << "Workers           : " << benchmark::WorkerCount << '\n'
              << "Work/node         : random [" << benchmark::MinWorkIterations
              << ", " << benchmark::MaxWorkIterations << "] iterations\n"
              << "Warm-up           : " << benchmark::WarmupCount << '\n'
              << "Samples           : " << benchmark::SampleCount << '\n'
              << "Seed              : 0x" << std::hex << benchmark::Seed << std::dec << "\n\n";

    std::uint64_t expectedChecksum = 0;
    bool hasExpectedChecksum = false;

    // Warm up both implementations and verify that they produce identical results.
    for (std::size_t i = 0; i < benchmark::WarmupCount; ++i)
    {
        benchmark::RunResult a;
        benchmark::RunResult b;

        if (i % 2 == 0)
        {
            a = benchmark::RunOne<AriadneGraph>(graphSpec);
            b = benchmark::RunOne<AriadneOpGraph>(graphSpec);
        }
        else
        {
            b = benchmark::RunOne<AriadneOpGraph>(graphSpec);
            a = benchmark::RunOne<AriadneGraph>(graphSpec);
        }

        if (!a.allNodesExecuted || !b.allNodesExecuted || a.checksum != b.checksum)
        {
            std::cerr << "ERROR: warm-up validation failed\n";
            return 2;
        }

        if (!hasExpectedChecksum)
        {
            expectedChecksum = a.checksum;
            hasExpectedChecksum = true;
        }
        else if (a.checksum != expectedChecksum)
        {
            std::cerr << "ERROR: checksum changed during warm-up\n";
            return 3;
        }
    }

    std::vector<double> ariadneSamples;
    std::vector<double> ariadneOpSamples;
    ariadneSamples.reserve(benchmark::SampleCount);
    ariadneOpSamples.reserve(benchmark::SampleCount);

    // Alternate the execution order in each sample to reduce ordering bias.
    for (std::size_t i = 0; i < benchmark::SampleCount; ++i)
    {
        benchmark::RunResult a;
        benchmark::RunResult b;

        if (i % 2 == 0)
        {
            a = benchmark::RunOne<AriadneGraph>(graphSpec);
            b = benchmark::RunOne<AriadneOpGraph>(graphSpec);
        }
        else
        {
            b = benchmark::RunOne<AriadneOpGraph>(graphSpec);
            a = benchmark::RunOne<AriadneGraph>(graphSpec);
        }

        if (!benchmark::VerifyResult(a, expectedChecksum, hasExpectedChecksum, "Ariadne") ||
            !benchmark::VerifyResult(b, expectedChecksum, hasExpectedChecksum, "AriadneOp"))
        {
            return 4;
        }

        if (a.checksum != b.checksum)
        {
            std::cerr << "ERROR: checksum mismatch between Ariadne and AriadneOp\n";
            return 5;
        }

        ariadneSamples.push_back(a.elapsedMicroseconds);
        ariadneOpSamples.push_back(b.elapsedMicroseconds);
    }

    std::cout << std::fixed << std::setprecision(2);
    benchmark::PrintStats("Ariadne", ariadneSamples);
    std::cout << '\n';
    benchmark::PrintStats("AriadneOp", ariadneOpSamples);

    const double ariadneMedian = benchmark::Median(ariadneSamples);
    const double ariadneOpMedian = benchmark::Median(ariadneOpSamples);

    std::cout << "\nAriadneOp speedup : " << (ariadneMedian / ariadneOpMedian) << "x\n"
              << "Checksum          : 0x" << std::hex << expectedChecksum << std::dec << '\n';

    return 0;
}