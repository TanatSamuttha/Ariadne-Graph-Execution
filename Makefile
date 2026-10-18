SRC := ./src
BENCH := ./benchmark
LANGGRAPHBENCH := $(BENCH)/langgraph

compiletest:
	g++ test/test.cpp -o exec -O0 -I$(SRC)

compilelgbench:
	g++ $(LANGGRAPHBENCH)/benchmark.cpp -o exec -O3 -I$(SRC) -s

compilemultibench:
	g++ $(BENCH)/multiworkers.cpp -o exec -O3 -I$(SRC) -s