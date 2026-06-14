Flags = -Wpedantic -pedantic-errors -std=c++23
CPP = g++

GRAPHVIZ_CFLAGS = $(shell pkg-config --cflags libgvc libcgraph)
GRAPHVIZ_LIBS   = $(shell pkg-config --libs libgvc libcgraph)

autograd: engine.o main.o
	$(CPP) $(Flags) -o autograd engine.o main.o $(GRAPHVIZ_LIBS)

engine.o: engine.cpp
	$(CPP) -c $(Flags) $(GRAPHVIZ_CFLAGS) engine.cpp -o engine.o

main.o: main.cpp
	$(CPP) -c $(Flags) main.cpp -o main.o

clean:
	rm -f *.o autograd