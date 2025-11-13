INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)_acc
SRC = $(BENCHMARK).c
HEADERS = $(BENCHMARK).h

SRC += $(UTIL_DIR)/polybench.c

DEPS        := Makefile.dep
DEP_FLAG    := -MM

#CC=clang
CC=gcc
LD=ld
OBJDUMP=objdump

#OPT=-O2 -pg -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
OPT=-O2 -pg -fopenmp
CFLAGS=$(OPT) -I. $(EXT_CFLAGS)
LDFLAGS=-lm $(EXT_LDFLAGS)

.PHONY: all exe clean veryclean

all : exe

exe : $(EXE)

$(EXE) : $(SRC)
	$(CC) $(CFLAGS) $(INCPATHS) $^ -o $@ $(LDFLAGS)

clean :
	-rm -vf -vf $(EXE) *~ 

veryclean : clean
	-rm -vf $(DEPS)

run: $(EXE)
	./$(EXE)

profile_val: $(EXE)
	valgrind --tool=cachegrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes ./$(EXE) $(EXT_ARGS)

profile_gprof: $(EXE)
	gprof ./$(EXE)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)