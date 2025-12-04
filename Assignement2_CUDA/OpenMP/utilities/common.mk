# --- Configurazione Percorsi ---
ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

# La directory con le utility (es. polybench.c)
UTIL_DIR ?= ./util

# Il file sorgente CUDA principale (sostituire con il nome del tuo file .cu)
EXERCISE ?= exercise1.cu

BUILD_DIR ?= ./build

# --- Variabili di Compilazione ---
NVCC=$(CUDA_HOME)/bin/nvcc
CXX=g++

# Il nome dell'eseguibile finale
BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)_acc

# Opzioni di ottimizzazione e debug
OPT:=-O2 -g
NVOPT:=-Xcompiler -fopenmp -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math

# Percorsi di inclusione
INCPATHS = -I$(UTIL_DIR)

# Flag per il compilatore C/C++
CXXFLAGS:=$(OPT) -I. $(INCPATHS) $(EXT_CXXFLAGS)
# Flag per il linker C/C++ (aggiunto -lm per math, -lcudart per CUDA runtime)
LDFLAGS:=-lm -lcudart $(EXT_LDFLAGS)

# Flag per NVCC (combinati CXXFLAGS e NVOPT)
NVCFLAGS:=$(CXXFLAGS) $(NVOPT)
# Flag per il linker NVCC (aggiunto -lgomp per OpenMP)
NVLDFLAGS:=$(LDFLAGS) -lgomp

# --- File Sorgente e Oggetti ---
# Tutti i file sorgente C che devono essere compilati
SRCS:= $(UTIL_DIR)/polybench.c
# Aggiungi qui altri file .c o .cpp se necessario
# SRCS += another_file.c

# Riscrivi come definisci il tuo file oggetto CUDA
EXERCISE_OBJ := $(BUILD_DIR)/$(EXERCISE:.cu=.o)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(EXERCISE_OBJ)

	# Oggetti C/C++ e l'oggetto CUDA
#OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(BUILD_DIR)/$(EXERCISE).o

# Alias per mkdir -p
MKDIR_P ?= mkdir -p

# --- Regole di Compilazione ---
.PHONY: all exe clean veryclean run profile

all : exe

exe : $(EXE)

# Regola per la creazione dell'eseguibile finale
$(EXE):	$(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)


# Regola per compilare file CUDA (.cu)
$(BUILD_DIR)/%.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

# Regola per compilare file CUDA (.cu)
#$(BUILD_DIR)/%.cu.o: %.cu
#	$(MKDIR_P) $(dir $@)
#	$(NVCC) $(NVCFLAGS) -c $< -o $@

# Regola per compilare file C (.c)
$(BUILD_DIR)/$(UTIL_DIR)/%.c.o: $(UTIL_DIR)/%.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regola per compilare file C++ (.cpp) - aggiunta per completezza
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@


# --- Regole di Esecuzione e Pulizia ---

run: $(EXE)
	./$(EXE) $(EXT_ARGS)

profile: $(EXE)
# Utilizza nvprof per il profiling del codice CUDA
	sudo $(CUDA_HOME)/bin/nvprof ./$(EXE) $(EXT_ARGS)

clean:
	-rm -fr $(BUILD_DIR) $(EXE) *~

veryclean : clean
# Mantengo questa regola come l'avevi, anche se la gestione delle dipendenze non è più automatica come prima
	-rm -vf $(DEPS)