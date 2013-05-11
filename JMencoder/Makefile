CC = gcc
CXX = g++
CFLAGS = -Wall
CUDA_ROOT = /opt/cuda-5.0
LDFLAGS = -lm #-L /opt/cuda-5.0/lib64 -lcudart
SOURCES = $(notdir $(wildcard src/*.c))
#OBJECTS = $(SOURCES:.c=.o) 此语句同下
OBJECTS = $(patsubst %.c,%.o,$(SOURCES))
VPATH = src:inc
CFLAGS += $(patsubst %, -I%, $(subst :, , $(VPATH)))
NVCCFLAGS = $(patsubst %, -I%, $(subst :, , $(VPATH)))
OBJECTS += cuda_h264.o

arch = $(shell uname -m)



NVCC = nvcc
NVCCFLAGS += -gencode arch=compute_11,code=sm_11 -ccbin gcc-4.6

ifeq ($(DEBUG), on)
	CFLAGS += -g
	NVCCFLAGS += -g -G
endif

ifeq ($(arch), x86_64)
	LDFLAGS += -L $(CUDA_ROOT)/lib64 -lcudart
else
	LDFLAGS += -L $(CUDA_ROOT)/lib -lcudart
endif

ifeq ($(CUDA_ROOT), )
	CUDA_ROOT = /opt/cuda-5.0
endif


.PHONY:all clean
all:lencod
	

lencod:$(addprefix objs/, $(OBJECTS))
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@ 
objs/%.o:src/%.c
	@[ -d objs ] || mkdir -p objs
	$(CC) $< $(CFLAGS) -c -o $@

objs/cuda_h264.o:src/cuda_h264.cu
	@[ -d objs ] || mkdir -p objs
	$(NVCC) $(NVCCFLAGS) $< -c  -o objs/cuda_h264.o

clean:
	rm -r -f -v *.o lencod objs
