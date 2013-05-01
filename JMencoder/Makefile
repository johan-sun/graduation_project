CC = gcc
CXX = g++
CFLAGS = -Wall
LDFLAGS = -lm
INCLUDE = -I inc
SOURCES = $(notdir $(wildcard src/*.c))
#OBJECTS = $(SOURCES:.c=.o) 此语句同下
OBJECTS = $(patsubst %.c,%.o,$(SOURCES))
VPATH = src:inc
CFLAGS += $(patsubst %, -I%, $(subst :, , $(VPATH)))

ifeq ($(DEBUG), on)
	CFLAGS += -g
endif
.PHONY:all clean
all:lencod
	

lencod:$(addprefix objs/, $(OBJECTS))
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@ 
objs/%.o:src/%.c
	@[ -d objs ] || mkdir -p objs
	$(CC) $< $(CFLAGS) -c -o $@

clean:
	rm -r -f -v *.o lencod objs
