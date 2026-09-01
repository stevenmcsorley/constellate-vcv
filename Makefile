RACK_DIR ?= ../Rack-SDK

FLAGS +=
CFLAGS +=
CXXFLAGS +=

LDFLAGS +=

SOURCES += $(wildcard src/*.cpp)

DISTRIBUTABLES += res
DISTRIBUTABLES += $(wildcard LICENSE*)
DISTRIBUTABLES += $(wildcard presets)

include $(RACK_DIR)/plugin.mk

.PHONY: test
test:
	$(CXX) -std=c++11 -O2 -Wall -Wextra -pedantic tests/engine_test.cpp -o tests/engine_test
	./tests/engine_test

