#include "../src/ConstellateEngine.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using constellate::Engine;

static void learnAlternating(Engine& engine, int repetitions, double step = 0.25) {
	double time = 0.0;
	for (int i = 0; i < repetitions; ++i) {
		engine.observe(0, time);
		time += step;
		engine.observe(1, time);
		time += step;
	}
}

static void testLearningAndTiming() {
	Engine engine;
	learnAlternating(engine, 32);
	assert(engine.hasSequence());
	assert(engine.transitionProbability(0, 1) > 0.99f);
	assert(engine.transitionProbability(1, 0) > 0.95f);
	assert(std::fabs(engine.intervalFor(0, 1) - 0.25f) < 0.01f);
	assert(std::fabs(engine.intervalFor(1, 0) - 0.25f) < 0.01f);
}

static void testDeterministicContinuation() {
	Engine engine;
	learnAlternating(engine, 32);
	engine.playLength = 0;
	engine.pushPlayback(0);
	engine.reseed(12345u);
	for (int i = 0; i < 64; ++i) {
		constellate::Choice choice = engine.choose(1, 1.f, 0.f);
		assert(choice.event == ((i & 1) ? 0 : 1));
		assert(choice.confidence > 0.95f);
	}
}

static void testDriftExploresAllStreams() {
	Engine engine;
	learnAlternating(engine, 16);
	engine.playLength = 0;
	engine.pushPlayback(0);
	engine.reseed(0xA57E1234u);
	int counts[Engine::CHANNELS] = {0, 0, 0, 0};
	for (int i = 0; i < 1000; ++i)
		++counts[engine.choose(4, 1.f, 1.f).event];
	for (int i = 0; i < Engine::CHANNELS; ++i)
		assert(counts[i] > 150);
}

static void testVariableOrderMemory() {
	Engine engine;
	double time = 0.0;
	// 0,1 predicts 2 while 3,1 predicts 0. A first-order model only sees 1.
	for (int i = 0; i < 48; ++i) {
		const int pattern[] = {0, 1, 2, 3, 1, 0};
		for (int event : pattern) {
			engine.observe(event, time);
			time += 0.1;
		}
	}
	engine.playLength = 0;
	engine.pushPlayback(0);
	engine.pushPlayback(1);
	engine.reseed(77u);
	for (int i = 0; i < 32; ++i) {
		constellate::Choice choice = engine.choose(2, 1.f, 0.f);
		assert(choice.event == 2);
		engine.playLength = 0;
		engine.pushPlayback(0);
		engine.pushPlayback(1);
	}
}

static void testClear() {
	Engine engine;
	learnAlternating(engine, 4);
	engine.clear();
	assert(!engine.hasSequence());
	assert(engine.observations == 0);
	for (int from = 0; from < Engine::CHANNELS; ++from)
		for (int to = 0; to < Engine::CHANNELS; ++to)
			assert(engine.transitionProbability(from, to) == 0.f);
}

int main() {
	testLearningAndTiming();
	testDeterministicContinuation();
	testDriftExploresAllStreams();
	testVariableOrderMemory();
	testClear();
	std::cout << "Constellate engine tests passed\n";
	return 0;
}
