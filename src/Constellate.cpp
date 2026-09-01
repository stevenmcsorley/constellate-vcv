#include "plugin.hpp"
#include "ConstellateEngine.hpp"

#include <array>
#include <atomic>
#include <cmath>

using constellate::Choice;
using ConstellateEngine = constellate::Engine;

static float constellateClamp(float value, float low, float high) {
	return value < low ? low : (value > high ? high : value);
}

struct Constellate : Module {
	enum ParamId {
		MEMORY_PARAM,
		AFFINITY_PARAM,
		DRIFT_PARAM,
		DENSITY_PARAM,
		MORPH_PARAM,
		LEARN_PARAM,
		HOLD_PARAM,
		PARAMS_LEN
	};
	enum InputId {
		ENUMS(EVENT_INPUTS, 4),
		CLOCK_INPUT,
		RESET_INPUT,
		INPUTS_LEN
	};
	enum OutputId {
		ENUMS(EVENT_OUTPUTS, 4),
		THREAD_OUTPUT,
		OUTPUTS_LEN
	};
	enum LightId {
		LEARN_LIGHT,
		HOLD_LIGHT,
		LIGHTS_LEN
	};

	ConstellateEngine engine;
	dsp::SchmittTrigger eventTriggers[4];
	dsp::SchmittTrigger clockTrigger;
	dsp::SchmittTrigger resetTrigger;
	dsp::BooleanTrigger learnButton;
	dsp::BooleanTrigger holdButton;
	dsp::PulseGenerator eventPulses[4];
	dsp::ClockDivider uiDivider;

	bool learningEnabled = true;
	bool holdEnabled = false;
	double elapsedSeconds = 0.0;
	float internalCountdown = 0.1f;
	float gateLength = 0.005f;
	float threadTarget = 0.f;
	float threadVoltage = 0.f;
	float pulseLevel[4] = {0.f, 0.f, 0.f, 0.f};
	int currentEvent = -1;

	std::atomic<float> uiWeights[16];
	std::atomic<float> uiPulses[4];
	std::atomic<float> uiConfidence;
	std::atomic<int> uiCurrentEvent;
	std::atomic<bool> clearRequested;
	std::atomic<bool> reseedRequested;

	Constellate() {
		config(PARAMS_LEN, INPUTS_LEN, OUTPUTS_LEN, LIGHTS_LEN);
		configParam(MEMORY_PARAM, 1.f, 4.f, 2.f, "Memory depth", " events");
		getParamQuantity(MEMORY_PARAM)->snapEnabled = true;
		configParam(AFFINITY_PARAM, 0.f, 1.f, 0.72f, "Affinity", "%", 0.f, 100.f);
		configParam(DRIFT_PARAM, 0.f, 1.f, 0.12f, "Drift", "%", 0.f, 100.f);
		configParam(DENSITY_PARAM, 0.f, 1.f, 0.65f, "Density", "%", 0.f, 100.f);
		configParam(MORPH_PARAM, 0.f, 1.f, 0.f, "Live / Dream morph", "%", 0.f, 100.f);
		configButton(LEARN_PARAM, "Toggle learning");
		configButton(HOLD_PARAM, "Hold learned constellation");

		static const char* names[4] = {"A", "B", "C", "D"};
		for (int i = 0; i < 4; ++i) {
			configInput(EVENT_INPUTS + i, string::f("%s event", names[i]));
			configOutput(EVENT_OUTPUTS + i, string::f("%s event", names[i]));
			configBypass(EVENT_INPUTS + i, EVENT_OUTPUTS + i);
		}
		configInput(CLOCK_INPUT, "Dream clock");
		configInput(RESET_INPUT, "Reset playback context");
		configOutput(THREAD_OUTPUT, "Thread confidence CV");

		uiDivider.setDivision(1024);
		for (int i = 0; i < 16; ++i)
			uiWeights[i].store(0.f);
		for (int i = 0; i < 4; ++i)
			uiPulses[i].store(0.f);
		uiConfidence.store(0.f);
		uiCurrentEvent.store(-1);
		clearRequested.store(false);
		reseedRequested.store(false);
	}

	void onReset() override {
		engine.clear();
		learningEnabled = true;
		holdEnabled = false;
		elapsedSeconds = 0.0;
		internalCountdown = 0.1f;
		threadTarget = 0.f;
		threadVoltage = 0.f;
		currentEvent = -1;
		for (int i = 0; i < 4; ++i)
			pulseLevel[i] = 0.f;
	}

	void emitEvent(int event, float confidence) {
		if (event < 0 || event >= 4)
			return;
		eventPulses[event].trigger(gateLength);
		pulseLevel[event] = 1.f;
		currentEvent = event;
		threadTarget = constellateClamp(confidence, 0.f, 1.f) * 10.f;
	}

	void updateUi() {
		for (int from = 0; from < 4; ++from)
			for (int to = 0; to < 4; ++to)
				uiWeights[from * 4 + to].store(
					engine.transitionProbability(from, to), std::memory_order_relaxed);
		for (int i = 0; i < 4; ++i)
			uiPulses[i].store(pulseLevel[i], std::memory_order_relaxed);
		uiConfidence.store(threadTarget / 10.f, std::memory_order_relaxed);
		uiCurrentEvent.store(currentEvent, std::memory_order_relaxed);
	}

	json_t* dataToJson() override {
		json_t* root = json_object();
		json_object_set_new(root, "formatVersion", json_integer(1));
		json_object_set_new(root, "learningEnabled", json_boolean(learningEnabled));
		json_object_set_new(root, "holdEnabled", json_boolean(holdEnabled));
		json_object_set_new(root, "gateLength", json_real(gateLength));
		json_object_set_new(root, "rngState", json_integer(engine.rngState));
		json_object_set_new(root, "observations", json_integer(engine.observations));
		json_object_set_new(root, "globalInterval", json_real(engine.globalInterval));
		json_object_set_new(root, "globalIntervalWeight", json_real(engine.globalIntervalWeight));

		json_t* base = json_array();
		for (int i = 0; i < 4; ++i)
			json_array_append_new(base, json_real(engine.baseWeights[i]));
		json_object_set_new(root, "baseWeights", base);

		// Sparse storage keeps ordinary patch files small even though the engine can
		// represent all 4^4 contexts.
		json_t* transitions = json_array();
		for (int order = 0; order < ConstellateEngine::MAX_ORDER; ++order) {
			int contexts = 1;
			for (int i = 0; i <= order; ++i)
				contexts *= 4;
			for (int context = 0; context < contexts; ++context) {
				for (int target = 0; target < 4; ++target) {
					float weight = engine.weights[order][context][target];
					if (weight <= 0.f)
						continue;
					json_t* entry = json_array();
					json_array_append_new(entry, json_integer(order));
					json_array_append_new(entry, json_integer(context));
					json_array_append_new(entry, json_integer(target));
					json_array_append_new(entry, json_real(weight));
					json_array_append_new(transitions, entry);
				}
			}
		}
		json_object_set_new(root, "transitions", transitions);

		json_t* delays = json_array();
		for (int from = 0; from < 4; ++from) {
			for (int to = 0; to < 4; ++to) {
				if (engine.delayWeight[from][to] <= 0.f)
					continue;
				json_t* entry = json_array();
				json_array_append_new(entry, json_integer(from));
				json_array_append_new(entry, json_integer(to));
				json_array_append_new(entry, json_real(engine.delayMean[from][to]));
				json_array_append_new(entry, json_real(engine.delayWeight[from][to]));
				json_array_append_new(delays, entry);
			}
		}
		json_object_set_new(root, "delays", delays);

		json_t* learnHistory = json_array();
		for (int i = 0; i < engine.learnLength; ++i)
			json_array_append_new(learnHistory, json_integer(engine.learnHistory[i]));
		json_object_set_new(root, "learnHistory", learnHistory);

		json_t* playHistory = json_array();
		for (int i = 0; i < engine.playLength; ++i)
			json_array_append_new(playHistory, json_integer(engine.playHistory[i]));
		json_object_set_new(root, "playHistory", playHistory);
		return root;
	}

	void dataFromJson(json_t* root) override {
		engine.clear();
		if (json_t* value = json_object_get(root, "learningEnabled"))
			learningEnabled = json_boolean_value(value);
		if (json_t* value = json_object_get(root, "holdEnabled"))
			holdEnabled = json_boolean_value(value);
		if (json_t* value = json_object_get(root, "gateLength"))
			gateLength = constellateClamp((float) json_real_value(value), 0.001f, 0.05f);
		if (json_t* value = json_object_get(root, "rngState"))
			engine.reseed((uint32_t) json_integer_value(value));
		if (json_t* value = json_object_get(root, "observations"))
			engine.observations = (uint32_t) std::max<json_int_t>(0, json_integer_value(value));
		if (json_t* value = json_object_get(root, "globalInterval"))
			engine.globalInterval = constellateClamp((float) json_real_value(value), 0.01f, 30.f);
		if (json_t* value = json_object_get(root, "globalIntervalWeight"))
			engine.globalIntervalWeight = std::max(0.f, (float) json_real_value(value));

		if (json_t* base = json_object_get(root, "baseWeights")) {
			for (int i = 0; i < 4 && i < (int) json_array_size(base); ++i)
				engine.baseWeights[i] = std::max(0.f, (float) json_real_value(json_array_get(base, i)));
		}

		if (json_t* transitions = json_object_get(root, "transitions")) {
			size_t index;
			json_t* entry;
			json_array_foreach(transitions, index, entry) {
				if (!json_is_array(entry) || json_array_size(entry) < 4)
					continue;
				int order = (int) json_integer_value(json_array_get(entry, 0));
				int context = (int) json_integer_value(json_array_get(entry, 1));
				int target = (int) json_integer_value(json_array_get(entry, 2));
				float weight = (float) json_real_value(json_array_get(entry, 3));
				if (order >= 0 && order < ConstellateEngine::MAX_ORDER && context >= 0 &&
					context < ConstellateEngine::MAX_CONTEXTS && target >= 0 && target < 4 &&
					std::isfinite(weight) && weight > 0.f)
					engine.weights[order][context][target] = std::min(weight, 10000.f);
			}
		}

		if (json_t* delays = json_object_get(root, "delays")) {
			size_t index;
			json_t* entry;
			json_array_foreach(delays, index, entry) {
				if (!json_is_array(entry) || json_array_size(entry) < 4)
					continue;
				int from = (int) json_integer_value(json_array_get(entry, 0));
				int to = (int) json_integer_value(json_array_get(entry, 1));
				float mean = (float) json_real_value(json_array_get(entry, 2));
				float weight = (float) json_real_value(json_array_get(entry, 3));
				if (from >= 0 && from < 4 && to >= 0 && to < 4 && std::isfinite(mean) &&
					std::isfinite(weight) && weight > 0.f) {
					engine.delayMean[from][to] = constellateClamp(mean, 0.002f, 60.f);
					engine.delayWeight[from][to] = std::min(weight, 10000.f);
				}
			}
		}

		if (json_t* history = json_object_get(root, "learnHistory")) {
			engine.learnLength = std::min((int) json_array_size(history), ConstellateEngine::MAX_ORDER);
			for (int i = 0; i < engine.learnLength; ++i)
				engine.learnHistory[i] = clamp((int) json_integer_value(json_array_get(history, i)), 0, 3);
		}
		if (json_t* history = json_object_get(root, "playHistory")) {
			engine.playLength = std::min((int) json_array_size(history), ConstellateEngine::MAX_ORDER);
			for (int i = 0; i < engine.playLength; ++i)
				engine.playHistory[i] = clamp((int) json_integer_value(json_array_get(history, i)), 0, 3);
		}
		engine.lastObserved = engine.learnLength > 0 ? engine.learnHistory[0] : -1;
		engine.lastObservedTime = -1.0;
		updateUi();
	}

	void process(const ProcessArgs& args) override {
		const float dt = args.sampleTime;
		elapsedSeconds += dt;

		if (clearRequested.exchange(false)) {
			engine.clear();
			threadTarget = 0.f;
			currentEvent = -1;
		}
		if (reseedRequested.exchange(false))
			engine.reseed((uint32_t) (elapsedSeconds * 1000003.0) ^ 0x57A4D11Fu);

		if (learnButton.process(params[LEARN_PARAM].getValue() > 0.5f))
			learningEnabled = !learningEnabled;
		if (holdButton.process(params[HOLD_PARAM].getValue() > 0.5f))
			holdEnabled = !holdEnabled;

		if (resetTrigger.process(inputs[RESET_INPUT].getVoltage(), 0.1f, 1.f)) {
			engine.resetPlaybackToLearned();
			internalCountdown = 0.01f;
			threadTarget = 0.f;
			currentEvent = -1;
		}

		const int memory = clamp((int) std::lround(params[MEMORY_PARAM].getValue()), 1, 4);
		const float affinity = params[AFFINITY_PARAM].getValue();
		const float drift = params[DRIFT_PARAM].getValue();
		const float density = params[DENSITY_PARAM].getValue();
		const float morph = params[MORPH_PARAM].getValue();

		bool anyEventInput = false;
		for (int i = 0; i < 4; ++i) {
			anyEventInput = anyEventInput || inputs[EVENT_INPUTS + i].isConnected();
			if (!eventTriggers[i].process(inputs[EVENT_INPUTS + i].getVoltage(), 0.1f, 1.f))
				continue;

			if (learningEnabled && !holdEnabled)
				engine.observe(i, elapsedSeconds);

			if (engine.hasSequence() && engine.random01() < morph) {
				Choice choice = engine.choose(memory, affinity, drift);
				emitEvent(choice.event, choice.confidence);
			}
			else {
				engine.pushPlayback(i);
				emitEvent(i, 0.f);
			}
		}

		const bool clockConnected = inputs[CLOCK_INPUT].isConnected();
		if (clockConnected && clockTrigger.process(inputs[CLOCK_INPUT].getVoltage(), 0.1f, 1.f)) {
			float eventProbability = morph * (0.1f + 0.9f * density);
			if (engine.hasSequence() && engine.random01() < eventProbability) {
				Choice choice = engine.choose(memory, affinity, drift);
				emitEvent(choice.event, choice.confidence);
			}
		}

		// With no event or clock cables, a learned constellation can free-run at its
		// own recorded timing. Density compresses/expands that learned pulse train.
		if (!anyEventInput && !clockConnected && engine.hasSequence() && morph > 0.001f) {
			internalCountdown -= dt;
			if (internalCountdown <= 0.f) {
				int previous = engine.playLength > 0 ? engine.playHistory[0] : -1;
				Choice choice = engine.choose(memory, affinity, drift);
				if (engine.random01() < morph * (0.1f + 0.9f * density))
					emitEvent(choice.event, choice.confidence);
				float speedScale = std::pow(2.f, 1.f - density * 2.f);
				internalCountdown += constellateClamp(
					engine.intervalFor(previous, choice.event) * speedScale, 0.01f, 30.f);
			}
		}
		else if (anyEventInput || clockConnected) {
			internalCountdown = constellateClamp(engine.globalInterval, 0.01f, 30.f);
		}

		for (int i = 0; i < 4; ++i) {
			bool high = eventPulses[i].process(dt);
			outputs[EVENT_OUTPUTS + i].setVoltage(high ? 10.f : 0.f);
			pulseLevel[i] *= std::exp(-dt * 8.f);
		}

		threadVoltage += (threadTarget - threadVoltage) * constellateClamp(dt * 24.f, 0.f, 1.f);
		outputs[THREAD_OUTPUT].setVoltage(threadVoltage);
		lights[LEARN_LIGHT].setBrightnessSmooth(learningEnabled && !holdEnabled ? 1.f : 0.08f, dt);
		lights[HOLD_LIGHT].setBrightnessSmooth(holdEnabled ? 1.f : 0.f, dt);

		if (uiDivider.process())
			updateUi();
	}
};

struct ConstellateDisplay : TransparentWidget {
	Constellate* module = nullptr;

	static NVGcolor colorFor(int index, unsigned char alpha = 255) {
		static const unsigned char colors[4][3] = {
			{0x25, 0xd6, 0xf2},
			{0xff, 0xb5, 0x25},
			{0xf0, 0x42, 0xa4},
			{0x83, 0xe3, 0x42}
		};
		return nvgRGBA(colors[index][0], colors[index][1], colors[index][2], alpha);
	}

	void drawLayer(const DrawArgs& args, int layer) override {
		if (layer != 1)
			return;
		NVGcontext* vg = args.vg;
		const float width = box.size.x;
		const float height = box.size.y;

		nvgSave(vg);
		nvgScissor(vg, RECT_ARGS(args.clipBox));
		nvgBeginPath(vg);
		nvgRoundedRect(vg, 0.5f, 0.5f, width - 1.f, height - 1.f, mm2px(1.4f));
		NVGpaint glass = nvgLinearGradient(vg, 0.f, 0.f, 0.f, height,
			nvgRGBA(0x06, 0x0b, 0x11, 0xff), nvgRGBA(0x02, 0x04, 0x08, 0xff));
		nvgFillPaint(vg, glass);
		nvgFill(vg);
		nvgStrokeColor(vg, nvgRGBA(0xc8, 0x8b, 0x2b, 0xb0));
		nvgStrokeWidth(vg, 1.f);
		nvgStroke(vg);

		const std::array<Vec, 4> nodes = {
			Vec(width * 0.19f, height * 0.22f),
			Vec(width * 0.81f, height * 0.22f),
			Vec(width * 0.19f, height * 0.78f),
			Vec(width * 0.81f, height * 0.78f)
		};

		for (int from = 0; from < 4; ++from) {
			for (int to = 0; to < 4; ++to) {
				float strength = module
					? module->uiWeights[from * 4 + to].load(std::memory_order_relaxed)
					: ((to == (from + 1) % 4) ? 0.72f : 0.08f);
				if (strength < 0.015f)
					continue;
				unsigned char alpha = (unsigned char) constellateClamp(22.f + strength * 205.f, 0.f, 255.f);
				if (from == to) {
					nvgBeginPath(vg);
					nvgCircle(vg, nodes[from].x, nodes[from].y, mm2px(3.f + strength * 1.2f));
					nvgStrokeColor(vg, colorFor(to, alpha));
					nvgStrokeWidth(vg, 0.7f + strength * 1.2f);
					nvgStroke(vg);
					continue;
				}

				Vec a = nodes[from];
				Vec b = nodes[to];
				Vec delta = b.minus(a);
				float side = from < to ? 1.f : -1.f;
				Vec control = a.plus(b).div(2.f).plus(Vec(-delta.y, delta.x).normalize().mult(side * mm2px(1.8f)));
				nvgBeginPath(vg);
				nvgMoveTo(vg, a.x, a.y);
				nvgBezierTo(vg, control.x, control.y, control.x, control.y, b.x, b.y);
				nvgStrokeColor(vg, colorFor(to, alpha));
				nvgStrokeWidth(vg, 0.55f + strength * 1.25f);
				nvgStroke(vg);
			}
		}

		int current = module ? module->uiCurrentEvent.load(std::memory_order_relaxed) : 1;
		for (int i = 0; i < 4; ++i) {
			float pulse = module ? module->uiPulses[i].load(std::memory_order_relaxed) : (i == 1 ? 0.7f : 0.15f);
			float emphasis = i == current ? 0.35f : 0.f;
			float glowAmount = constellateClamp(pulse + emphasis, 0.f, 1.f);
			NVGpaint halo = nvgRadialGradient(vg, nodes[i].x, nodes[i].y, 0.f, mm2px(4.6f),
				colorFor(i, (unsigned char) (glowAmount * 180.f)), colorFor(i, 0));
			nvgBeginPath(vg);
			nvgCircle(vg, nodes[i].x, nodes[i].y, mm2px(4.6f));
			nvgFillPaint(vg, halo);
			nvgFill(vg);
			nvgBeginPath(vg);
			nvgCircle(vg, nodes[i].x, nodes[i].y, mm2px(1.35f + pulse * 0.45f));
			nvgFillColor(vg, colorFor(i));
			nvgFill(vg);
			nvgStrokeColor(vg, nvgRGBA(0xff, 0xff, 0xff, 0xc0));
			nvgStrokeWidth(vg, 0.65f);
			nvgStroke(vg);
		}

		float confidence = module ? module->uiConfidence.load(std::memory_order_relaxed) : 0.55f;
		float meterWidth = (width - mm2px(4.f)) * confidence;
		nvgBeginPath(vg);
		nvgRoundedRect(vg, mm2px(2.f), height - mm2px(2.1f), meterWidth, mm2px(0.7f), mm2px(0.35f));
		nvgFillColor(vg, nvgRGBA(0xff, 0xb5, 0x25, 0xd8));
		nvgFill(vg);

		nvgRestore(vg);
	}
};

struct ConstellateWidget : ModuleWidget {
	ConstellateWidget(Constellate* module) {
		setModule(module);
		setPanel(createPanel(asset::plugin(pluginInstance, "res/Constellate.svg")));

		addChild(createWidget<ScrewBlack>(Vec(0, 0)));
		addChild(createWidget<ScrewBlack>(Vec(box.size.x - RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewBlack>(Vec(0, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		addChild(createWidget<ScrewBlack>(Vec(box.size.x - RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

		ConstellateDisplay* display = new ConstellateDisplay;
		display->module = module;
		display->box.pos = mm2px(Vec(26.f, 19.5f));
		display->box.size = mm2px(Vec(39.44f, 39.f));
		addChild(display);

		addParam(createParamCentered<RoundLargeBlackKnob>(mm2px(Vec(14.f, 30.f)), module, Constellate::MEMORY_PARAM));
		addParam(createParamCentered<RoundLargeBlackKnob>(mm2px(Vec(14.f, 51.5f)), module, Constellate::AFFINITY_PARAM));
		addParam(createParamCentered<RoundLargeBlackKnob>(mm2px(Vec(77.44f, 30.f)), module, Constellate::DRIFT_PARAM));
		addParam(createParamCentered<RoundLargeBlackKnob>(mm2px(Vec(77.44f, 51.5f)), module, Constellate::DENSITY_PARAM));
		addParam(createParamCentered<RoundHugeBlackKnob>(mm2px(Vec(45.72f, 74.f)), module, Constellate::MORPH_PARAM));

		addParam(createParamCentered<LEDButton>(mm2px(Vec(34.f, 91.5f)), module, Constellate::LEARN_PARAM));
		addChild(createLightCentered<MediumLight<YellowLight>>(mm2px(Vec(34.f, 91.5f)), module, Constellate::LEARN_LIGHT));
		addParam(createParamCentered<LEDButton>(mm2px(Vec(57.44f, 91.5f)), module, Constellate::HOLD_PARAM));
		addChild(createLightCentered<MediumLight<BlueLight>>(mm2px(Vec(57.44f, 91.5f)), module, Constellate::HOLD_LIGHT));

		const float inputX[6] = {8.5f, 23.4f, 38.3f, 53.2f, 68.1f, 83.f};
		for (int i = 0; i < 4; ++i)
			addInput(createInputCentered<PJ301MPort>(mm2px(Vec(inputX[i], 106.5f)), module, Constellate::EVENT_INPUTS + i));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(inputX[4], 106.5f)), module, Constellate::CLOCK_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(inputX[5], 106.5f)), module, Constellate::RESET_INPUT));

		const float outputX[5] = {15.9f, 30.8f, 45.72f, 60.6f, 75.5f};
		for (int i = 0; i < 4; ++i)
			addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(outputX[i], 120.5f)), module, Constellate::EVENT_OUTPUTS + i));
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(outputX[4], 120.5f)), module, Constellate::THREAD_OUTPUT));

	}

	void appendContextMenu(Menu* menu) override {
		Constellate* module = dynamic_cast<Constellate*>(this->module);
		if (!module)
			return;
		menu->addChild(new MenuSeparator);
		menu->addChild(createMenuLabel("Constellation memory"));
		menu->addChild(createMenuItem("Clear learned constellation", "", [module]() {
			module->clearRequested.store(true);
		}));
		menu->addChild(createMenuItem("Reseed dream generator", "", [module]() {
			module->reseedRequested.store(true);
		}));
		menu->addChild(createSubmenuItem("Trigger length", "", [module](Menu* submenu) {
			static const float lengths[] = {0.001f, 0.002f, 0.005f, 0.01f, 0.02f};
			for (float length : lengths) {
				submenu->addChild(createCheckMenuItem(
					string::f("%g ms", length * 1000.f), "",
					[module, length]() { return std::fabs(module->gateLength - length) < 0.0001f; },
					[module, length]() { module->gateLength = length; }));
			}
		}));
	}
};

Model* modelConstellate = createModel<Constellate, ConstellateWidget>("Constellate");
