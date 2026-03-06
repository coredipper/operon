---
title: Operon Feedback Loop Homeostasis
emoji: ⚖️
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Negative feedback loop homeostasis simulation
---

# ⚖️ Feedback Loop Homeostasis

Simulate a negative feedback loop driving a value toward a setpoint -- like a thermostat, blood sugar regulation, or any biological homeostasis mechanism.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "Temperature control" or "Underdamped oscillation") and click **Run Simulation** to see the value converge toward the setpoint over time.
2. Increase the **Gain** slider to see faster convergence, or push it too high to trigger oscillation. Raise **Damping** to suppress overshoot.
3. Set a **Disturbance step** and **Disturbance magnitude** to inject a sudden perturbation mid-simulation and watch the loop reject it.
4. Compare "Overdamped" vs. "Underdamped" presets to see the tradeoff between speed and stability.

## How It Works

The NegativeFeedbackLoop computes a correction at each step based on the error (distance from setpoint), scaled by gain and damped to prevent oscillation. This mirrors how biological systems maintain homeostasis -- applying proportional corrections that self-regulate toward equilibrium.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
