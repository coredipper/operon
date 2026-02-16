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

Simulate a **NegativeFeedbackLoop** controlling a value toward a setpoint. Configure gain, damping, and disturbances to watch convergence, oscillation, or overdamping in real time.

## Features

- **6 presets**: Temperature control, oscillating convergence, overdamped, underdamped, disturbance rejection
- **Tunable parameters**: Setpoint, gain, damping, iterations, disturbance injection
- **Convergence analysis**: Steps to within 1% of setpoint, loop statistics

## How It Works

The `NegativeFeedbackLoop` computes a correction at each step based on the error (distance from setpoint), scaled by gain and damped to prevent oscillation. This mirrors biological homeostasis — thermostats, blood sugar regulation, and neural feedback.

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
