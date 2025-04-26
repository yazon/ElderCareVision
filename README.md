# Elder Care Vision System

## Overview

AI-powered system for elderly care monitoring with fall detection and health status inquiry capabilities. Uses computer vision and voice interaction to assess situations and initiate emergency responses.

## Features

- **State-Driven Coordination**\
  Three operational states managed by Coordinator:

  - Image analysis (fall detection)
  - Voice-based health inquiry
  - Emergency response handling

- **Multi-Agent Architecture**

  - PersonStateAnalyzerAgent: Camera-based fall detection
  - HealthStatusInquiryAgent: Voice interaction system
  - Coordinator: State machine for system orchestration

- **Emergency Workflow**

  - Dual confidence thresholds for fall detection
  - Voice confirmation protocol
  - Integrated emergency calling system

## Installation

**Create a virtual environment:**

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Unix/Linux
source .venv/bin/activate
```

**Windows**:

```
install.bat
```

**Other systems**:

```
python -m pip install -U pip wheel setuptools
pip install -e .[dev]
```

Configure environment variables:

```
  cp .env.example .env
```

## Configuration

System environment variables:

Required API keys:
OPENAI_API_KEY=your-key-here

## Usage Examples

1. Start main system:
   python -m elder_care_vision.main

## State Machine Logic

The system operates based on a state machine managed by the `Coordinator`. The possible states are:

1. **ANALYZING_IMAGE**:

   - This is the default initial state.
   - The `PersonStateAnalyzerAgent` continuously analyzes the camera feed for potential falls.
   - The system monitors the `confidence_level` reported by the agent.
   - **Transitions**:
     - If `confidence_level` > `confidence_threshold_1` (high confidence of a fall), transitions to `CALLING_EMERGENCY`.
     - If `confidence_level` >= `confidence_threshold_2` (medium confidence of a fall), transitions to `INQUIRING_HEALTH`.
     - Otherwise, remains in `ANALYZING_IMAGE`.

1. **INQUIRING_HEALTH**:

   - Triggered when the fall detection confidence is medium.
   - The `HealthStatusInquiryAgent` initiates a voice interaction to ask the person about their well-being.
   - The agent determines the person's `health_status` (e.g., "OK", "Needs Help", "Not OK").
   - **Transitions**:
     - If `health_status` indicates an emergency (configured in `health_status_needs_help` or `health_status_not_ok`), transitions to `CALLING_EMERGENCY`.
     - Otherwise (e.g., status is "OK"), transitions back to `ANALYZING_IMAGE`.

1. **CALLING_EMERGENCY**:

   - Triggered either by high fall detection confidence or by an emergency health status reported after inquiry.
   - The `emergency_call_tool` is invoked to initiate contact with emergency services or designated contacts.
   - Information from the context, such as `FallDetectionResult` and `health_status`, is passed to the tool.
   - **Transitions**:
     - After the emergency call process is completed (or attempted), transitions back to `ANALYZING_IMAGE` to resume monitoring.

The thresholds (`confidence_threshold_1`, `confidence_threshold_2`) and emergency statuses are configurable via the system's configuration files.

## Best Practices

1. Environment Setup

   - Use dedicated camera with 1080p+ resolution
   - Position microphone within 2m of monitoring area
   - Maintain ambient noise < 50dB

1. Monitoring Configuration

   - Set confidence_threshold_1 between 70-80
   - Set confidence_threshold_2 between 45-70

## Notes

- Hardware Requirements:
  - Integrated webcam for real-time processing
  - Noise-canceling microphone array suggested
