# Project Overview

Elder Care Vision is an AI-powered PTZ camera monitoring system designed to ensure the safety of elderly individuals, empowering them to live independently while offering peace of mind to caregivers and families. Our solution detects emergencies in real-time, actively communicates with seniors to verify their wellbeing, and automatically notifies designated contacts during critical situations.

## The Challenge

As society ages, maintaining both independence and safety for seniors becomes increasingly challenging. Traditional emergency alert systems, such as wearable panic buttons, often fail—particularly if a senior is unconscious, disoriented, or unable to reach the device. There is a critical need for a more intelligent, proactive, and reliable safety system.

## Our Solution

Elder Care Vision utilizes the two-tier AI algorithm to:

- Analyze body points in real-time for accurate movement detection
- Two stage verification system to minimize false alarms
- Interpret environmental and situational context, ensuring alerts are triggered only by true anomalies
- Dynamic parameter tuning based on collected metrics

## Key Innovations

### Active Interaction

The system actively verbally communicates with seniors, checking their status after detecting irregularities, ensuring human-like interaction and reducing unnecessary alerts.

### Autonomous Operation

No wearable devices needed, increasing adoption rates and ensuring uninterrupted monitoring without issues related to device loss, dead batteries, discomfort, or maintenance. The system operates seamlessly and unobtrusively in any environment.

### Self-Improving Algorithm

Elder Care Vision learns continuously from new data, adapting to different behaviors, environments, and scenarios. The first-tier response system reacts quickly even in unfamiliar situations, minimizing reliance on static rules and significantly reducing reaction time to emerging risks.

### Comprehensive Incident Documentation

Every detected incident is automatically recorded, help to create a detailed medical report for healthcare providers, aiding in treatment, and improving preventive care strategies.

### Advanced Autonomous Calling System

Our breakthrough in elderly care technology includes:

#### Intelligent Communication Protocol

- Automated wellness checks triggered by detected anomalies
- Natural language processing for human-like conversations
- Voice recognition to verify cognitive state

#### Emergency Response Integration

- Direct connection to emergency services when necessary
- Automated escalation based on response urgency
- Smart routing to available caregivers or family members
- Integration with local emergency dispatch systems

#### Proactive Health Monitoring

- Pattern recognition for early warning signs
- Behavioral analysis to predict potential issues

## Who Benefits from Elder Care Vision?

### For Families

- Continuous monitoring without invading personal privacy
- Instant notifications of incidents and emergencies
- Peace of mind guaranteed – your home free from tragic discoveries

### For Seniors

- Freedom to live independently with discreet safety support
- Confidence that help will arrive quickly if needed
- No need for wearables or device maintenance

### For Care Facilities

- Streamlined staff workflows and resource allocation
- Improved resident safety and quality of life
- Potential reduction in care-related operational costs
- Cost-effective resource optimization through AI verification

## Technology Stack

- Seamless integration with existing alarm, emergency, and communication networks
- Two tier AI algorithms analyzing body posture and movement
- Intelligent data buffering and false alarm reduction systems

## Target Audience

- Families caring for independently living seniors
- Nursing homes and long-term care facilities
- Seniors who value both independence and safety
- Rehabilitation centers and assisted living communities

## Core TOOLS USED

### Core Vision and Detection

- `mediapipe` + `opencv-python` + `numpy`: Core vision processing stack
  - Mediapipe handles real-time body pose detection and tracking
  - OpenCV manages video capture and frame processing
  - NumPy provides numerical computation support for pose analysis

### AI and Analysis

- `openai` + `openai-agents`: AI processing system
  - OpenAI services for fall analysis and situation assessment
  - AI agents coordinate different monitoring aspects
  - Advanced pattern recognition and decision making

### Emergency Response System

- `adb_shell` + `pure-python-adb`: Android device communication
  - Emergency calls and SMS functionality
  - Device control during fall detection events
  - Automated response system

### Audio Processing

- `sounddevice` + `soundfile`: Audio management system
  - Emergency alert playback
  - Real-time audio communication
  - Message processing for emergency responses

### Web Interface and Communication

- `flask` + `websockets`: Real-time monitoring interface
  - Web-based status monitoring
  - Live fall detection results display
  - Real-time health status updates
  - Bidirectional system communication

### Data Visualization

- `matplotlib` + `pillow`: Visual feedback system
  - Fall detection data visualization
  - Movement pattern analysis
  - Warning overlay generation
  - Frame sequence processing

### Configuration and System Management

- `pydantic` + `python-dotenv` + `PyYAML`: System configuration stack
  - Configuration validation and management
  - Environment variable handling
  - Secure API key storage
  - Logging configuration

### Monitoring and Telemetry

- `opentelemetry-instrumentation-asyncio`: System monitoring
  - Asynchronous operation tracking
  - Communication pipeline monitoring

## Elder Care Vision Team Contributions

### Wojciech Czaplejewicz - System Architect & Core Infrastructure

- **Technologies:** Python, OpenAI Responses API, OpenAI Agents SDK, Speech Processing (TTS/STT), State Machine Architecture, Configuration Management, Logging, CI/CD Tools
- **Key Contributions:**
  - Established project foundation (repository structure, environment configuration)
  - Designed core architecture (state machine, agent systems, configuration management)
  - Implemented integration services (OpenAI, TTS/STT, confidence measurement)
  - Set up DevOps infrastructure (requirements management, code quality tools)

### Tomasz Bobowski - Computer Vision & Detection Algorithm Development

- **Technologies:** Computer Vision Libraries, LLMs, Development of CV debug Tools, Performance Optimization, Real-time Data Processing
- **Key Contributions:**
  - Developed real time two-tier AI image recognition system
  - Created frontend for Algorithm debugging interface with real-time recognition visualization overlay
  - Implemented system monitoring with stats tracking and performance monitoring

### Damian Baranski - Emergency Response System

- **Technologies:** Emergency Response Systems, Image Analysis, OpenAI Integration, Audio Processing
- **Key Contributions:**
  - Built emergency call system with image analyzer and phone audio integration (By hacking physical phone via adb debug interface!)
  - Integrated OpenAI for automated image description and text-to-speech
  - Connected emergency components with main system architecture
  - Documented emergency systems and integration procedures

### Mateusz Paczynski - Camera Systems Specialist

- **Technologies:** Camera Control Systems, API Development, Authentication, PTZ Technologies
- **Key Contributions:**
  - Developed comprehensive camera API with PTZ control and patrol functionality
  - Implemented authentication systems with security features and access control
  - Created image capture capabilities including burst and snapshot features
  - Integrated camera systems with the core architecture

## Project Timeline Highlights

### Day 1 (April 25, 2025)

- 18:07 - Project initialization and pre-commit setup
- 18:31 - Logging system implementation
- 18:46 - Response API development
- 18:50 - Configuration and tools setup
- 19:42 - Initial LLM integration work
- 20:12 - Head position tracking implementation
- 20:29 - UI display and window resolution
- 20:45 - Emergency call tools integration
- 21:24 - Camera API initial implementation
- 21:47 - OpenAI image description integration
- 22:02 - State machine and agent system
- 22:24 - Emergency call documentation
- 22:28 - PTZ control API implementation
- 22:44 - Image recognition system completion
- 23:24 - Emergency call system integration
- 23:54 - Oopsie system documentation

### Day 2 (April 26, 2025)

- 00:37 - TTS and STT implementation
- 01:03 - Camera patrol and burst features
- 01:15 - HSI Agent integration
- 01:33 - Prompt system improvements
- 01:56 - Emergency caller logging enhancement
- — - Some Sleep!
- 07:32 - Code quality improvements
- 07:50 - Coordinator logging updates
- 08:02 - Health Status retry implementation
- 08:23 - System refactoring
- 08:49 - Temporary integration setup
- 09:54 - OpenAI service confidence level integration
- 14:00 - Finishing bug fixing and integration merge

## Project Prior Work Summary

We engaged in preliminary brainstorming sessions and planning activities before the main development work began:

### Technology Assessment

- Evaluated local neural network model performance and its system impact
- Ran joint tracking examples to compare speed and computational requirements
- No formal reasoning or analysis was developed from these tests

### Feature Exploration

- Investigated options for implementing phone calling functionality (API vs. real phone)
- Compared camera system options (360° cameras vs. tracking cameras and its algorithms)
- Explored potential use cases for LLM models at different project stages

### Project Organization

- Created an empty repository 4 days before the hackathon
- Designed the overall architecture concept
- Divided work among team members, with each person choosing their own tasks and approaches
