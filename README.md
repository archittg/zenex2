# Space Edge AI

# Autonomous Satellite Intelligence & Smart Downlink Simulator

# Don't transmit pixels. Transmit intelligence.

Space Edge AI is a simulator we built to show how satellites could actually process imagery onboard instead of just beaming everything down to Earth and figuring it out later. The idea is simple: run the AI on the satellite itself, decide what's actually worth sending, and only transmit that. Not the whole image every time.

This matters because satellites deal with some pretty hard constraints — bandwidth, power, storage, and the fact that you only get a communication window with the ground station every so often. You can't just brute-force your way past those.

# The problem we're solving

Right now, most satellite pipelines look like this:

Satellite → Capture Image → Store → Transmit Huge Image → Ground AI

The AI only runs once the image reaches the ground. Which means:

You're burning bandwidth sending images full of clouds, ocean, empty terrain — stuff nobody cares about
Transmission itself costs energy, and satellites don't have much to spare
Everything is stuck waiting for the next ground-station pass before anything useful happens
Latency piles up for no good reason

Basically, a lot of the data being sent down was never going to be useful in the first place.

# What we're doing differently

We flip the order — run the AI first, on the satellite, then decide what to send:

Satellite Image → Edge AI → Object/Event Detection → Intelligence Extraction
→ Mission Priority Engine → Smart Downlink → Ground Station

So instead of transmitting a full image, the satellite sends something like this:

json
{

  "target": "ship",
  
  "confidence": 0.94,
  
  "latitude": 19.076,
  
  "longitude": 72.877,
  
  "priority": "HIGH"
  
}

A few bytes instead of megabytes, and it's already telling you what matters.

# What it actually does

# Edge AI detection
YOLO runs on the imagery and picks out whatever's relevant to the mission.

# Intelligence extraction
Every detection gets turned into a compact metadata packet: object type, confidence, bounding box, location, timestamp, priority.

# Mission-aware priority engine
Not everything that gets detected deserves to be sent. A wildfire and a cloud are not the same:

Wildfire → CRITICAL → send it now

Ship → HIGH → send it

Building → LOW → can wait

Cloud → discard, don't bother

# Smart downlink
Simulates the realistic constraints: limited bandwidth, transmission capacity, latency, and the ground station not always being available.

# Resource-aware simulation 
Also models battery, CPU, and storage limits, because a satellite running out of power mid-pass is a real thing.

# Confidence-aware transmission
How much data actually gets sent depends on how sure the model is:

Above 90% confidence → just the metadata, that's enough

60–90% → metadata plus a thumbnail

Below 60% → metadata plus a cropped image, since we're less sure and want the ground team to be able to double check

# Edge optimization 
We also want to compare a normal FP32 model against an INT8-optimized one running through ONNX Runtime, to show this can realistically run on lightweight edge hardware, not just a beefy GPU.

# Architecture
Satellite Image
      ↓
Image Ingestion
      ↓
Edge AI / YOLO
      ↓
Intelligence Extraction
      ↓
Priority Engine
      ↓
Smart Downlink
      ↓
Ground Station
Tech stack

Python for pretty much everything. PyTorch and Ultralytics YOLO for detection, OpenCV and NumPy for image handling. FastAPI on the backend, React on the frontend, Plotly for charts and Leaflet/MapLibre for the map view. ONNX + ONNX Runtime for the edge optimization piece. PostgreSQL/PostGIS if we end up needing persistent storage with geospatial queries — optional for now. Everything tracked in Git/GitHub.

Project layout
space-edge-ai/
│
├── ai_engine/
│   ├── detector.py
│   ├── inference.py
│   └── preprocessing.py
│
├── mission_engine/
│   ├── priority.py
│   ├── scheduler.py
│   ├── battery.py
│   └── bandwidth.py
│
├── backend/
│   ├── main.py
│   └── routes/
│
├── frontend/
│   └── src/
│
├── models/
├── datasets/
├── simulation/
├── tests/
│
├── requirements.txt
└── README.md

# Why we're building this

The end goal isn't just "satellites that take pictures and send them down." It's satellites that actually understand what they're looking at, decide what's worth telling us, and communicate that — on their own, without waiting around for someone on the ground to sort through the noise.

Don't transmit pixels. Transmit intelligence.
