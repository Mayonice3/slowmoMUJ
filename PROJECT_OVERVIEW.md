# AI Video Slow-Motion (ChronoFlow) - Project Overview

This document provides a comprehensive breakdown of the project structure, architecture, and the specific role of each file. This project implements advanced video frame interpolation using the **RIFE HDv3** model with a memory-efficient **streaming assembly** architecture.

---

## 🏛️ Project Architecture

The project is designed with a modular approach, separating the user interface, the processing engine, and the AI model logic. It follows a **Streaming Workflow** which processes video frames in pairs and writes them directly to the output file, avoiding the massive disk space and RAM overhead of traditional image-sequence methods.

### 📁 Directory Structure

```text
slowmoMUJ/
├── src/                    # Main source code
│   ├── ui/                 # Graphical User Interface
│   ├── engine/             # Core processing & AI logic
│   │   └── rife/           # RIFE Model package
│   └── main.py             # Command-line entry point
├── data/                   # Input/Output and reports
├── requirements.txt        # Dependency list
└── PROJECT_OVERVIEW.md     # This documentation
```

---

## 📄 File-by-File Explanation

### 1. Main Entry Points
*   **`src/ui/app.py`**: The Modern GUI.
    *   **Role**: Provides a professional, dark-mode window for users.
    *   **Key Features**: Built with `CustomTkinter`. Handles file selection, parameter configuration (multiplier, FPS), and features a real-time console log. It uses **Threading** to ensure the UI stays responsive during heavy AI computation.
*   **`src/main.py`**: The CLI (Command Line Interface).
    *   **Role**: A script for running the pipeline via terminal without a GUI. Useful for batch processing or automated workflows.

### 2. The Engine (`src/engine/`)
*   **`src/engine/interpolation.py`**: The Streaming Controller.
    *   **Role**: Orchestrates the video processing loop.
    *   **Logic**: Opens the video, reads frames, sends them to the RIFE model, and writes the results immediately to the output using `cv2.VideoWriter`. 
    *   **UI Integration**: Includes a "callback" system to update the GUI progress bar in real-time.
*   **`src/engine/rife_processor.py`**: The AI Wrapper.
    *   **Role**: Simplified interface for the complex RIFE model.
    *   **Logic**: Handles device detection (MPS for Mac, CUDA for NVIDIA, or CPU). It manages frame normalization, padding (to multiples of 32), and **FP16 Half-Precision** optimization for faster processing.

### 3. The AI Model (`src/engine/rife/`)
*   **`src/engine/rife/RIFE_HDv3.py`**: Model Wrapper.
    *   **Role**: Higher-level model management (loading weights, running inference). Updated to support dynamic device assignment for Apple Silicon.
*   **`src/engine/rife/IFNet_HDv3.py`**: Neural Network Architecture.
    *   **Role**: Defines the Intermediate Flow Network (IFNet) structure. This is the "brain" that calculates how objects move between two frames.
*   **`src/engine/rife/model/warplayer.py`**: Image Warping.
    *   **Role**: Performs the actual shifting of pixels using motion vectors.
    *   **Optimization**: Contains a custom workaround for Apple Silicon (MPS) to handle edge padding without crashing.
*   **`src/engine/rife/model/loss.py`**: Mathematical Loss Functions.
    *   **Role**: Originally used for training, but used here to provide structural calculations. Cleaned and optimized to be device-agnostic.
*   **`src/engine/rife/flownet.pkl`**: Pre-trained Weights.
    *   **Role**: The "learned memory" of the AI, containing millions of parameters trained on thousands of videos.

---

## 🚀 Key Technologies & Optimizations

1.  **RIFE HDv3**: A state-of-the-art AI model for video frame interpolation that produces smoother, more natural results than traditional linear blending.
2.  **Apple Silicon (MPS) Support**: Customized to run natively on Mac GPU (M1/M2/M3 chips) using Metal Performance Shaders.
3.  **FP16 Half-Precision**: Uses 16-bit floats instead of 32-bit. This results in a **2x-5x speedup** on modern hardware with nearly zero loss in visual quality.
4.  **Streaming Assembly**: Unlike many AI tools that save thousands of PNG files to your disk, this tool streams the output. This saves gigabytes of space and reduces SSD wear.
5.  **Threaded GUI**: The GUI runs on a separate thread from the AI, ensuring that users can still see logs and move the window while the video is processing.

---

## 🛠️ Usage for University Project
When documenting this for your project, you can highlight the **Mathematical Layer** (how RIFE calculates optical flow in `IFNet_HDv3.py`) and the **Systems Layer** (how Python handles multi-threading and video codecs in `app.py` and `interpolation.py`).
