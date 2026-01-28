# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered video slow-motion tool using RIFE HDv3 (Real-Time Intermediate Flow Estimation) for frame interpolation. Takes a video, generates intermediate frames using neural network inference, and outputs a slow-motion video.

## Commands

### Run the GUI
```bash
python src/ui/app.py
```

### Run CLI pipeline
```bash
python src/main.py
```
Processes `video.mp4` from project root, outputs to `data/output_slowmo.mp4`.

### Install dependencies
```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, OpenCV, CustomTkinter, tqdm

## Architecture

### Processing Flow
1. **Entry Point** (`src/main.py` or `src/ui/app.py`) → initializes RIFEProcessor
2. **RIFEProcessor** (`src/engine/rife_processor.py`) → wraps RIFE model, handles device detection (MPS/CUDA/CPU), FP16 optimization
3. **Streaming Controller** (`src/engine/interpolation.py`) → reads video frame-by-frame, calls processor for interpolation, writes directly to output (no intermediate files)
4. **RIFE Model** (`src/engine/rife/`) → neural network that calculates optical flow between frames

### Key Design Decisions
- **Streaming assembly**: Frames are written directly to output video, avoiding disk-heavy PNG sequences
- **Device auto-detection**: Automatically uses MPS (Apple Silicon), CUDA (NVIDIA), or CPU
- **FP16 half-precision**: Enabled on GPU for 2-5x speedup
- **Recursive interpolation**: Supports 2x, 4x, 8x multipliers via recursive frame generation in `get_interpolated_frames()`

### Module Structure
- `src/engine/rife_processor.py` - High-level API: `RIFEProcessor.process_pair(img0, img1)` returns interpolated frame
- `src/engine/interpolation.py` - Video processing loop: `process_video_streaming()` with progress callback for GUI
- `src/engine/rife/RIFE_HDv3.py` - Model wrapper, loads `flownet.pkl` weights
- `src/engine/rife/IFNet_HDv3.py` - Neural network architecture (IFNet)
- `src/engine/rife/model/warplayer.py` - Pixel warping with MPS compatibility workaround

### Data Flow
```
Input Video → cv2.VideoCapture → frame pairs → RIFEProcessor → interpolated frames → cv2.VideoWriter → Output Video
```

## Important Notes

- Model weights file `flownet.pkl` (~12MB) must exist in `src/engine/rife/`
- Frames must be padded to multiples of 32 for the neural network (handled by RIFEProcessor)
- GUI runs inference in a separate thread to keep UI responsive
- Output goes to `data/` directory; first 20 frames also saved as PNGs to `data/report_frames/` for inspection
