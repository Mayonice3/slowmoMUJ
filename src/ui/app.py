import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

# Add root directory to sys.path to import engine modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.rife_processor import RIFEProcessor
from engine.interpolation import process_video_streaming

# --- Setup Appearance ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TextRedirector:
    """Helper to redirect stdout/stderr to a text widget in the UI"""
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str)
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self):
        pass

class SlowmoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Configuration ---
        self.title("AI Video Slow-Mo (RIFE v4.25)")
        self.geometry("900x650")

        # --- Variables ---
        self.input_video_path = ctk.StringVar(value="")
        self.processor = None
        
        # Grid Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT SIDEBAR (Settings) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI SlowMo", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 20))

        # 1. File Selection
        self.select_btn = ctk.CTkButton(self.sidebar_frame, text="Select Video", command=self.open_file)
        self.select_btn.pack(padx=20, pady=10)
        
        self.file_label = ctk.CTkLabel(self.sidebar_frame, text="No file selected", wraplength=180, font=ctk.CTkFont(size=11))
        self.file_label.pack(padx=20, pady=5)

        ctk.CTkLabel(self.sidebar_frame, text="--- Parameters ---").pack(pady=(20, 5))

        # 2. Expansion Factor (Multiplier)
        ctk.CTkLabel(self.sidebar_frame, text="Frames multiplier (2x, 4x, etc)").pack()
        self.multiplier_var = ctk.StringVar(value="2")
        self.multiplier_menu = ctk.CTkComboBox(self.sidebar_frame, values=["2", "4", "8"], variable=self.multiplier_var)
        self.multiplier_menu.pack(padx=20, pady=10)

        # 3. Target Output FPS
        ctk.CTkLabel(self.sidebar_frame, text="Output Playback FPS").pack()
        self.fps_slider = ctk.CTkSlider(self.sidebar_frame, from_=5, to=120, number_of_steps=115)
        self.fps_slider.set(30)
        self.fps_slider.pack(padx=20, pady=10)
        
        self.fps_label = ctk.CTkLabel(self.sidebar_frame, text="FPS: 30")
        self.fps_label.pack()
        self.fps_slider.configure(command=lambda v: self.fps_label.configure(text=f"FPS: {int(v)}"))

        # 4. RIFE Scale (Motion Handling)
        ctk.CTkLabel(self.sidebar_frame, text="Motion Handling (Scale)").pack(pady=(10, 0))
        self.scale_var = ctk.StringVar(value="1.0x (Standard)")
        self.scale_menu = ctk.CTkComboBox(
            self.sidebar_frame, 
            values=["1.0x (Standard)", "0.5x (Fast Motion)", "0.25x (Very Fast)", "0.125x (Extreme)"],
            variable=self.scale_var
        )
        self.scale_menu.pack(padx=20, pady=10)
        
        # Quality mode info label
        self.quality_info = ctk.CTkLabel(
            self.sidebar_frame, 
            text="1.0x: Normal\n0.5x: Fast Motion\n0.125x: Extreme Motion",
            wraplength=180,
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.quality_info.pack(pady=(0, 10))

        # 5. TTA Ensemble Switch
        self.ensemble_var = ctk.StringVar(value="off")
        self.ensemble_switch = ctk.CTkSwitch(
            self.sidebar_frame, 
            text="TTA / Ensemble", 
            variable=self.ensemble_var, 
            onvalue="on", 
            offvalue="off"
        )
        self.ensemble_switch.pack(padx=20, pady=(15, 5))
        
        ctk.CTkLabel(
            self.sidebar_frame, 
            text="Reduces artifacts\n(2x slower)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(pady=(0, 10))

        # --- MAIN CONTENT AREA ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)

        # Big Start Button
        self.run_btn = ctk.CTkButton(self.main_frame, text="START PROCESSING", height=50, 
                                     font=ctk.CTkFont(weight="bold"), fg_color="#E67E22", hover_color="#D35400",
                                     command=self.start_processing_thread)
        self.run_btn.grid(row=0, column=0, padx=20, pady=(40, 10), sticky="ew")

        # Stop Button (initially disabled/hidden)
        self.stop_btn = ctk.CTkButton(self.main_frame, text="STOP", height=40,
                                      font=ctk.CTkFont(weight="bold"), fg_color="#C0392B", hover_color="#922B21",
                                      state="disabled",
                                      command=self.stop_processing)
        self.stop_btn.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

        # Progress Bar
        self.prog_label = ctk.CTkLabel(self.main_frame, text="Status: Ready")
        self.prog_label.grid(row=2, column=0, padx=20, pady=5)
        
        self.progressbar = ctk.CTkProgressBar(self.main_frame)
        self.progressbar.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.progressbar.set(0)

        # Console Text Area
        ctk.CTkLabel(self.main_frame, text="Process Console Log:").grid(row=4, column=0, padx=20, pady=(20, 0), sticky="w")
        self.console_box = ctk.CTkTextbox(self.main_frame, height=300, corner_radius=10, state="disabled")
        self.console_box.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")

        # --- Initialize Console Redirection ---
        sys.stdout = TextRedirector(self.console_box)
        sys.stderr = TextRedirector(self.console_box)
        
        # Thread control
        self.stop_event = threading.Event()

    def open_file(self):
        filename = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4 *.mov *.avi")])
        if filename:
            self.input_video_path.set(filename)
            self.file_label.configure(text=f"Selected: {os.path.basename(filename)}")
    
    def update_progress(self, value):
        self.progressbar.set(value)
        self.prog_label.configure(text=f"Processing: {int(value * 100)}%")

    def start_processing_thread(self):
        if not self.input_video_path.get():
            print("❌ Please select a video file first.")
            return

        # Reset stop event
        self.stop_event.clear()

        # Disable UI components
        self.run_btn.configure(state="disabled", text="PROCESSING...")
        self.stop_btn.configure(state="normal")  # Enable Stop button
        self.select_btn.configure(state="disabled")
        self.multiplier_menu.configure(state="disabled")
        self.scale_menu.configure(state="disabled")
        self.ensemble_switch.configure(state="disabled")

        # Start heavy lifting in a separate thread
        thread = threading.Thread(target=self.run_inference)
        thread.daemon = True
        thread.start()
        
    def stop_processing(self):
        if not self.stop_event.is_set():
            self.stop_event.set()
            self.stop_btn.configure(state="disabled", text="Stopping...")
            print("\n[UI] Stop signal sent... waiting for current frame to finish.")

    def run_inference(self):
        try:
            # 1. Init model if not exists
            if self.processor is None:
                print("--- [UI] INITIALIZING RIFE AI MODEL ---")
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine", "rife")
                self.processor = RIFEProcessor(model_dir=model_dir)

            # 2. Run Processing
            input_path = self.input_video_path.get()
            base_project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_project_dir, "data")
            output_path = os.path.join(data_dir, "gui_output_slowmo.mp4")
            report_frames_dir = os.path.join(data_dir, "report_frames")
            
            # WIPE old report frames so they don't mix with new ones
            if os.path.exists(report_frames_dir):
                import shutil
                shutil.rmtree(report_frames_dir)
            os.makedirs(report_frames_dir)
            print(f"--- [UI] PREPARED CLEAN REPORT DIRECTORY ---")
            
            # Get values from UI
            multiplier = int(self.multiplier_var.get())
            playback_fps = float(self.fps_slider.get())
            ensemble_enabled = (self.ensemble_var.get() == "on")
            
            # Parse scale value
            scale_str = self.scale_var.get()
            if "0.5x" in scale_str:
                scale = 0.5
            elif "0.25x" in scale_str:
                scale = 0.25
            elif "0.125x" in scale_str:
                scale = 0.125
            else:
                scale = 1.0
            
            if not os.path.exists(data_dir): os.makedirs(data_dir)

            process_video_streaming(
                input_path=input_path,
                output_path=output_path,
                processor=self.processor,
                target_fps_multiplier=multiplier,
                output_fps=playback_fps,
                scale=scale,
                ensemble=ensemble_enabled,
                stop_event=self.stop_event,
                progress_callback=self.update_progress,
                save_png_count=20,
                png_output_dir=report_frames_dir
            )
            
            # Force 100% at the end only if not stopped
            if not self.stop_event.is_set():
                self.update_progress(1.0)
                print(f"!!==FINISHED==!! Saved to: {output_path}")
            else:
                print(f"!!==STOPPED==!! Processing cancelled.")

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
        finally:
            self.run_btn.configure(state="normal", text="START PROCESSING")
            self.stop_btn.configure(state="disabled", text="STOP")
            self.select_btn.configure(state="normal")
            self.multiplier_menu.configure(state="normal")
            self.scale_menu.configure(state="normal")
            self.ensemble_switch.configure(state="normal")

if __name__ == "__main__":
    app = SlowmoApp()
    app.mainloop()
