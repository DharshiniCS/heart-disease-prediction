from tkinter import *
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class echo_image_select_option:
    def __init__(self):
        self.master = 'ar_master'
        self.title = 'Echocardiogram Analysis'
        self.backround_color = '#111827'
        self.text_color = '#FFF'
        self.line_color = '#10B981'
        self.model = load_model('models/echo_model.h5')  # You'll need to train/save this model

    def set_window_design(self):
        root = Tk()
        root.title(self.title)
        w, h = 800, 600
        ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
        x, y = (ws/2)-(w/2), (hs/2)-(h/2)
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        
        # Main Canvas
        canvas = Canvas(root, bg=self.backround_color)
        canvas.pack(fill=BOTH, expand=True)
        
        # Title
        canvas.create_text(w//2, 30, text="ECHOCARDIOGRAM ANALYSIS", 
                         font=("Arial", 20), fill=self.text_color)
        
        # Upload Button
        btn_upload = Button(canvas, text="Upload Echo Image", 
                          command=lambda: self.upload_image(canvas, w, h),
                          bg=self.line_color, fg='white', font=('Arial', 12))
        canvas.create_window(w//2, 80, window=btn_upload)
        
        # Analysis Frame
        analysis_frame = Frame(canvas, bg='#1E293B')
        canvas.create_window(w//2, h//2, window=analysis_frame, width=600, height=400)
        
        # Result Display
        self.result_text = Text(analysis_frame, wrap=WORD, bg='#1E293B', 
                              fg='white', font=('Arial', 12))
        self.result_text.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        root.mainloop()
    
    def upload_image(self, canvas, w, h):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            # Process image
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0) / 255.0
            
            # Predict (example - adapt to your model)
            prediction = self.model.predict(img)
            conditions = ['Normal', 'Cardiomyopathy', 'Valve Disease']
            result_idx = np.argmax(prediction)
            
            # Display results
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, f"PREDICTION RESULTS:\n\n")
            self.result_text.insert(END, f"Primary Condition: {conditions[result_idx]}\n")
            self.result_text.insert(END, f"Confidence: {prediction[0][result_idx]:.2f}\n\n")
            self.result_text.insert(END, "Full Prediction Scores:\n")
            for i, cond in enumerate(conditions):
                self.result_text.insert(END, f"{cond}: {prediction[0][i]:.4f}\n")
