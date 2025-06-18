from tkinter import *
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ecg_image_select_option:
    def __init__(self):
        self.master = 'ar_master'
        self.title = 'ECG Analysis'
        self.backround_color = '#111827'
        self.text_color = '#FFF'
        self.line_color = '#10B981'
        self.model = load_model('models/ecg_model.h5')  # You'll need to train/save this model

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
        canvas.create_text(w//2, 30, text="ECG IMAGE ANALYSIS", 
                          font=("Arial", 20), fill=self.text_color)
        
        # Upload Button
        btn_upload = Button(canvas, text="Upload ECG Image", 
                          command=lambda: self.upload_image(canvas, w, h),
                          bg=self.line_color, fg='white', font=('Arial', 12))
        canvas.create_window(w//2, 80, window=btn_upload)
        
        # Result Display Area
        self.result_label = Label(canvas, text="", bg=self.backround_color, 
                                fg='white', font=('Arial', 14))
        canvas.create_window(w//2, h-50, window=self.result_label)
        
        root.mainloop()
    
    def upload_image(self, canvas, w, h):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            # Process and predict
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0) / 255.0
            
            prediction = self.model.predict(img)
            class_idx = np.argmax(prediction)
            classes = ['Normal', 'Abnormal']
            result = classes[class_idx]
            
            # Display image and result
            self.display_image(canvas, file_path, w//2, h//2)
            self.result_label.config(text=f"Prediction: {result} (Confidence: {prediction[0][class_idx]:.2f})")
    
    def display_image(self, canvas, img_path, x, y):
        img = Image.open(img_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        
        # Clear previous image if exists
        if hasattr(self, 'img_label'):
            self.img_label.destroy()
        
        self.img_label = Label(canvas, image=img_tk)
        self.img_label.image = img_tk
        canvas.create_window(x, y, window=self.img_label)
