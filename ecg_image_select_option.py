import asyncio
import os
import time
from tkinter import Tk, messagebox, ttk
from tkinter import *
from tkinter.ttk import Treeview
from PIL import Image, ImageTk
import ar_master
import csv
import numpy as np
import cv2

from ecg_image_classification import ecg_image_classify
from ecg_image_classification1 import ecg_image_classify1


class ecg_image_select_option():
    path = ""
    feature = 0
    soil = ""
    ph = 0
    fname = ''

    def __init__(self):
        self.master = 'ar_master'
        self.title = 'Cardiovascular Disease'
        self.titlec = 'CARDIOVASCULAR DISEASE'
        self.backround_color = '#000'
        self.title_color = '#000'
        self.menu_backround_color = '#273746'
        self.text_color = '#FFF'
        self.backround_image = 'images/background_hd.jpg'
        self.account_no = ''
        self.blink_text_color = '#FFF'
        self.line_color = '#fff'
        self.body_color = '#34495e'
        self.button_backround_color = '#f39c12'
        self.current_button = "#10B981"
        self.menu_backround_color = '#003366'

        self.menu_backround_color_active = '#FF8000'
        self.menu_color_active = '#FFF'
        self.menu_color_disable = '#000'
    def set_window_design(self):
        root = Tk()
        w = 750
        h = 400
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.bg = ImageTk.PhotoImage(file='images/background_hd.jpg')
        root.title(self.title)
        root.resizable(False, False)
        bg = ImageTk.PhotoImage(file=self.backround_image)
        canvas = Canvas(root, width=200, height=300)
        canvas.pack(fill="both", expand=True)

        canvas.create_image(0, 0, image=bg, anchor=NW)

        canvas.create_rectangle(10, 10, w - 10, h - 10, outline=self.line_color, width=1)
        canvas.create_rectangle(8,8, w - 8, h - 8, outline=self.line_color, width=1)

        canvas.create_text(400, 40, text=self.titlec, font=("Times New Roman", 24), fill=self.text_color)
        def clickHandler(event):
            root.destroy()
            ar = ecg_image_classify()
            ar.home_window()
        image = Image.open('images/test.png')
        img = image.resize((125, 125))
        my_img = ImageTk.PhotoImage(img)
        image_id = canvas.create_image(250, 200, image=my_img)
        canvas.tag_bind(image_id, "<1>", clickHandler)
        admin_id = canvas.create_text(250, 300, text="TRAINING", font=("Times New Roman", 24), fill=self.text_color)
        canvas.tag_bind(admin_id, "<1>", clickHandler)

        def clickHandler1(event):
            root.destroy()
            ar = ecg_image_classify1()
            ar.home_window()
        image1 = Image.open('images/train.png')
        img1 = image1.resize((125, 125))
        my_img1 = ImageTk.PhotoImage(img1)
        image_id1 = canvas.create_image(500, 200, image=my_img1)
        canvas.tag_bind(image_id1, "<1>", clickHandler1)

        admin_id1 = canvas.create_text(500, 300, text="TESTING", font=("Times New Roman", 24), fill=self.text_color)
        canvas.tag_bind(admin_id1, "<1>", clickHandler1)

        def clickHandler2(event):
            root.destroy()






        root.mainloop()



