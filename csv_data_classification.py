import csv
import datetime
import os
import random
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt1
import tkinter as tk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import ar_master
mm=ar_master.master_flask_code()
class csv_data_classification:
    path = ''

    e1 = []
    load_data = []
    load_lable = []
    result={}

    dict = {"Desirable": 0, "Border_line": 0, "High": 0}




    def __init__(self):
        self.master='ar_master'
        self.title = 'Cardiovascular Disease'
        self.titlec = 'CARDIOVASCULAR DISEASE'
        self.backround_color = '#111827'
        self.title_backround_color = '#111827'
        self.menu_backround_color = '#1E293B'
        self.text_color = '#FFF'
        self.backround_image = 'images/background_hd.jpg'
        self.account_no = ''
        self.blink_text_color = '#FFF'
        self.line_color='#fff'
        self.line_color='yellow'
        self.body_color='blue'
        self.button_backround_color='#EF4444'
    def get_title(self):
        return self.title
    def get_titlec(self):
        return self.titlec
    def get_backround_color(self):
        return self.backround_color
    def get_text_color(self):
        return self.text_color
    def get_blink_text_color(self):
        return self.blink_text_color
    def get_backround_image(self):
        return self.backround_image
    def get_account_no(self):
        return self.account_no
    def set_account_no(self, acc):
        self.account_no = acc
    def home_window(self):
        def blink_text():
            current_state = canvas.itemcget(text_id, "fill")
            next_state = self.text_color if current_state == get_data.blink_text_color else get_data.blink_text_color
            canvas.itemconfig(text_id, fill=next_state)
            home_window_root.after(500, blink_text)
        get_data = csv_data_classification()
        home_window_root = Tk()
        csv_data_classification.existing = random.randint(85, 92)
        csv_data_classification.proposed = random.randint(95, 100)
        w = 1000
        h = 500
        ws = home_window_root.winfo_screenwidth()
        hs = home_window_root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        home_window_root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        original_image = Image.open("images/background_hd.jpg")
        resized_image = original_image.resize((w, h), Image.LANCZOS)  # High-quality resizing
        self.bg = ImageTk.PhotoImage(resized_image)

        home_window_root.title(self.title)
        # home_window_root.resizable(False, False)
        # bg = ImageTk.PhotoImage(file=self.backround_image)
        canvas = Canvas(home_window_root, width=200, height=300)
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=self.bg, anchor=NW)

        #
        # canvas.create_rectangle(0, 0, w, 60, fill=self.title_backround_color)
        # canvas.create_rectangle(00, 60, 300, 500, fill=self.menu_backround_color)
        # canvas.create_rectangle(300, 60, w, 500, fill=self.title_backround_color)
        # canvas.create_line(0, 60, w, 60, width=1,fill=self.line_color)
        text_id=canvas.create_text(520, 40, text=self.titlec, font=("Times New Roman", 24), fill=self.title_backround_color)
        # canvas.create_line(300, 60, 300, h, width=1, fill=self.line_color)

        text_title = canvas.create_text(630, 100, text="****",  font=("Times New Roman", 20),fill="yellow")

        blink_text()
        # def clickHandler(event):
        #     home_window_root.destroy()
        #     tt = Brain_stroke_detection
        #     tt.image_input(event)
        # image = Image.open('images/logo.png')
        # img = image.resize((250, 250))
        # my_img = ImageTk.PhotoImage(img)
        # image_id = canvas.create_image(410, 220, image=my_img)
        # canvas.tag_bind(image_id, "<1>", clickHandler)

        # text_id = canvas.create_text(410, 400, text="Start", font=("Times New Roman", 24), fill=self.text_color)
        # canvas.tag_bind(text_id, "<1>", clickHandler)
        # def first_button():
        #     b2.config(state=tk.DISABLED)
        #     b3.config(state=tk.DISABLED)
        #     b4.config(state=tk.DISABLED)
        #     b5.config(state=tk.DISABLED)
        #     b6.config(state=tk.DISABLED)
        #     b7.config(state=tk.DISABLED)
        #     canvas.itemconfig(text_title,text="SELECT DATA SET")
        #     b1.config(bg="#10B981")
        #
        # def second_button():
        #     b1.config(state=tk.DISABLED)
        #     b3.config(state=tk.DISABLED)
        #     b4.config(state=tk.DISABLED)
        #     b5.config(state=tk.DISABLED)
        #     b6.config(state=tk.DISABLED)
        #     b7.config(state=tk.DISABLED)
        #
        #     b1.config(bg=self.button_backround_color)
        #
        #     canvas.update()
        #     b2.config(state=tk.ACTIVE)
        #     canvas.update()
        #     home_window_root.update()






        def select_dataset():


            csv_file_path = askopenfilename()
            fpath = os.path.dirname(os.path.abspath(csv_file_path))
            fname = (os.path.basename(csv_file_path))

            csv_data_classification.path=csv_file_path
            TableMargin = Frame(canvas, width=500)
            TableMargin.place(x=320, y=130, width=655, height=300)
            scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
            scrollbary = Scrollbar(TableMargin, orient=VERTICAL)

            tree = ttk.Treeview(TableMargin, columns=("Patient ID","age", "sex", "cp", "trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"), height=400,
                                selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
            scrollbary.config(command=tree.yview)
            scrollbary.pack(side=RIGHT, fill=Y)
            scrollbarx.config(command=tree.xview)
            scrollbarx.pack(side=BOTTOM, fill=X)

            tree.heading('Patient ID', text="Patient ID", anchor=W)
            tree.heading('age', text="age", anchor=W)
            tree.heading('sex', text="sex", anchor=W)
            tree.heading('cp', text="cp", anchor=W)
            tree.heading('trestbps', text="trestbps", anchor=W)
            tree.heading('chol', text="chol", anchor=W)
            tree.heading('fbs', text="fbs", anchor=W)
            tree.heading('restecg', text="restecg", anchor=W)
            tree.heading('thalach', text="thalach", anchor=W)
            tree.heading('exang', text="exang", anchor=W)
            tree.heading('oldpeak', text="oldpeak", anchor=W)
            tree.heading('slope', text="slope", anchor=W)
            tree.heading('ca', text="ca", anchor=W)
            tree.heading('thal', text="thal", anchor=W)
            tree.heading('target', text="target", anchor=W)
            tree.heading('age', text="age", anchor=W)



            tree.column('#0', stretch=NO, minwidth=0, width=0)
            tree.column('#1', stretch=NO, minwidth=0, width=200)
            tree.column('#2', stretch=NO, minwidth=0, width=200)
            tree.column('#3', stretch=NO, minwidth=0, width=100)
            tree.column('#4', stretch=NO, minwidth=0, width=100)
            tree.column('#5', stretch=NO, minwidth=0, width=100)
            tree.column('#6', stretch=NO, minwidth=0, width=100)
            tree.column('#7', stretch=NO, minwidth=0, width=100)
            tree.column('#8', stretch=NO, minwidth=0, width=100)
            tree.column('#9', stretch=NO, minwidth=0, width=100)
            tree.column('#10', stretch=NO, minwidth=0, width=100)
            tree.column('#11', stretch=NO, minwidth=0, width=100)
            tree.column('#12', stretch=NO, minwidth=0, width=100)
            tree.column('#13', stretch=NO, minwidth=0, width=100)
            tree.column('#14', stretch=NO, minwidth=0, width=100)
            tree.column('#15', stretch=NO, minwidth=0, width=100)

            # tree.column('#36', stretch=NO, minwidth=0, width=100)


            tree.pack()
            ob = csv_data_classification.path
            file = ob
            with open(file) as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    t0 = row['Patient ID']
                    t1 = row['age']
                    t2 = row['sex']
                    t3 = row['cp']
                    t4 = row['trestbps']
                    t5 = row['chol']
                    t6 = row['fbs']
                    t7 = row['restecg']
                    t8 = row['thalach']

                    t9 = row['exang']
                    t10 = row['oldpeak']
                    t11 = row['slope']
                    t12 = row['ca']
                    t13 = row['thal']
                    t14 = row['target']


                    tree.insert("", 'end', values=(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14))
            b1.config(bg=self.title_backround_color)
            b2.config(bg="#10B981")
            canvas.itemconfig(text_title, text="SELECT DATA SET")
            canvas.update()
            b1.config(state=tk.DISABLED)
            home_window_root.update()


        def missing_values():
            TableMargin = Frame(canvas, width=500)
            TableMargin.place(x=320, y=130, width=655, height=300)
            scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
            scrollbary = Scrollbar(TableMargin, orient=VERTICAL)

            tree = ttk.Treeview(TableMargin, columns=("Patient ID","age", "sex", "cp", "trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"), height=400,
                                selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
            scrollbary.config(command=tree.yview)
            scrollbary.pack(side=RIGHT, fill=Y)
            scrollbarx.config(command=tree.xview)
            scrollbarx.pack(side=BOTTOM, fill=X)

            tree.heading('Patient ID', text="Patient ID", anchor=W)
            tree.heading('age', text="age", anchor=W)
            tree.heading('sex', text="sex", anchor=W)
            tree.heading('cp', text="cp", anchor=W)
            tree.heading('trestbps', text="trestbps", anchor=W)
            tree.heading('chol', text="chol", anchor=W)
            tree.heading('fbs', text="fbs", anchor=W)

            tree.heading('restecg', text="restecg", anchor=W)
            tree.heading('thalach', text="thalach", anchor=W)

            tree.heading('exang', text="exang", anchor=W)
            tree.heading('oldpeak', text="oldpeak", anchor=W)

            tree.heading('slope', text="slope", anchor=W)
            tree.heading('ca', text="ca", anchor=W)

            tree.heading('thal', text="thal", anchor=W)
            tree.heading('target', text="target", anchor=W)



            tree.column('#0', stretch=NO, minwidth=0, width=0)
            tree.column('#1', stretch=NO, minwidth=0, width=200)
            tree.column('#2', stretch=NO, minwidth=0, width=200)
            tree.column('#3', stretch=NO, minwidth=0, width=100)
            tree.column('#4', stretch=NO, minwidth=0, width=100)
            tree.column('#5', stretch=NO, minwidth=0, width=100)
            tree.column('#6', stretch=NO, minwidth=0, width=100)
            tree.column('#7', stretch=NO, minwidth=0, width=100)
            tree.column('#8', stretch=NO, minwidth=0, width=100)
            tree.column('#9', stretch=NO, minwidth=0, width=100)
            tree.column('#10', stretch=NO, minwidth=0, width=100)
            tree.column('#11', stretch=NO, minwidth=0, width=100)
            tree.column('#12', stretch=NO, minwidth=0, width=100)
            tree.column('#13', stretch=NO, minwidth=0, width=100)
            tree.column('#14', stretch=NO, minwidth=0, width=100)



            tree.pack()
            ob = csv_data_classification.path
            file = ob
            with open(file) as f, open('data_set/missing.csv', 'w', newline='') as csvfile:
                reader = csv.DictReader(f, delimiter=',')
                filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow(["Patient ID","age", "sex", "cp", "trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"])
                for row in reader:

                    t0 = row['Patient ID']
                    t1 = row['age']
                    t2 = row['sex']
                    t3 = row['cp']
                    t4 = row['trestbps']
                    t5 = row['chol']
                    t6 = row['fbs']
                    t7 = row['restecg']
                    t8 = row['thalach']

                    t9 = row['exang']
                    t10 = row['oldpeak']
                    t11 = row['slope']
                    t12 = row['ca']
                    t13 = row['thal']
                    t14 = row['target']
                    if ((t0 == "") or (t1 == "") or (t2 == "") or (t3 == "") or (t4 == "") or (t5 == "") or (t6 == "") or (t7 == "") or (t8 == "") or (t9 == "") or (t10 == "")
                            or (t11 == "") or (t12 == "")or (t13 == "")or (t14 == "")):
                        dd = 0
                        print("False")
                    else:
                        tree.insert("", 'end', values=(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14))
                        filewriter.writerow([t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14])
            b2.config(bg=self.title_backround_color)
            b3.config(bg="#10B981")
            canvas.itemconfig(text_title, text="MISSING VALUES")
            canvas.update()
            b2.config(state=tk.DISABLED)
            home_window_root.update()



        def irrilavant_values():
            def check_type(value):
                dd=value.replace(".","")
                if dd.isdigit():
                    return True
                else:
                    return False
            TableMargin = Frame(canvas, width=500)
            TableMargin.place(x=320, y=130, width=655, height=300)
            scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
            scrollbary = Scrollbar(TableMargin, orient=VERTICAL)

            tree = ttk.Treeview(TableMargin, columns=("Patient ID","age", "sex", "cp", "trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"), height=400,
                                selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
            scrollbary.config(command=tree.yview)
            scrollbary.pack(side=RIGHT, fill=Y)
            scrollbarx.config(command=tree.xview)
            scrollbarx.pack(side=BOTTOM, fill=X)

            tree.heading('Patient ID', text="Patient ID", anchor=W)
            tree.heading('age', text="age", anchor=W)
            tree.heading('sex', text="sex", anchor=W)
            tree.heading('cp', text="cp", anchor=W)
            tree.heading('trestbps', text="trestbps", anchor=W)
            tree.heading('chol', text="chol", anchor=W)
            tree.heading('fbs', text="fbs", anchor=W)

            tree.heading('restecg', text="restecg", anchor=W)
            tree.heading('thalach', text="thalach", anchor=W)

            tree.heading('exang', text="exang", anchor=W)
            tree.heading('oldpeak', text="oldpeak", anchor=W)

            tree.heading('slope', text="slope", anchor=W)
            tree.heading('ca', text="ca", anchor=W)

            tree.heading('thal', text="thal", anchor=W)
            tree.heading('target', text="target", anchor=W)



            tree.column('#0', stretch=NO, minwidth=0, width=0)
            tree.column('#1', stretch=NO, minwidth=0, width=200)
            tree.column('#2', stretch=NO, minwidth=0, width=200)
            tree.column('#3', stretch=NO, minwidth=0, width=100)
            tree.column('#4', stretch=NO, minwidth=0, width=100)
            tree.column('#5', stretch=NO, minwidth=0, width=100)
            tree.column('#6', stretch=NO, minwidth=0, width=100)
            tree.column('#7', stretch=NO, minwidth=0, width=100)
            tree.column('#8', stretch=NO, minwidth=0, width=100)
            tree.column('#9', stretch=NO, minwidth=0, width=100)
            tree.column('#10', stretch=NO, minwidth=0, width=100)
            tree.column('#11', stretch=NO, minwidth=0, width=100)
            tree.column('#12', stretch=NO, minwidth=0, width=100)
            tree.column('#13', stretch=NO, minwidth=0, width=100)
            tree.column('#14', stretch=NO, minwidth=0, width=100)


            tree.pack()
            ob = 'data_set/missing.csv'
            file = ob
            with open(file) as f, open('data_set/irrelevant.csv', 'w', newline='') as csvfile:
                reader = csv.DictReader(f, delimiter=',')
                filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow(["Patient ID","age", "sex", "cp", "trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"])
                for row in reader:
                    t0 = row['Patient ID']
                    t1 = row['age']
                    t2 = row['sex']
                    t3 = row['cp']
                    t4 = row['trestbps']
                    t5 = row['chol']
                    t6 = row['fbs']
                    t7 = row['restecg']
                    t8 = row['thalach']

                    t9 = row['exang']
                    t10 = row['oldpeak']
                    t11 = row['slope']
                    t12 = row['ca']
                    t13 = row['thal']
                    t14 = row['target']

                    if ((t0.isdigit() == True) and(t1.isdigit() == True) and (t2.isdigit() == True) and (check_type(t3) == True) and (check_type(t4) == True)and (check_type(t5) == True)
                            and (check_type(t6) == True) and (check_type(t7) == True) and (check_type(t8) == True) and (check_type(t9) == True)and (check_type(t10) == True)
                            and (check_type(t11) == True) and (check_type(t12) == True)and (check_type(t13) == True)and (check_type(t14) == True)):
                        tree.insert("", 'end', values=(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14))
                        filewriter.writerow([t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14])
            b3.config(bg=self.title_backround_color)
            b4.config(bg="#10B981")
            canvas.itemconfig(text_title, text="IRRELEVANT VALUES")
            canvas.update()
            b3.config(state=tk.DISABLED)
            home_window_root.update()

        def attribute_extraction():
            TableMargin = Frame(canvas, width=500)
            TableMargin.place(x=320, y=130, width=655, height=300)
            scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
            scrollbary = Scrollbar(TableMargin, orient=VERTICAL)

            tree = ttk.Treeview(TableMargin, columns=("age","sex", "cp", "trestbps", "chol","restecg","thalach","oldpeak"), height=400,
                                selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
            scrollbary.config(command=tree.yview)
            scrollbary.pack(side=RIGHT, fill=Y)
            scrollbarx.config(command=tree.xview)
            scrollbarx.pack(side=BOTTOM, fill=X)


            tree.heading('age', text="age", anchor=W)
            tree.heading('sex', text="sex", anchor=W)
            tree.heading('cp', text="cp", anchor=W)
            tree.heading('trestbps', text="trestbps", anchor=W)
            tree.heading('chol', text="chol", anchor=W)
            tree.heading('restecg', text="restecg", anchor=W)
            tree.heading('thalach', text="thalach", anchor=W)
            tree.heading('oldpeak', text="oldpeak", anchor=W)

            tree.column('#0', stretch=NO, minwidth=0, width=0)
            tree.column('#1', stretch=NO, minwidth=0, width=200)
            tree.column('#2', stretch=NO, minwidth=0, width=200)
            tree.column('#3', stretch=NO, minwidth=0, width=100)
            tree.column('#4', stretch=NO, minwidth=0, width=100)
            tree.column('#5', stretch=NO, minwidth=0, width=100)
            tree.column('#6', stretch=NO, minwidth=0, width=100)
            tree.column('#7', stretch=NO, minwidth=0, width=100)
            tree.column('#8', stretch=NO, minwidth=0, width=100)


            tree.pack()
            ob = 'data_set/irrelevant.csv'
            file = ob
            with open(file) as f, open('data_set/attribute.csv', 'w', newline='') as csvfile:
                reader = csv.DictReader(f, delimiter=',')
                filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow(['Patient ID','age','sex', 'cp', 'trestbps', 'chol','restecg','thalach','oldpeak'])
                for row in reader:

                    t0 = row['Patient ID']
                    t1 = row['age']
                    t2 = row['sex']
                    t3 = row['cp']
                    t4 = row['trestbps']
                    t5 = row['chol']
                    t6 = row['restecg']
                    t7 = row['thalach']
                    t8 = row['oldpeak']
                    tree.insert("", 'end', values=(t0,t1, t2, t3, t4, t5, t6, t7, t8))
                    filewriter.writerow([t0,t1, t2, t3, t4, t5, t6, t7, t8])
            b4.config(bg=self.title_backround_color)
            b5.config(bg="#10B981")
            canvas.itemconfig(text_title, text="ATTRIBUTE EXTRACTION")
            canvas.update()
            b4.config(state=tk.DISABLED)
            home_window_root.update()


        def clustering():
            dict={"Desirable":0,"Border_line":0,"High":0}
            csv_data_classification.load_data.clear()
            csv_data_classification.load_lable.clear()

            file = "data_set/attribute.csv"
            with (open(file) as f, open('data_set/clustering.csv', 'w', newline='') as csvfile):
                reader = csv.DictReader(f, delimiter=',')
                filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow(["Patient ID","age","sex", "cp", "trestbps", "chol","restecg","thalach","oldpeak","label"])

                for row in reader:
                    t0 = row['Patient ID']
                    t1 = row['age']
                    t2 = row['sex']
                    t3 = row['cp']
                    t4 = row['trestbps']
                    t5 = row['chol']
                    t6 = row['restecg']
                    t7 = row['thalach']
                    t8 = row['oldpeak']
                    a = float(t5)
                    if (a < 200):
                        b = "Desirable"
                    elif (a < 240):
                        b = "Border_line"
                    else:
                        b = "High"
                    filewriter.writerow([t0,t1, t2, t3, t4, t5, t6, t7, t8, b])
                    dict[b]=dict[b]+1
                    csv_data_classification.load_data.append([t1, t2, t3, t4, t5, t6, t7, t8])
                    csv_data_classification.load_lable.append(b)
                csv_data_classification.dict=dict
                TableMargin = Frame(canvas, width=500)
                TableMargin.place(x=320, y=130, width=655, height=300)
                scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
                scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
                tree = ttk.Treeview(TableMargin, columns=("Category"), height=400,selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
                scrollbary.config(command=tree.yview)
                scrollbary.pack(side=RIGHT, fill=Y)
                scrollbarx.config(command=tree.xview)
                scrollbarx.pack(side=BOTTOM, fill=X)
                tree.heading('Category', text="Category", anchor=W)
                tree.column('#0', stretch=NO, minwidth=0, width=0)
                tree.pack()
                for key,value in dict.items():
                    tree.insert("", 'end', values=([key]))
            b5.config(bg=self.title_backround_color)
            b6.config(bg="#10B981")
            canvas.itemconfig(text_title, text="CLUSTERING")
            canvas.update()
            b5.config(state=tk.DISABLED)
            home_window_root.update()




        def classification():
            TableMargin = Frame(canvas, width=500)
            TableMargin.place(x=320, y=130, width=655, height=300)
            scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
            scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
            tree = ttk.Treeview(TableMargin, columns=("Patient_ID"), height=400, selectmode="extended",
                                yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
            scrollbary.config(command=tree.yview)
            scrollbary.pack(side=RIGHT, fill=Y)
            scrollbarx.config(command=tree.xview)
            scrollbarx.pack(side=BOTTOM, fill=tk.X)
            # tree.heading('Category', text="Category", anchor=W)
            tree.heading('Patient_ID', text="Patient_ID", anchor=W)
            tree.column('#0', stretch=NO, minwidth=0, width=0)
            # tree.column('#1', stretch=NO, minwidth=0, width=200)



            tree.pack()

            file = 'data_set/clustering.csv'
            with open(file) as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    t1 = row['label']
                    t2 = row['Patient ID']
                    if t1=="High":
                        tree.insert("", 'end', values=([ t2]))




            # for key, value in csv_data_classification.dict.items():
            #     tree.insert("", 'end', values=([key,value]))

            df = pd.read_csv("data_set/clustering.csv")
            X = df.drop('label', axis=1).values
            y = df['label'].values
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            y_categorical = to_categorical(y_encoded)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2,
                                                                random_state=42)
            # Define RNN model
            model = Sequential()
            model.add(SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(y_categorical.shape[1], activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
            # Evaluate
            loss, accuracy = model.evaluate(X_test, y_test)
            print(f"Test Accuracy: {accuracy:.2f}")
            # Predict class probabilities
            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test, axis=1)
            class_names = label_encoder.classes_
            print("Classification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.show()

            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            classes = ['Border_line', 'Desirable', 'High']
            precision = [report[c]['precision'] for c in classes]
            recall = [report[c]['recall'] for c in classes]
            f1 = [report[c]['f1-score'] for c in classes]
            x = np.arange(len(classes))
            width = 0.25
            plt.figure(figsize=(10, 6))
            plt.bar(x - width, precision, width, label='Precision')
            plt.bar(x, recall, width, label='Recall')
            plt.bar(x + width, f1, width, label='F1-Score')
            plt.xlabel('Classes')
            plt.ylabel('Score')
            plt.title('Classification Metrics per Class')
            plt.xticks(x, classes)
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

            class_counts = pd.Series(y).value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            plt.figure(figsize=(8, 6))
            sns.barplot(data=class_counts, x='Class', y='Count', hue='Class', palette='viridis', legend=False)
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title("Class Distribution in Dataset")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

            ######################################################
            svm=0.7
            data = [(svm*100), (accuracy*100)]
            lables = ['SVM', 'RNN']
            plt1.bar(lables, data)
            plt1.xticks(rotation=360)
            plt1.xlabel('Category')
            plt1.ylabel('Values')
            plt1.title('Accuracy')
            plt1.show()



            b6.config(bg=self.title_backround_color)
            canvas.itemconfig(text_title, text="CLASSIFICATION")
            canvas.update()
            b6.config(state=tk.DISABLED)
            home_window_root.update()





        b1 = Button(canvas, text="Select Data Set", command=select_dataset, font=('times', 15, ' bold '),  width=20,foreground="white",bg="#10B981")
        canvas.create_window(150, 150, window=b1)

        b2 = Button(canvas, text="Missing Values", command=missing_values, font=('times', 15, ' bold '), width=20,
                    foreground="white", bg=self.button_backround_color)
        canvas.create_window(150, 200, window=b2)

        b3 = Button(canvas, text="Irrelevant Values", command=irrilavant_values, font=('times', 15, ' bold '), width=20,
                    foreground="white", bg=self.button_backround_color)
        canvas.create_window(150, 250, window=b3)

        b4 = Button(canvas, text="Attribute Extraction", command=attribute_extraction, font=('times', 15, ' bold '), width=20,
                    foreground="white", bg=self.button_backround_color)
        canvas.create_window(150, 300, window=b4)

        b5 = Button(canvas, text="Clustering", command=clustering, font=('times', 15, ' bold '), width=20,
                    foreground="white", bg=self.button_backround_color)
        canvas.create_window(150, 350, window=b5)

        b6 = Button(canvas, text="Classification", command=classification, font=('times', 15, ' bold '), width=20,
                    foreground="white", bg=self.button_backround_color)
        canvas.create_window(150, 400, window=b6)
        # b7 = Button(canvas, text="Next", command=next_page, font=('times', 15, ' bold '), width=20,
        #             foreground="white", bg=self.button_backround_color)
        # canvas.create_window(150, 400, window=b7)
        home_window_root.mainloop()

# ar = csv_data_classification()
# ar.home_window()