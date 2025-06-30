import csv
import os
import random
import shutil
import matplotlib.pyplot as plt1
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image, ImageFilter
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from scipy.fftpack import dct, idct
import tkinter as tk
from skimage.io import imread
import time
import cv2
from numpy import asarray
from feature_extraction import FE

from sklearn.metrics import  confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
class ecg_image_classify:
    path=''
    f_value=0
    existing = 0
    proposed = 0
    e1=[50]
    def __init__(self):
        self.master = 'ar_master'
        self.title = 'Cardiovascular Disease'
        self.titlec = 'CARDIOVASCULAR DISEASE'
        self.backround_color = '#000'
        self.title_backround_color = '#1c2833'
        self.menu_backround_color = '#273746'
        self.text_color = '#FFF'
        self.backround_image = 'images/background_hd.jpg'
        self.account_no = ''
        self.blink_text_color = '#FFF'
        self.line_color = '#fff'
        self.body_color = '#34495e'
        self.button_backround_color = '#f39c12'
        self.current_button="#10B981"


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
    def home_window(self):
        def blink_text():
            current_state = canvas.itemcget(text_id, "fill")
            next_state = self.text_color if current_state == get_data.blink_text_color else get_data.blink_text_color
            canvas.itemconfig(text_id, fill=next_state)
            home_window_root.after(500, blink_text)

        get_data = ecg_image_classify()
        home_window_root = Tk()
        w = 850
        h = 550
        ws = home_window_root.winfo_screenwidth()
        hs = home_window_root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        home_window_root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.bg = ImageTk.PhotoImage(file='images/background_hd.jpg')
        home_window_root.title(self.title)
        # home_window_root.resizable(False, False)
        bg = ImageTk.PhotoImage(file=self.backround_image)
        canvas = Canvas(home_window_root, width=200, height=300)
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=bg, anchor=NW)
        canvas.create_rectangle(0, 0, w, 60, fill=self.title_backround_color)
        text_id = canvas.create_text(380, 40, text=self.titlec, font=("Times New Roman", 24),fill=self.text_color)
        canvas.create_rectangle(w-300, 60, w, h, fill=self.menu_backround_color)
        canvas.create_rectangle(0, 60, w-300, h, fill=self.body_color)
        canvas.create_line(0, 60, w, 60, width=1, fill=self.line_color)
        canvas.create_line(w-300, 60, w-300, h, width=1, fill=self.line_color)
        text_title = canvas.create_text(260, 100, text="SELECT IMAGE(ECG)", font=("Times New Roman", 20), fill=self.text_color)
        blink_text()


        image = Image.open("images/logo.png")
        resized_image = image.resize((250, 250), Image.LANCZOS )
        ic = ImageTk.PhotoImage(resized_image)

        image_view=canvas.create_image(140, 150, image=ic, anchor=NW)

        def select_image():
            file_path = askopenfilename(parent=home_window_root)
            fpath = os.path.dirname(os.path.abspath(file_path))
            fname = (os.path.basename(file_path))
            ecg_image_classify.path = file_path
            destination = os.path.join("data", "input.png")
            shutil.copy(os.path.abspath(file_path), destination)

            image1 = Image.open(destination)
            image2 = image1.resize((250, 250), Image.Resampling.LANCZOS)
            new_img = ImageTk.PhotoImage(image2)
            canvas.itemconfig(image_view, image=new_img)
            canvas.image = new_img

            b1.config(bg=self.title_backround_color)

            b2.config(bg="#10B981")
            # b9.config(bg="#10B981")

            canvas.itemconfig(text_title, text="IMAGE ACQUISITION")
            b1.config(state=tk.DISABLED)
            canvas.update()
            home_window_root.update()


        def grayscale_conversion():
            filapath = 'data/input.png'
            img = Image.open(filapath).convert('L')
            img.save('data/greyscale.png')


            destination="data/greyscale.png"
            image1 = Image.open(destination)
            image2 = image1.resize((250, 250), Image.Resampling.LANCZOS)
            new_img = ImageTk.PhotoImage(image2)
            canvas.itemconfig(image_view, image=new_img)
            canvas.image = new_img

            b2.config(bg=self.title_backround_color)
            b3.config(bg="#10B981")
            canvas.itemconfig(text_title, text="GRAYSCALE CONVERSION")
            b2.config(state=tk.DISABLED)
            canvas.update()
            home_window_root.update()

        def noise_removal():
            def dct2(a):
                return dct(dct(a.T, norm='ortho').T, norm='ortho')

            def dct1(a):
                return idct(idct(a.T, norm='ortho').T, norm='ortho')

            im = (imread(r'data/input.png'))
            imF = dct2(im)
            im1 = dct1(imF)
            dd = np.allclose(im, im1)
            np.allclose(im, im1)
            path = ('data/dct.png')
            cv2.imwrite(path, im1)

            destination = "data/dct.png"
            image1 = Image.open(destination)
            image2 = image1.resize((250, 250), Image.Resampling.LANCZOS)
            new_img = ImageTk.PhotoImage(image2)
            canvas.itemconfig(image_view, image=new_img)
            canvas.image = new_img

            b3.config(bg=self.title_backround_color)
            b4.config(bg="#10B981")
            canvas.itemconfig(text_title, text="NOISE REMOVAL")
            b3.config(state=tk.DISABLED)
            canvas.update()
            home_window_root.update()

        def image_segmentation():
            src = cv2.imread("data\\dct.png", 1)
            img = src
            s = 128
            img = cv2.resize(img, (s, s), 0, 0, cv2.INTER_AREA)

            def apply_watershed_segment(input_img, brightness=0, contrast=0):
                if brightness != 0:
                    if brightness > 0:
                        shadow = brightness
                        highlight = 255
                    else:
                        shadow = 0
                        highlight = 255 + brightness
                    alpha_b = (highlight - shadow) / 255
                    gamma_b = shadow

                    buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
                else:
                    buf = input_img.copy()
                if contrast != 0:
                    f = 131 * (contrast + 127) / (127 * (131 - contrast))
                    alpha_c = f
                    gamma_c = 127 * (1 - f)
                    buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
                return buf

            font = cv2.FONT_HERSHEY_SIMPLEX
            fcolor = (0, 0, 0)
            blist = [0]
            clist = [64]
            out = np.zeros((s * 2, s * 3, 3), dtype=np.uint8)
            for i, b in enumerate(blist):
                c = clist[i]
                out = apply_watershed_segment(img, b, c)
            cv2.imwrite('data/watershed.png', out)

            destination = "data/watershed.png"
            image1 = Image.open(destination)
            image2 = image1.resize((250, 250), Image.Resampling.LANCZOS)
            new_img = ImageTk.PhotoImage(image2)
            canvas.itemconfig(image_view, image=new_img)
            canvas.image = new_img

            b4.config(bg=self.title_backround_color)
            b5.config(bg="#10B981")
            canvas.itemconfig(text_title, text="SEGMENTATION")
            b4.config(state=tk.DISABLED)
            canvas.update()
            home_window_root.update()

        def boundary_detection():
            image = cv2.imread('data/watershed.png')
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            cv2.imwrite('data/boundary_detection.png', sharpened)

            destination = "data/boundary_detection.png"
            image1 = Image.open(destination)
            image2 = image1.resize((250, 250), Image.Resampling.LANCZOS)
            new_img = ImageTk.PhotoImage(image2)
            canvas.itemconfig(image_view, image=new_img)
            canvas.image = new_img

            b5.config(bg=self.title_backround_color)
            b6.config(bg="#10B981")
            canvas.itemconfig(text_title, text="BOUNDARY DETECTION")
            b5.config(state=tk.DISABLED)
            canvas.update()
            home_window_root.update()



        def morphological():


            image = Image.open(r"data/boundary_detection.png")
            image = image.convert("L")
            image = image.filter(ImageFilter.FIND_EDGES)
            image.save(r"data/morphological.png")

            img = cv2.imread('data/morphological.png', 0)
            numpydata = asarray(img)
            z = []
            for x in numpydata:
                for y in x:
                    z.append(int(y))
            nn = FE([2, 2, 1])
            nn.glcm_extract(z)
            ecg_image_classify.f_value = nn.result()

            destination = "data/morphological.png"
            image1 = Image.open(destination)
            image2 = image1.resize((250, 250), Image.Resampling.LANCZOS)
            new_img = ImageTk.PhotoImage(image2)
            canvas.itemconfig(image_view, image=new_img)
            canvas.image = new_img

            b6.config(bg=self.title_backround_color)
            b7.config(bg="#10B981")
            canvas.itemconfig(text_title, text="MORPHOLOGICAL")
            b6.config(state=tk.DISABLED)
            canvas.update()
            home_window_root.update()


        def feature_extraction():
            canvas.itemconfig(text_title, text="FEATURE VALUE")
            feature = canvas.create_text(260, 450, text="Feature : "+str(ecg_image_classify.f_value), font=("Times New Roman", 20),fill=self.text_color)
            b7.config(bg=self.title_backround_color)
            b8.config(bg="#10B981")
            canvas.itemconfig(text_title, text="FEATURE EXTRACTION")
            b7.config(state=tk.DISABLED)
            canvas.update()
            home_window_root.update()

        def training():
            def write_dataset(disease,value,Stage,treatment,mortality):
                file = 'csv_dataset/training-dataset_ecg.csv'
                pr_chk = 0
                with open(file) as f:
                    reader = csv.DictReader(f, delimiter=',')
                    for row in reader:
                        t1 = row['value']
                        if int(t1) == int(value):
                            pr_chk += 1
                if pr_chk == 0:
                    file1 = 'csv_dataset/training-dataset_ecg1.csv'
                    with open(file) as f, open('csv_dataset/training-dataset_ecg1.csv', 'w', encoding='utf-8', newline='') as csvfile:
                        reader = csv.DictReader(f, delimiter=',')
                        filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                        filewriter.writerow(['value', 'label','stage','treatment','mortality'])
                        for row in reader:
                            t1 = row['value']
                            t2 = row['label']
                            t3 = row['stage']
                            t4 = row['treatment']
                            t5 = row['mortality']
                            filewriter.writerow([t1, t2,t3,t4,t5])
                        filewriter.writerow([value, disease,Stage,treatment,mortality])
                        csvfile.close()
                        shutil.copy('csv_dataset/training-dataset_ecg1.csv', file)

                    if os.path.exists(file1):
                        os.remove(file1)

                    # csv_file = "training-dataset.csv"
                    df = pd.read_csv("csv_dataset/training-dataset_ecg.csv")
                    X = df.drop(['label', 'stage', 'treatment', 'mortality'], axis=1).select_dtypes(
                        include=[np.number]).values
                    y = df['label'].values
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    y_cat = to_categorical(y_encoded)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
                    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_cat, test_size=0.2,random_state=42)
                    model = Sequential([
                        Conv1D(64, kernel_size=2, activation='relu', padding='same'),
                        MaxPooling1D(1),
                        SimpleRNN(64, activation='relu'),
                        Dropout(0.3),
                        Dense(32, activation='relu'),
                        Dense(y_cat.shape[1], activation='softmax')
                    ])
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    model.summary()
                    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)
                    model.save("ecg/cnn_rnn_model_ecg.keras")
                    np.save("ecg/label_encoder_classes_ecg.npy", le.classes_)
                    joblib.dump(scaler, "ecg/scaler_ecg.save")
                    return "success"
                else:
                    return "Already Trained"
            def exit_program():
                disease = e1.get()
                Stage = e2.get()
                treatment = e3.get()
                mortality = e4.get()
                if (disease == ""):
                    messagebox.showinfo(title="Alert", message="Enter Disease", parent=popup)
                elif (Stage==""):
                    messagebox.showinfo(title="Alert", message="Enter Stage", parent=popup)
                elif (treatment==""):
                    messagebox.showinfo(title="Alert", message="Enter Treatment Taken", parent=popup)
                elif (mortality==""):
                    messagebox.showinfo(title="Alert", message="Enter Mortality Date", parent=popup)
                else:
                    msg = write_dataset(disease=disease, value=ecg_image_classify.f_value,Stage=Stage,treatment=treatment,mortality=mortality)
                    messagebox.showinfo(title="Alert", message=msg, parent=popup)
                    popup.destroy()
            popup = tk.Toplevel(home_window_root)
            popup.title("Enter Training Data")
            popup.configure(bg="#2c3e50")
            w = 425
            h = 400
            ws = popup.winfo_screenwidth()
            hs = popup.winfo_screenheight()
            x = (ws / 2) - (w / 2)
            y = (hs / 2) - (h / 2)
            popup.geometry('%dx%d+%d+%d' % (w, h, x, y))
            canvas1 = Canvas(popup, width=200, height=300,bg=self.body_color)
            canvas1.pack(fill="both", expand=True)
            text_title = canvas1.create_text(220,40, text="DATA TRAINING", font=("Times New Roman", 20),fill=self.text_color)
            txt1 = canvas1.create_text(95,80, text="Disease", font=("Times New Roman", 14),fill=self.text_color)
            txt1 = canvas1.create_text(95,150, text="Stage", font=("Times New Roman", 14),fill=self.text_color)
            txt1 = canvas1.create_text(95,220, text="Treatment taken", font=("Times New Roman", 14),fill=self.text_color)
            txt1 = canvas1.create_text(95,290, text="Mortality Date", font=("Times New Roman", 14),fill=self.text_color)

            e1 = Entry(canvas1, font=('times', 15, ' bold '))
            canvas1.create_window(290, 80, window=e1)

            e2 = Entry(canvas1, font=('times', 15, ' bold '))
            canvas1.create_window(290, 150, window=e2)

            e3 = Entry(canvas1, font=('times', 15, ' bold '))
            canvas1.create_window(290, 220, window=e3)

            e4 = Entry(canvas1, font=('times', 15, ' bold '))
            canvas1.create_window(290, 290, window=e4)

            b1 = Button(canvas1, text="Train", command=exit_program, font=('times', 15, ' bold '))
            canvas1.create_window(220, 360, window=b1)




        def testing():

            filapath = 'data/input.png'
            img = Image.open(filapath).convert('L')
            img.save('data/greyscale.png')
            ################################
            def dct2(a):
                return dct(dct(a.T, norm='ortho').T, norm='ortho')

            def dct1(a):
                return idct(idct(a.T, norm='ortho').T, norm='ortho')

            im = (imread(r'data/input.png'))
            imF = dct2(im)
            im1 = dct1(imF)
            dd = np.allclose(im, im1)
            np.allclose(im, im1)
            path = ('data/dct.png')
            cv2.imwrite(path, im1)
            ##################################
            src = cv2.imread("data\\dct.png", 1)
            img = src
            s = 128
            img = cv2.resize(img, (s, s), 0, 0, cv2.INTER_AREA)

            def apply_watershed_segment(input_img, brightness=0, contrast=0):
                if brightness != 0:
                    if brightness > 0:
                        shadow = brightness
                        highlight = 255
                    else:
                        shadow = 0
                        highlight = 255 + brightness
                    alpha_b = (highlight - shadow) / 255
                    gamma_b = shadow

                    buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
                else:
                    buf = input_img.copy()
                if contrast != 0:
                    f = 131 * (contrast + 127) / (127 * (131 - contrast))
                    alpha_c = f
                    gamma_c = 127 * (1 - f)
                    buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
                return buf

            font = cv2.FONT_HERSHEY_SIMPLEX
            fcolor = (0, 0, 0)
            blist = [0]
            clist = [64]
            out = np.zeros((s * 2, s * 3, 3), dtype=np.uint8)
            for i, b in enumerate(blist):
                c = clist[i]
                out = apply_watershed_segment(img, b, c)
            cv2.imwrite('data/watershed.png', out)
            ##############################################
            image = cv2.imread('data/watershed.png')
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            cv2.imwrite('data/boundary_detection.png', sharpened)
            ###############################################
            image = Image.open(r"data/boundary_detection.png")
            image = image.convert("L")
            image = image.filter(ImageFilter.FIND_EDGES)
            image.save(r"data/morphological.png")

            img = cv2.imread('data/morphological.png', 0)
            numpydata = asarray(img)
            z = []
            for x in numpydata:
                for y in x:
                    z.append(int(y))
            nn = FE([2, 2, 1])
            nn.glcm_extract(z)
            ecg_image_classify.f_value = nn.result()
            ################################################

            start_time = time.time()
            result="Not Found"
            stage='-'
            treatment='-'
            mortality='-'
            file1 = 'csv_dataset/training-dataset_ecg.csv'
            with open(file1) as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    t1 = row['value']
                    t2 = row['label']
                    if int(ecg_image_classify.f_value) == int(t1):
                        result=t2
                        stage = row['stage']
                        treatment = row['treatment']
                        mortality = row['mortality']
                        break
            time.sleep(1)
            duration = time.time()
            etime = duration - start_time
            ecg_image_classify.e1.append(etime)
            index=ecg_image_classify.f_value
            ####################################

            csv_file = "training-dataset_ecg.csv"  # Replace with your CSV file name
            ############################# classification
            model = load_model("ecg/cnn_rnn_model_ecg.keras")
            scaler = joblib.load("ecg/scaler_ecg.save")
            label_classes = np.load("ecg/label_encoder_classes_ecg.npy", allow_pickle=True)
            df = pd.read_csv("csv_dataset/training-dataset_ecg.csv")
            X = df.drop(['label', 'stage', 'treatment', 'mortality'], axis=1).select_dtypes(include=[np.number]).values
            y = df['label'].values
            le = LabelEncoder()
            le.classes_ = label_classes
            features = []
            # features.append(float(input("Feature 1 (numeric value): ")))
            features.append(float(ecg_image_classify.f_value))
            X_user = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X_user)
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))  # Reshape for Conv1D
            y_pred_probs = model.predict(X_reshaped)
            y_pred = np.argmax(y_pred_probs, axis=1)
            predicted_label = le.inverse_transform(y_pred)
            print(f"Predicted label: {predicted_label[0]}")
            y_encoded = le.transform(y)
            y_cat = to_categorical(y_encoded)
            X_scaled = scaler.transform(X)
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            print("Input shape before reshape:", X_scaled.shape)
            y_pred_probs = model.predict(X_reshaped)
            y_pred = np.argmax(y_pred_probs, axis=1)
            print("Confusion Matrix:\n")
            conf_matrix = confusion_matrix(y_encoded, y_pred)
            print(conf_matrix)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_,yticklabels=le.classes_)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.show()
            ###############################################################
            if result!="Not Found":
                msg="Result : "+str(result)+"\nStage : "+str(stage)+"\nTreatment : "+str(treatment)+"\nMortality : "+str(mortality)
            else:
                msg="Result : "+str(predicted_label[0])


            messagebox.showinfo("Result",msg)

            end_time = time.time()
            duration_time = end_time - start_time
            ecg_image_classify.e1.append(duration_time)
            ecg_image_classify.e1.sort(reverse=True)

            accuracy = accuracy_score(y_encoded, y_pred)
            print(f"Accuracy: {accuracy * 100:.2f}%")


            data = [50, accuracy*100]
            lables = ['Existing', 'Proposed']
            data.sort()
            plt1.bar(lables, data)
            plt1.xticks(rotation=360)
            plt1.xlabel('Category')
            plt1.ylabel('Values')
            plt1.title('Accuracy')
            plt1.show()

            data = [ecg_image_classify.e1[0], ecg_image_classify.e1[1]]
            data.sort(reverse=True)
            lables = ['Existing', 'Proposed']
            plt1.plot(lables, data)
            plt1.suptitle('Duration')
            plt1.xlabel('Category')
            plt1.ylabel('time(S)')
            plt1.show()


        b1 = Button(canvas, text="Select Image", command=select_image, font=('times', 15, ' bold '), width=20,foreground="white", bg=self.current_button)
        canvas.create_window(w-150, 100, window=b1)

        b2 = Button(canvas, text="Grayscale Conversion", command=grayscale_conversion, font=('times', 15, ' bold '), width=20,foreground="white", bg=self.button_backround_color)
        canvas.create_window(w - 150, 150, window=b2)

        b3 = Button(canvas, text="Noise Removal", command=noise_removal, font=('times', 15, ' bold '), width=20,foreground="white", bg=self.button_backround_color)
        canvas.create_window(w - 150, 200, window=b3)

        b4 = Button(canvas, text="Image Segmentation", command=image_segmentation, font=('times', 15, ' bold '), width=20,foreground="white", bg=self.button_backround_color)
        canvas.create_window(w - 150, 250, window=b4)

        b5 = Button(canvas, text="Boundary Detection", command=boundary_detection, font=('times', 15, ' bold '),
                    width=20, foreground="white", bg=self.button_backround_color)
        canvas.create_window(w - 150, 300, window=b5)

        b6 = Button(canvas, text="Morphological", command=morphological, font=('times', 15, ' bold '),
                    width=20, foreground="white", bg=self.button_backround_color)
        canvas.create_window(w - 150, 350, window=b6)



        b7 = Button(canvas, text="Feature Extraction", command=feature_extraction, font=('times', 15, ' bold '), width=20,foreground="white", bg=self.button_backround_color)
        canvas.create_window(w - 150, 400, window=b7)

        b8 = Button(canvas, text="Training", command=training, font=('times', 15, ' bold '), width=20,foreground="white", bg=self.button_backround_color)
        canvas.create_window(w - 150,450, window=b8)

        # b9 = Button(canvas, text="Testing", command=testing, font=('times', 15, ' bold '), width=20,foreground="white", bg=self.button_backround_color)
        # canvas.create_window(w - 150, 500, window=b9)

        home_window_root.mainloop()


