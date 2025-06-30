
from tkinter import *
from PIL import Image, ImageTk
from csv_data_classification import csv_data_classification
from ecg_image_select_option import ecg_image_select_option
from echo_image_classify_select import echo_image_select_option

class cardiovascular_disease():
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
            tt = csv_data_classification()
            tt.home_window()
        image = Image.open('images/csv.png')
        img = image.resize((125, 125))
        my_img = ImageTk.PhotoImage(img)
        image_id = canvas.create_image(200, 200, image=my_img)
        canvas.tag_bind(image_id, "<1>", clickHandler)
        admin_id = canvas.create_text(200, 300, text="CSV", font=("Times New Roman", 24), fill=self.text_color)
        canvas.tag_bind(admin_id, "<1>", clickHandler)

        def clickHandler1(event):
            root.destroy()
            ar1 = echo_image_select_option()
            ar1.set_window_design()
        image1 = Image.open('images/echo.png')
        img1 = image1.resize((125, 125))
        my_img1 = ImageTk.PhotoImage(img1)
        image_id1 = canvas.create_image(400, 200, image=my_img1)
        canvas.tag_bind(image_id1, "<1>", clickHandler1)

        admin_id1 = canvas.create_text(400, 300, text="ECHO", font=("Times New Roman", 24), fill=self.text_color)
        canvas.tag_bind(admin_id1, "<1>", clickHandler1)

        def clickHandler2(event):
            root.destroy()
            sh1 = ecg_image_select_option()
            sh1.set_window_design()


        image2 = Image.open('images/ecc.png')
        img2 = image2.resize((125, 125))
        my_img2 = ImageTk.PhotoImage(img2)
        image_id2 = canvas.create_image(600, 200, image=my_img2)
        canvas.tag_bind(image_id2, "<1>", clickHandler2)

        admin_id1 = canvas.create_text(600, 300, text="ECG", font=("Times New Roman", 24), fill=self.text_color)
        canvas.tag_bind(admin_id1, "<1>", clickHandler2)



        root.mainloop()



ar = cardiovascular_disease()
ar.set_window_design()