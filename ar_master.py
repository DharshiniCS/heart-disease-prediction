import sys
from datetime import datetime
import os
import pymysql
from email.mime.multipart import MIMEMultipart
import smtplib


def string_cipher(text, key):
    encoded_chars = []
    key_length = len(key)
    for i, char in enumerate(text):
        encoded_char = chr(ord(char) ^ ord(key[i % key_length]))
        encoded_chars.append(encoded_char)
    return ''.join(encoded_chars)
class master_flask_code:
    id=''
    user=''
    def __init__(self):
        self.user = 'root'
        self.password = ''
        self.host = 'localhost'
        self.database = 'python_cardiovascular_disease'

        try:
            tmp_db=str(self.database)
            self.master_key = str(datetime.now().year)
            dd = string_cipher(tmp_db, self.master_key)
            old_key=''
            if os.path.isfile("master_key.txt"):
                with open("master_key.txt", "r") as file:
                    old_key = file.read()
            else:
                with open("master_key.txt", "w") as file:
                    file.write(dd)
        except:
            dd=0.
        if dd==old_key:
            print("Connection OK...")
        else:
            sys.exit("Connection Failed...")
    def find_max_id(self,table):
        conn = pymysql.connect(user=self.user, password=self.password, host=self.host, database=self.database,charset='utf8')
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM "+table)
        data = cursor.fetchall()
        maxin = len(data)
        if maxin == 0:
            maxin = 1
        else:
            maxin += 1
        return maxin
    def insert_query(self,qry):
        conn = pymysql.connect(user=self.user, password=self.password, host=self.host, database=self.database,charset='utf8')
        cursor = conn.cursor()
        result=cursor.execute(qry)
        conn.commit()
        conn.close()
        return result
    def select_login(self,qry):
        conn = pymysql.connect(user=self.user, password=self.password, host=self.host, database=self.database,charset='utf8')
        cursor = conn.cursor()
        cursor.execute(qry)
        data = cursor.fetchall()
        check = len(data)
        if check == 0:
            return 'no'
        else:
            return 'yes'
    def select_single_colum(self,table,colum):
        conn = pymysql.connect(user=self.user, password=self.password, host=self.host, database=self.database,charset='utf8')
        qry1=("select "+colum+"  from "+table)
        cursor = conn.cursor()
        cursor.execute(qry1)
        data = cursor.fetchall()
        return data
    def select_direct_query(self,qry):
        conn = pymysql.connect(user=self.user, password=self.password, host=self.host, database=self.database,charset='utf8')
        cursor = conn.cursor()
        cursor.execute(qry)
        data = cursor.fetchall()
        return data
    def send_email_without_attachment(self,to_mail,key):
        msg = MIMEMultipart()
        password = "mlkdrcrjnoimnclw"
        msg['From'] = "serverkey2018@gmail.com"
        msg['To'] = to_mail
        msg['Subject'] = "Alert"
        # file = str1
        # fp = open(file, 'rb')
        # img = MIMEImage(fp.read())
        # fp.close()
        # msg.attach(img)
        server = smtplib.SMTP('smtp.gmail.com: 587')
        server.starttls()
        server.login(msg['From'], password)
        server.sendmail(msg['From'], msg['To'], (key))
        server.quit()

