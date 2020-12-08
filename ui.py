from tkinter import *
import os

from cffi.setuptools_ext import execfile


def helloCallBack():
    execfile('main.py')
window=Tk()
btn=Button(window, text="Create Gesture", fg='blue', command=helloCallBack)
btn.place(x=120, y=100, width=100)
btn2=Button(window, text="Scan Gesture", fg='blue')
btn2.place(x=120, y=150, width=100)
btn3=Button(window, text="View Chart", fg='blue')
btn3.place(x=120, y=200, width=100)

lbl=Label(window, text="Sign-o-Tech", fg='red', font=("Helvetica", 16))
lbl.place(x=120, y=50)

window.title('Sign-o-Tech')
#window.geometry("300x200+10+10")
window.geometry('350x350')
window.mainloop()