from tkinter import *
from PIL import ImageTk, Image

width = 400
height = 510

window = Tk()
window.resizable(False,False)
# window.geometry(str(width)+"x"+str(height))
window.title("ChatInTheNameOfTrueLove") # Reference to the tagline of the Muv-Luv series
# frame = Frame(window,width=300,height=300)
# frame.pack()
# frame.place(anchor='center',relx=0.5,rely=0.5)
# img = ImageTk.PhotoImage(Image.open("test.png"))
img = Image.open("test.png")
img = img.resize((img.size[0],img.size[1]), resample=Image.ANTIALIAS) # copy the image so that we don't make the original way too small
# img_resized = img.resize((300,300))
img.thumbnail((400,400))
canvas = Canvas(window,width=400,height=450)
canvas.pack()

new_img = ImageTk.PhotoImage(img)

canvas.create_image(width/2,height/2,anchor=CENTER,image=new_img)


# image_label = Label(frame,image = img)
# image_label.pack(fill = "x")
response = Label(window,text="",font=('Arial',17,'bold'),fg='#84F692')
response.pack(fill = "x")
chat_box = Entry(window,font=('Arial',17,'bold'),fg='#D5D5E8')
chat_box.pack(fill = "x")

def key_pressed(e):
    if (e.keysym == "Return"):
        response.configure(text = "This is new text" + chat_box.get())

def entered_text():
    res = "REPLACE WITH MODEL CODE" + chat_box.get()
    response.configure(text = res) 

window.bind("<KeyRelease>",key_pressed)
# window.rowconfigure(11)
# window.columnconfigure(11)
# cols, rows = window.grid_size()

# for col in range(cols):
#     window.grid_columnconfigure(col,minsize=20)

# for row in range(rows):
#     window.grid_rowconfigure(row,minsize=20)

window.mainloop() 
