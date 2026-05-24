import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
import os


MODEL_PATH = "pneumonia_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train your model first.")


model = load_model(MODEL_PATH, compile=False)

def select_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select Chest X-ray Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path or not os.path.exists(file_path):
        result_label.config(text="No image selected", fg="red")
        instruction_text.set("")
        return


    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    prediction = model.predict(img_array)[0][0]


    display_img = Image.open(file_path).resize((200, 200))
    photo = ImageTk.PhotoImage(display_img)
    image_label.config(image=photo)
    image_label.image = photo


    if prediction > 0.5:
        result_label.config(text="PNEUMONIA DETECTED", fg="#d90429")
        instruction_text.set(
            "Recommended Actions:\n"
            "• Consult a qualified physician immediately\n"
            "• Follow prescribed medication\n"
            "• Take adequate rest and fluids\n"
            "• Monitor fever, cough & breathing issues\n"
            "• Hospital care may be required if severe"
        )
    else:
        result_label.config(text="NORMAL (NO PNEUMONIA)", fg="#2b9348")
        instruction_text.set(
            "Preventive Instructions:\n"
            "• Maintain good respiratory hygiene\n"
            "• Avoid smoking & polluted air\n"
            "• Eat nutritious food\n"
            "• Exercise regularly\n"
            "• Seek medical advice if symptoms persist"
        )


root = tk.Tk()
root.title("AI-Based Pneumonia Detection System")
root.geometry("600x780")
root.config(bg="#f1faee")


title = tk.Label(
    root,
    text="Pneumonia Detection Using AI (Machine Learning) & Health Suggestion",
    font=("Helvetica", 20, "bold"),
    bg="#1d3557",
    fg="white",
    pady=18
)
title.pack(fill="x")


student_info = tk.Label(
    root,
    text="Developed By\n"
         "Kushol Bhadra (ID: 223002141)\n"
         "BSc in CSE",
    font=("Arial", 18, "bold"),
    bg="#e9ecef",
    fg="#1d3557",
    pady=14
)
student_info.pack(fill="x", padx=12, pady=12)

university_label = tk.Label(
    root,
    text="Green University of Bangladesh",
    font=("Arial", 15, "bold"),
    bg="#e9ecef",
    fg="#2b9348",
    pady=8
)
university_label.pack(fill="x", padx=12, pady=(0, 12))

subtitle = tk.Label(
    root,
    text="Upload Chest X-ray Image Below",
    font=("Arial", 15, "bold"),
    bg="#f1faee",
    fg="#1d3557"
)
subtitle.pack(pady=12)

image_label = tk.Label(root, bg="#f1faee")
image_label.pack(pady=12)

btn = tk.Button(
    root,
    text="Select X-ray Image",
    command=select_and_predict,
    font=("Arial", 15, "bold"),
    bg="#457b9d",
    fg="white",
    padx=25,
    pady=12,
    relief="flat"
)
btn.pack(pady=18)

result_label = tk.Label(
    root,
    text="Result:",
    font=("Arial", 12, "bold"),
    bg="#f1faee",
    fg="#343a40"
)
result_label.pack(pady=12)

instruction_text = tk.StringVar()
instructions = tk.Label(
    root,
    textvariable=instruction_text,
    font=("Arial", 10),
    bg="#f1faee",
    justify="left",
    wraplength=520
)
instructions.pack(pady=12)

footer = tk.Label(
    root,
    text="ML Lab Project | Pneumonia Detection System | 2026",
    font=("Arial", 11, "bold"),
    bg="#1d3557",
    fg="white",
    pady=10
)
footer.pack(side="bottom", fill="x")

root.mainloop()