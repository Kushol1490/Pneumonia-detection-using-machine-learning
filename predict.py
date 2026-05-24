import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog
import os
import sys


MODEL_PATH = "pneumonia_model.h5"


if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at '{MODEL_PATH}'. Train your model first.")
    sys.exit(1)


model = load_model(MODEL_PATH, compile=False)


Tk().withdraw()


img_path = filedialog.askopenfilename(
    title="Select Chest X-ray Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not img_path or not os.path.exists(img_path):
    print("No image selected. Program terminated.")
    sys.exit(1)


img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)


prediction = model.predict(img_array)[0][0]


if prediction > 0.5:
    print("\nRESULT: PNEUMONIA DETECTED\n")
    print("Recommended Actions:")
    print("- Consult a qualified physician immediately.")
    print("- Follow prescribed antibiotics or antiviral treatment.")
    print("- Take adequate rest and stay hydrated.")
    print("- Monitor symptoms such as fever, chest pain, or breathing difficulty.")
    print("- Hospitalization may be required in severe cases.")
else:
    print("\nRESULT: NORMAL (NO PNEUMONIA DETECTED)\n")
    print("Preventive Instructions:")
    print("- Maintain good respiratory hygiene.")
    print("- Avoid smoking and exposure to air pollution.")
    print("- Eat a balanced diet to strengthen immunity.")
    print("- Get regular medical check-ups if symptoms persist.")
    print("- Seek medical advice if cough or fever worsens.")
