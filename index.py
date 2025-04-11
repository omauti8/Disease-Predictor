import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ------------------ Dataset ------------------
data = [
    (["fever", "cough", "sore throat"], "Flu"),
    (["chills", "body ache", "fatigue"], "Flu"),
    (["headache", "nausea", "dizziness"], "Migraine"),
    (["sensitivity to light", "blurred vision", "vomiting"], "Migraine"),
    (["chest pain", "shortness of breath", "sweating"], "Heart Attack"),
    (["left arm pain", "nausea", "anxiety"], "Heart Attack"),
    (["abdominal pain", "diarrhea", "vomiting"], "Food Poisoning"),
    (["nausea", "cramping", "fatigue"], "Food Poisoning"),
    (["fatigue", "weight loss", "frequent urination"], "Diabetes"),
    (["blurred vision", "increased thirst", "slow healing"], "Diabetes"),
    (["itching", "rash", "swelling"], "Allergy"),
    (["sneezing", "runny nose", "red eyes"], "Allergy"),
    (["joint pain", "stiffness", "swelling"], "Arthritis"),
    (["limited motion", "fatigue", "warm joints"], "Arthritis"),
    (["back pain", "numbness", "tingling"], "Spinal Disc Problem"),
    (["leg pain", "stiffness", "weakness"], "Spinal Disc Problem"),
    (["sore throat", "cough", "fever"], "Flu"),
    (["rash", "itchy eyes", "congestion"], "Allergy"),
    (["vomiting", "fever", "abdominal cramps"], "Food Poisoning"),
    (["increased hunger", "frequent urination", "dry mouth"], "Diabetes"),
]

# ------------------ Model Training ------------------
df = pd.DataFrame(data, columns=["Symptoms", "Disease"])
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])
le = LabelEncoder()
y = le.fit_transform(df["Disease"])
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# ------------------ GUI Setup ------------------
root = tk.Tk()
root.title("Disease Predictor")
root.geometry("500x550")
root.configure(bg="#f0f8ff")

tk.Label(root, text="🩺 Disease Predictor", font=("Helvetica", 16, "bold"), bg="#f0f8ff").pack(pady=20)
tk.Label(root, text="Select your symptoms (Ctrl/Cmd + click):", font=("Helvetica", 12), bg="#f0f8ff").pack()

# Multi-select Listbox for symptoms
symptom_list = sorted(mlb.classes_)
listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=40, height=15, exportselection=False)
for symptom in symptom_list:
    listbox.insert(tk.END, symptom.title())
listbox.pack(pady=10)

# Prediction Result Label
result_label = tk.Label(root, text="", font=("Helvetica", 13, "bold"), fg="darkgreen", bg="#f0f8ff")
result_label.pack(pady=10)

# Predict function
def predict_disease():
    result_label.config(text="")  # Clear previous
    selected_indices = listbox.curselection()
    if not selected_indices:
        messagebox.showwarning("No selection", "Please select at least one symptom.")
        return
    selected_symptoms = [symptom_list[i] for i in selected_indices]
    try:
        input_vec = mlb.transform([selected_symptoms])
        pred = model.predict(input_vec)
        result = le.inverse_transform(pred)[0]
        result_label.config(text=f"🧾 Predicted Disease: {result}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

# Clear function
def clear_selection():
    listbox.selection_clear(0, tk.END)
    result_label.config(text="")  # Clear prediction result

# Buttons
btn_frame = tk.Frame(root, bg="#f0f8ff")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Predict Disease", command=predict_disease, bg="#4682B4", fg="white", font=("Helvetica", 11), width=16).pack(side="left", padx=10)
tk.Button(btn_frame, text="Clear Selection", command=clear_selection, bg="#4682B4", fg="white", font=("Helvetica", 11), width=16).pack(side="right", padx=10)

# Footer
tk.Label(root, text="Built with ❤️ using Python & ML", bg="#f0f8ff", fg="gray").pack(side="bottom", pady=10)

root.mainloop()
