import tkinter as tk
from tkinter import messagebox

from main import K_VALUE, check_accuracy, load_students, predict


students = load_students()


def make_prediction():
    try:
        study_hours = int(study_hours_entry.get())
        attendance = int(attendance_entry.get())
        assignment_score = int(assignment_entry.get())
        previous_marks = int(previous_marks_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter numbers only.")
        return

    result = predict(
        students,
        study_hours,
        attendance,
        assignment_score,
        previous_marks
    )

    result_label.config(text="Predicted Result: " + result)


window = tk.Tk()
window.title("Student Pass/Fail Prediction")
window.geometry("430x430")
window.configure(bg="#f4f7fb")

title_label = tk.Label(
    window,
    text="Student Pass/Fail Prediction",
    font=("Arial", 18, "bold"),
    bg="#f4f7fb",
    fg="#1f2937"
)
title_label.pack(pady=15)

info_text = "Algorithm: KNN | K = " + str(K_VALUE)
info_label = tk.Label(window, text=info_text, font=("Arial", 11), bg="#f4f7fb")
info_label.pack()

accuracy = round(check_accuracy(students), 2)
accuracy_label = tk.Label(
    window,
    text="Model Accuracy: " + str(accuracy) + "%",
    font=("Arial", 11),
    bg="#f4f7fb"
)
accuracy_label.pack(pady=5)

form_frame = tk.Frame(window, bg="#f4f7fb")
form_frame.pack(pady=15)

tk.Label(form_frame, text="Study Hours", bg="#f4f7fb").grid(row=0, column=0, padx=10, pady=8, sticky="w")
study_hours_entry = tk.Entry(form_frame, width=20)
study_hours_entry.grid(row=0, column=1, padx=10, pady=8)

tk.Label(form_frame, text="Attendance %", bg="#f4f7fb").grid(row=1, column=0, padx=10, pady=8, sticky="w")
attendance_entry = tk.Entry(form_frame, width=20)
attendance_entry.grid(row=1, column=1, padx=10, pady=8)

tk.Label(form_frame, text="Assignment Score", bg="#f4f7fb").grid(row=2, column=0, padx=10, pady=8, sticky="w")
assignment_entry = tk.Entry(form_frame, width=20)
assignment_entry.grid(row=2, column=1, padx=10, pady=8)

tk.Label(form_frame, text="Previous Marks", bg="#f4f7fb").grid(row=3, column=0, padx=10, pady=8, sticky="w")
previous_marks_entry = tk.Entry(form_frame, width=20)
previous_marks_entry.grid(row=3, column=1, padx=10, pady=8)

predict_button = tk.Button(
    window,
    text="Predict",
    command=make_prediction,
    font=("Arial", 12, "bold"),
    bg="#2563eb",
    fg="white",
    width=16
)
predict_button.pack(pady=10)

result_label = tk.Label(
    window,
    text="Predicted Result:",
    font=("Arial", 14, "bold"),
    bg="#f4f7fb",
    fg="#111827"
)
result_label.pack(pady=15)

window.mainloop()
