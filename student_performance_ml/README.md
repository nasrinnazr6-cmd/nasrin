# Student Pass/Fail Prediction ML

This is a simple beginner Python machine learning project made for easy explanation in a presentation.

The project predicts whether a student will **Pass** or **Fail** using:

- Study hours
- Attendance percentage
- Assignment score
- Previous marks

## Files

- `src/app.py` - app window with input boxes and predict button
- `src/main.py` - terminal version with the ML logic
- `web/index.html` - browser webpage version
- `data/students.csv` - sample student data
- `PROJECT_EXPLANATION.md` - simple explanation for presentation
- `.vscode/launch.json` - VS Code run button setup

## How To Run In VS Code

1. Open VS Code.
2. Go to `File > Open Folder`.
3. Open this folder:
   `student_performance_ml`
4. Open `src/app.py`.
5. Click the Run button.
6. Enter values in the app window and click **Predict**.

## How To Open Webpage

Open this file in your browser:

```text
web/index.html
```

The webpage has input boxes and a Predict button.

## What The Project Does

The code uses the **K-Nearest Neighbors** algorithm.

Simple meaning:

- The program stores old student records.
- It asks for new student details.
- It compares the new student with old students.
- It finds the 3 most similar students.
- If most similar students passed, the prediction is **Pass**.
- Otherwise, the prediction is **Fail**.

## Sample Input

```text
Study Hours: 6
Attendance: 80
Assignment Score: 75
Previous Marks: 72
```

## Sample Output

```text
Predicted Result: Pass
```
