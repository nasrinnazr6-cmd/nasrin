# Project Explanation

## Project Title

Student Pass/Fail Prediction using Machine Learning

## Aim

The aim of this project is to predict whether a student will pass or fail based on study performance details.

## Inputs Used

The model uses four inputs:

- Study hours
- Attendance percentage
- Assignment score
- Previous marks

## Algorithm Used

This project uses the K-Nearest Neighbors algorithm, also called KNN.

KNN is easy to understand. It compares a new student with old student records and finds the most similar students.

In this project, K is 3. That means the program checks the 3 nearest students.

## How Prediction Works

1. The program reads student data from `students.csv`.
2. It asks the user to enter new student details.
3. It calculates the distance between the new student and every old student.
4. It selects the 3 nearest students.
5. It checks whether most of those students passed or failed.
6. It shows the final prediction.

## Why This Is Machine Learning

The program makes predictions by using previous data. It does not use only one fixed rule. It compares the new input with examples from the dataset.

## Example

If a student studies 6 hours, has 80% attendance, good assignment marks, and good previous marks, the model will compare this student with similar students in the dataset.

If most similar students passed, the output will be:

```text
Predicted Result: Pass
```

## Advantages

- Simple and easy to understand
- Good for beginner ML presentation
- Uses real dataset format
- No external package required
- Can run directly in VS Code

