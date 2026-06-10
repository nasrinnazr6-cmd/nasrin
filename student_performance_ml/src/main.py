import csv
import math


DATA_FILE = "data/students.csv"
K_VALUE = 3


def load_students():
    students = []

    with open(DATA_FILE, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            students.append({
                "name": row["name"],
                "study_hours": int(row["study_hours"]),
                "attendance": int(row["attendance"]),
                "assignment_score": int(row["assignment_score"]),
                "previous_marks": int(row["previous_marks"]),
                "result": row["result"]
            })

    return students


def find_distance(student, study_hours, attendance, assignment_score, previous_marks):
    hours_difference = student["study_hours"] - study_hours
    attendance_difference = student["attendance"] - attendance
    assignment_difference = student["assignment_score"] - assignment_score
    marks_difference = student["previous_marks"] - previous_marks

    distance = math.sqrt(
        hours_difference ** 2
        + attendance_difference ** 2
        + assignment_difference ** 2
        + marks_difference ** 2
    )

    return distance


def predict(students, study_hours, attendance, assignment_score, previous_marks):
    distances = []

    for student in students:
        distance = find_distance(
            student,
            study_hours,
            attendance,
            assignment_score,
            previous_marks
        )

        distances.append({
            "name": student["name"],
            "distance": distance,
            "result": student["result"]
        })

    distances.sort(key=lambda item: item["distance"])
    nearest_students = distances[:K_VALUE]

    pass_count = 0
    fail_count = 0

    for student in nearest_students:
        if student["result"] == "Pass":
            pass_count = pass_count + 1
        else:
            fail_count = fail_count + 1

    if pass_count > fail_count:
        return "Pass"

    return "Fail"


def check_accuracy(students):
    correct = 0

    for student in students:
        prediction = predict(
            students,
            student["study_hours"],
            student["attendance"],
            student["assignment_score"],
            student["previous_marks"]
        )

        if prediction == student["result"]:
            correct = correct + 1

    accuracy = correct / len(students) * 100
    return accuracy


def show_dataset(students):
    print("Student Dataset")
    print("---------------")

    for student in students:
        print(
            student["name"],
            "| Study Hours:",
            student["study_hours"],
            "| Attendance:",
            student["attendance"],
            "| Assignment:",
            student["assignment_score"],
            "| Previous Marks:",
            student["previous_marks"],
            "| Result:",
            student["result"]
        )


def main():
    students = load_students()

    show_dataset(students)

    print()
    print("Model Training Completed")
    print("------------------------")
    print("Algorithm: K-Nearest Neighbors")
    print("K value:", K_VALUE)
    print("Total training records:", len(students))
    print("Model accuracy:", round(check_accuracy(students), 2), "%")

    print()
    print("Prediction")
    print("----------")

    study_hours = int(input("Enter study hours: "))
    attendance = int(input("Enter attendance percentage: "))
    assignment_score = int(input("Enter assignment score: "))
    previous_marks = int(input("Enter previous marks: "))

    result = predict(
        students,
        study_hours,
        attendance,
        assignment_score,
        previous_marks
    )

    print()
    print("Predicted Result:", result)


if __name__ == "__main__":
    main()
