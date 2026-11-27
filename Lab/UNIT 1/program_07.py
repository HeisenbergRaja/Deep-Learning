# Program 7: University
# Question: Calculate mean, variance, standard deviation of student marks.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Dataset: Student Marks across 4 Courses
# ----------------------------------------------------------------------
courses = ['Linear Algebra', 'Machine Learning', 'Deep Learning', 'Computer Vision']
# 10 Students x 4 Courses marks matrix
marks = np.array([
    [78, 85, 92, 80],
    [65, 72, 70, 68],
    [88, 90, 95, 91],
    [92, 88, 89, 94],
    [55, 60, 58, 62],
    [70, 75, 80, 72],
    [84, 82, 88, 85],
    [95, 96, 98, 92],
    [62, 68, 65, 70],
    [74, 79, 83, 76]
], dtype=float)

n_students, n_courses = marks.shape
all_marks = marks.flatten()
N = len(all_marks)

# ----------------------------------------------------------------------
# 2. First-Principles Implementation of Statistical Formulas
# ----------------------------------------------------------------------
# Mean: mu = sum(x_i) / N
mean_manual = np.sum(all_marks) / N

# Population Variance: sigma^2 = sum((x_i - mu)^2) / N
var_pop_manual = np.sum((all_marks - mean_manual) ** 2) / N

# Sample Variance (Bessel's correction): s^2 = sum((x_i - mean)^2) / (N - 1)
var_sample_manual = np.sum((all_marks - mean_manual) ** 2) / (N - 1)

# Population Standard Deviation: sigma = sqrt(sigma^2)
std_pop_manual = np.sqrt(var_pop_manual)

# Sample Standard Deviation: s = sqrt(s^2)
std_sample_manual = np.sqrt(var_sample_manual)

# ----------------------------------------------------------------------
# 3. Verification with NumPy Built-in Functions
# ----------------------------------------------------------------------
mean_np = np.mean(all_marks)
var_pop_np = np.var(all_marks, ddof=0)
var_sample_np = np.var(all_marks, ddof=1)
std_pop_np = np.std(all_marks, ddof=0)
std_sample_np = np.std(all_marks, ddof=1)

print("=" * 70)
print("UNIVERSITY: STATISTICAL ANALYSIS OF STUDENT MARKS")
print("=" * 70)
print(f"Cohort Size: {n_students} Students | Number of Courses: {n_courses} | Total Grades: {N}")
print("-" * 70)
print("OVERALL COHORT PERFORMANCE STATISTICS:")
print(f"Mean (Average Mark)              : {mean_manual:.4f}  [NumPy: {mean_np:.4f}]")
print(f"Population Variance (sigma^2)    : {var_pop_manual:.4f} [NumPy: {var_pop_np:.4f}]")
print(f"Sample Variance (s^2, ddof=1)    : {var_sample_manual:.4f} [NumPy: {var_sample_np:.4f}]")
print(f"Population Std Dev (sigma)       : {std_pop_manual:.4f}  [NumPy: {std_pop_np:.4f}]")
print(f"Sample Std Dev (s, ddof=1)       : {std_sample_manual:.4f}  [NumPy: {std_sample_np:.4f}]")

# ----------------------------------------------------------------------
# 4. Course-Wise Analysis
# ----------------------------------------------------------------------
print("-" * 70)
print("COURSE-WISE BREAKDOWN:")
print(f"{'Course':<18} | {'Mean':<8} | {'Variance (Pop)':<15} | {'Std Dev (Pop)':<14} | {'Range'}")
print("-" * 70)

course_means = []
course_stds = []

for j, course_name in enumerate(courses):
    col = marks[:, j]
    c_mean = np.mean(col)
    c_var = np.var(col, ddof=0)
    c_std = np.std(col, ddof=0)
    c_min, c_max = np.min(col), np.max(col)
    course_means.append(c_mean)
    course_stds.append(c_std)
    print(f"{course_name:<18} | {c_mean:>6.2f} | {c_var:>14.2f} | {c_std:>13.2f} | {c_min:.0f} - {c_max:.0f}")

# Find highest scoring & most consistent course
highest_course = courses[np.argmax(course_means)]
most_consistent_course = courses[np.argmin(course_stds)]

print("-" * 70)
print("ACADEMIC INSIGHTS:")
print(f"- Highest Performing Course : {highest_course} (Mean = {np.max(course_means):.2f})")
print(f"- Most Consistent Course    : {most_consistent_course} (Lowest Std Dev = {np.min(course_stds):.2f})")
print("=" * 70)

# ----------------------------------------------------------------------
# 5. Course Performance Visualization using Matplotlib
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
x_pos = np.arange(len(courses))
plt.bar(x_pos, course_means, yerr=course_stds, capsize=7, color='steelblue', alpha=0.85, edgecolor='black')
plt.xticks(x_pos, courses, rotation=15)
plt.ylabel("Average Marks")
plt.title("Course Mean Marks with Standard Deviation Error Bars")
plt.ylim(0, 110)
plt.axhline(mean_manual, color='red', linestyle='--', label=f'Cohort Mean ({mean_manual:.2f})')
plt.legend(loc='lower right')
plt.tight_layout()
# plt.show() # Display plot window (uncomment when running interactively)
plt.close()

# Output:
# ======================================================================
# UNIVERSITY: STATISTICAL ANALYSIS OF STUDENT MARKS
# ======================================================================
# Cohort Size: 10 Students | Number of Courses: 4 | Total Grades: 40
# ----------------------------------------------------------------------
# OVERALL COHORT PERFORMANCE STATISTICS:
# Mean (Average Mark)              : 79.1500  [NumPy: 79.1500]
# Population Variance (sigma^2)    : 140.1775 [NumPy: 140.1775]
# Sample Variance (s^2, ddof=1)    : 143.7718 [NumPy: 143.7718]
# Population Std Dev (sigma)       : 11.8397  [NumPy: 11.8397]
# Sample Std Dev (s, ddof=1)       : 11.9905  [NumPy: 11.9905]
# ----------------------------------------------------------------------
# COURSE-WISE BREAKDOWN:
# Course             | Mean     | Variance (Pop)  | Std Dev (Pop)  | Range
# ----------------------------------------------------------------------
# Linear Algebra     |  76.30 |         162.61 |         12.75 | 55 - 95
# Machine Learning   |  79.50 |         108.05 |         10.39 | 60 - 96
# Deep Learning      |  81.80 |         162.36 |         12.74 | 58 - 98
# Computer Vision    |  79.00 |         112.40 |         10.60 | 62 - 94
# ----------------------------------------------------------------------
# ACADEMIC INSIGHTS:
# - Highest Performing Course : Deep Learning (Mean = 81.80)
# - Most Consistent Course    : Machine Learning (Lowest Std Dev = 10.39)
# ======================================================================

# ======================================================================
