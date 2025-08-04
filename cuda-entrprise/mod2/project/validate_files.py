import os
import sys
import csv

acceptable_vals = [-1.0, 1.0]

acceptable_vals_str = "[-1.0, 1.0]"

SUCCESS_PRINT_OUT = """Validator Results: All of the required files exist, are readable, and follow the expected pattern for the autograder, 
            you should create the submission file and submit the assignment and see your grade.  
            If the grader fails to complete grading, then submit a discussion thread, stating that the files were validated but the grader could not complete its task."""

FAILURE_PRINT_OUT = """Validator Results: One or more of the assignment files has failed, as identified in the error messages above.  
            Fix them and revalidate your files before creating the submission file and submitting the assignment. """

# Based on google search result's AI results
def can_be_cast_to_float(val_str: str):
    try:
        float(val_str)
        return True
    except ValueError:
        return False

def general_validate_file(filename: str, num_values: int = 128):
    results = True
    print(f"Validating file: {filename}")

    with open(filename, 'r') as file:
        num_lines = len(file.readlines())
        if num_lines > 1:
            print(f"Error: The file {filename} has more than the 1 line that is required by the grader, please fix your code/the file to not include any newline characters.")
            results = False

    with open(filename, 'r') as file:
        # Perform file operations here
        content = file.read()
        line = content.strip()
        vals = line.split(',')
        file_num_vals = len(vals)
        if not file_num_vals == num_values:
            print(f"Error: The number values in the file were: {file_num_vals} but the expected number is: {num_values}")
            results = False
        for index, val_str in enumerate(vals):
            if not can_be_cast_to_float(val_str=val_str):
                print(f"Error: index: {index} value: {val_str} is a not float, which is required.")
                results = False
    return results

def validate_comparison_file(filename: str, num_values: int = 128):
    results = True
    with open(filename, 'r') as file:
        # Perform file operations here
        content = file.read()
        line = content.strip()
        vals = line.split(',')
        for index, val_str in enumerate(vals):
            if not can_be_cast_to_float(val_str=val_str):
                print(f"Error: index: {index} value: {val} is a not float, which is required.")
                results = False
            else:
                val = float(val_str)
                if not val in acceptable_vals:
                    print(f"Error value: {val_str} is not in acceptable values list: {acceptable_vals_str}")
                    results = False
    return results

def validate_file(filename: str, num_values: int = 128, is_output_file: str = False):
    results = True
    if not os.path.exists(filename) or not os.access(filename, os.R_OK):
        print(f"The file '{filename}' does not exist or is not readable, your assignment submission will fail if this file doesn't exist or isn't readable.")
    results = general_validate_file(filename=filename, num_values=num_values)
    if is_output_file:
        results = results and validate_comparison_file(filename=filename, num_values=num_values)
    if not results:
        print(f"File: {filename} is invalid, as the above message should make clear. Please fix the errors in your code that are causing these issues and revalidate.")
    return results

if __name__ == '__main__':
    results_input_a = validate_file(filename="input_a.csv", num_values=128, is_output_file=False)
    results_input_b = validate_file(filename="input_b.csv", num_values=128, is_output_file=False)
    results_output_a = validate_file(filename="output_a.csv", num_values=128, is_output_file=True)
    results_output_b = validate_file(filename="output_b.csv", num_values=128, is_output_file=True)
    results = results_input_a and results_input_b and results_output_a and results_output_b
    if results:
        print(SUCCESS_PRINT_OUT)
    else:
        print(FAILURE_PRINT_OUT)