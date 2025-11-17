"""
Test file with intentional error for testing auto-fix
"""

def divide_numbers(a, b):
    return a / b  # Bug: No zero check


def main():
    result = divide_numbers(10, 0)  # This will cause ZeroDivisionError
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
