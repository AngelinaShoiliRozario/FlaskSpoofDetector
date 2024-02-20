def clear_file(file_path):
    try:
        with open(file_path, 'w') as file:
            # Truncate the file to zero length
            file.truncate(0)
        print(f"File '{file_path}' has been cleared.")
    except Exception as e:
        print(f"Error occurred while clearing the file: {e}")

# Example usage:
file_path = 'matchdata.txt'
clear_file(file_path)
