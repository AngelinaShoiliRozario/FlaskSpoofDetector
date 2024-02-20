# Open the text file in read mode
with open('matchdata.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Print each line
        print(line.strip())  # .strip() removes leading and trailing whitespace (like newline characters)
