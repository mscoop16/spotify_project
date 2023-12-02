"""

A helper file for transforming the emotions dataset to a csv

"""

import csv

def txt_to_csv(input_file, output_file, delimiter=';'):
    """Translate text to csv format"""
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        # Specify the column names for the CSV file
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for line in infile:
            text, label = line.strip().split(delimiter)
            writer.writerow({'text': text, 'label': label})

if __name__ == "__main__":
    input_file = 'emotions/test.txt'
    output_file = 'emotions/test.csv'

    txt_to_csv(input_file, output_file)
