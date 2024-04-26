
import csv
import requests

with open('places.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # skip the header
        if row[3] == '이미지URL':
            continue
        print(row)
        url = row[3]
        response = requests.get(url)

        name = row[0]
        # replace the space with underscore
        name = name.replace(' ', '_')
        with open(f'images/{name}.jpg', 'wb') as file:
            file.write(response.content)