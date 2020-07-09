import json
import csv

# Opening JSON file and loading the data
# into the variable data
with open('News_Category_Dataset_v2.json') as json_file:
    #data = json.load(json_file)
    data = [json.loads(line) for line in open('News_Category_Dataset_v2.json', 'r')]


#news_data = data['article_details']
news_data = data[:6788]

# now we will open a file for writing
data_file = open('noncovid.csv', 'w')

# create the csv writer object
csv_writer = csv.writer(data_file)

# Counter variable used for writing
# headers to the CSV file
count = 0

for art in news_data:
    if count == 0:

        # Writing headers of CSV file
        header = art.keys()
        csv_writer.writerow(header)
        count += 1

    # Writing data of CSV file
    csv_writer.writerow(art.values())
    if count == 10000:
        break
data_file.close()
