import csv

with open('sample.csv') as csvfile:
    readCSV=csv.reader(csvfile,delimiter=',')
    names=[]
    for row in readCSV:
        name=row[1]
        names.append(name)
    print(names,'   ')
