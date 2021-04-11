import csv
import os
import random

#数据集movielens  使用的是Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
data_set_name="ml-latest-small"
data_file="ratings.csv"
data_set_path=os.path.join(os.getcwd(),data_set_name,data_file)

trian_data_path=os.path.join(os.getcwd(),data_set_name,"traindata.csv")
test_data_path=os.path.join(os.getcwd(),data_set_name,"testdata.csv")
train_set_percentage=0.8

def load_data(data_path):
    data = []
    with open(data_path) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)

        for row in f_csv:
            userId, movieId, rating, timestamp = row
            userId=int(userId)-1
            movieId=int(movieId)-1
            rating=int(float(rating))
            data.append([userId,movieId,rating])
    return data

def splite_data():
    trian_data=[]
    test_data=[]
    trian_data_f_csv=open(trian_data_path,"w",newline="",encoding='utf-8')
    trian_data_csv_writer=csv.writer(trian_data_f_csv)
    test_data_f_csv=open(test_data_path,"w",newline="",encoding='utf-8')
    test_data_csv_writer=csv.writer(test_data_f_csv)
    with open(data_set_path) as f:
        f_csv = csv.reader(f)
        headers=next(f_csv)
        trian_data_csv_writer.writerow(headers)
        test_data_csv_writer.writerow(headers)
        for row in f_csv:
            userId, movieId, rating, timestamp=row
            rand = random.uniform(0, 1)
            if rand<=train_set_percentage:
                trian_data.append([userId, movieId, rating])
                trian_data_csv_writer.writerow(row)
            else:
                test_data.append([userId, movieId, rating])
                test_data_csv_writer.writerow(row)

    return trian_data,test_data



load_data(trian_data_path)
