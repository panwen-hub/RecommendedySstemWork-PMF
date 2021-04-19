import csv
import os
import random
import numpy as np


#数据集movielens  使用的是Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
data_set_name="ml-latest-small"
data_file="ratings.csv"
data_set_path=os.path.join(os.getcwd(),data_set_name,data_file)

trian_data_path=os.path.join(os.getcwd(),data_set_name,"traindata.csv")
test_data_path=os.path.join(os.getcwd(),data_set_name,"testdata.csv")
train_set_percentage=0.8

movie_genres_file=os.path.join(os.getcwd(),data_set_name,"movies.csv")
movie_tags_file=os.path.join(os.getcwd(),data_set_name,"tags.csv")
movie_generes_list_path=os.path.join(os.getcwd(),data_set_name,"moviesgenres.txt")
movie_tags_list_path=os.path.join(os.getcwd(),data_set_name,"movietags.txt")
movie_num=0
movie_id_max=193609
user_id_max=610



def getMovieGeneresList():
    movie_genres=[]
    if os.path.exists(movie_generes_list_path):
        with open(movie_generes_list_path,encoding='UTF-8') as f:
            lines=f.readlines()
            for line in lines:
                if line !='\n':
                    line=line.rstrip('\n')
                    movie_genres.append(line)
    else:
        with open(movie_genres_file,encoding='UTF-8') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            with open(movie_generes_list_path,'x') as write_f:
                try:
                    for row in f_csv:
                        movieId, title, genres = row
                        genresAttribute = genres.split('|')
                        for item in genresAttribute:
                            if item not in movie_genres:
                                movie_genres.append(item)
                                write_f.write(item+'\n')
                except:
                    print('Error in row', row)
    return movie_genres





def getMovieTagsList():
    movie_tags_list = []

    if os.path.exists(movie_tags_list_path):
        with open(movie_tags_list_path,encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                if line != '\n':
                    line = line.rstrip('\n')
                    movie_tags_list.append(line)
    else:
        with open(movie_tags_file,encoding='UTF-8') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            with open(movie_tags_list_path, 'x') as write_f:
                try:
                    for row in f_csv:
                        userId, movieId, tag, timestamp = row
                        if tag not in movie_tags_list:
                            movie_tags_list.append(tag)
                            write_f.write(tag + '\n')

                except:
                    print('Error in Tag', row)




    return movie_tags_list



# 通过地址加载数据
def load_data(data_path):
    data = []
    with open(data_path,encoding='UTF-8') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)

        for row in f_csv:
            userId, movieId, rating, timestamp = row
            userId=int(userId)-1
            movieId=int(movieId)-1
            rating=int(float(rating))
            data.append([userId,movieId,rating])
    return data


# 根据比例划分数据集
def splite_data():
    trian_data=[]
    test_data=[]
    trian_data_f_csv=open(trian_data_path,"w",newline="",encoding='utf-8')
    trian_data_csv_writer=csv.writer(trian_data_f_csv)
    test_data_f_csv=open(test_data_path,"w",newline="",encoding='utf-8')
    test_data_csv_writer=csv.writer(test_data_f_csv)
    with open(data_set_path,encoding='UTF-8') as f:
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


def getMovieIDList():
    movie_id_list= []
    if movie_num==0:
        with open(movie_genres_file,encoding='UTF-8') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            try:
                for row in f_csv:
                    movieId, title, genres = row
                    movieId=int(movieId)
                    if movieId not in movie_id_list:
                        movie_id_list.append(movieId)

            except:
                print('Error in row', row)
    return np.array(movie_id_list)


def getMovieGenresMatrix():
    movie_genres_list=getMovieGeneresList()
    movie_genres_matrix = np.zeros((movie_id_max+1,len(movie_genres_list)))
    with open(movie_genres_file,encoding='UTF-8') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        try:
            for row in f_csv:
                movieId, title, genres = row
                genresAttribute = genres.split('|')
                for item in genresAttribute:
                   index=movie_genres_list.index(item)
                   movie_genres_matrix[int(movieId)][int(index)] = 1

        except BaseException:
            print('Error in row', row)
            print('Error')

    return movie_genres_matrix

def getUserMatrix(file_path):
    movie_genres_matrix = getMovieGenresMatrix()
    movie_genres_list=getMovieGeneresList()
    user_attr_positive_matrix=np.zeros((user_id_max+1,len(movie_genres_list)))
    user_attr_negative_matrix=np.zeros((user_id_max+1,len(movie_genres_list)))
    positive_list=[[] for i in range(movie_id_max+1)]
    negative_list=[[] for i in range(movie_id_max+1)]
    with open(file_path, encoding='UTF-8') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)

        for row in f_csv:
            userId,movieId,rating,timestamp = row
            userId=int(userId)
            movieId=int(movieId)
            rating=float(rating)
            if rating>=4:
                positive_list[movieId].append(userId)
                attr_list = movie_genres_matrix[movieId]
                for i in range(len(attr_list)):
                    if attr_list[i] == 1:
                        user_attr_positive_matrix[userId][i] = 1

            elif rating<=3:
                negative_list[movieId].append(userId)
                attr_list = movie_genres_matrix[movieId]
                for i in range(len(attr_list)):
                    if attr_list[i] == 1:
                        user_attr_negative_matrix[userId][i] = 1

    return user_attr_positive_matrix,user_attr_negative_matrix,positive_list,negative_list











