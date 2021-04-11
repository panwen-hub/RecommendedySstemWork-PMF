import numpy as np
import copy
import DataSet
import matplotlib.pyplot as plt
class PMF():
    def __init__(self,d=30,lambda_u=1e-3,lambda_v=1e-3,learning_rate=0.01,max_epoch=10,batch_size=2000,batch_num=40):
        self.lambda_u=lambda_u
        self.lambda_v=lambda_v
        self.learning_rate=learning_rate
        self.max_epoch=max_epoch
        self.train_data=np.array(DataSet.load_data(DataSet.trian_data_path))
        self.test_data=np.array(DataSet.load_data(DataSet.test_data_path))
        self.data=np.array(DataSet.load_data(DataSet.data_set_path))
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.R=None
        self.user_num=0
        self.movie_num=0
        self.rating_mean=0
        self.createR()
        self.d=d
        self.U=np.random.randn(self.user_num,self.d)
        self.V=np.random.randn(self.movie_num,self.d)
        self.U_grad=np.zeros((self.user_num,self.d))
        self.V_grad=np.zeros((self.movie_num, self.d))

        self.train_rmse_list=[]
        self.test_rmse_list=[]

# 初始化R，和相关参数设置
    def createR(self):
        max_num=np.amax(self.data,axis=0)
        self.user_num=max_num[0]+1
        self.movie_num=max_num[1]+1
        self.movie_mean = np.mean(self.train_data[:, 2])
        self.R=np.zeros([self.user_num,self.movie_num])
        for item in range(0,len(self.train_data)):
            i=int(self.train_data[int(item)][0])-1
            j=int(self.train_data[int(item)][1])-1
            r=self.train_data[int(item)][2]
            self.R[i][j]=r

# R,U使用随机梯度下降进行对R进行矩阵分解
    def train(self):
        train_num=self.train_data.shape[0]
        test_num=self.test_data.shape[0]
        mean_rating=np.mean(self.train_data[:,2])

        for i in range(self.max_epoch):
            train_order=np.arange(train_num)
            np.random.shuffle(train_order)
            for batch in range(self.batch_num):
                # 获取要使用随机梯度下降进行训练的序列号
                train_batch_order=train_order[batch*self.batch_size:(batch+1)*self.batch_size]
                train_user_id= np.array(self.train_data[train_order[train_batch_order], 0], dtype='int32')
                train_movie_id= np.array(self.train_data[train_order[train_batch_order], 1], dtype='int32')
                pred=np.sum(np.multiply(self.U[train_user_id,:],self.V[train_movie_id,:]),axis=1)

                Err=pred-self.train_data[train_order[train_batch_order],2]+self.rating_mean
                user_batch_grad=np.multiply(Err[:,np.newaxis],self.V[train_movie_id,:])+self.lambda_u*self.U[train_user_id,:]
                movie_batch_grad = np.multiply(Err[:, np.newaxis], self.U[train_user_id, :]) + self.lambda_v * self.V[train_movie_id,:]

                # 计算梯度
                for index in range(self.batch_size):
                    self.U_grad[train_user_id[index],:]+=user_batch_grad[index,:]
                    self.V_grad[train_movie_id[index],:]+=movie_batch_grad[index,:]

                self.U=self.U-self.learning_rate*self.U_grad/self.batch_size
                self.V=self.V-self.learning_rate*self.V_grad/self.batch_size

                # 计算损失函数，计算训练和测试的RMSE
                if batch==self.batch_num-1:
                    loss=np.linalg.norm(Err)**2+0.5 * self.lambda_u* np.linalg.norm(self.U) ** 2 + 0.5 * self.lambda_v* np.linalg.norm(self.V) ** 2
                    totalloss=loss/train_num


                    trainpred=np.sum(np.multiply(self.U[self.train_data[:,0]-1, :], self.V[self.train_data[:,1]-1, :]), axis=1)
                    trainErr = trainpred - self.train_data[:,2] +  mean_rating
                    self.train_rmse_list.append(np.linalg.norm(trainErr)/np.sqrt(train_num))


                    testpred = np.sum(np.multiply(self.U[self.test_data[:,0]-1, :], self.V[self.test_data[:,1]-1, :]), axis=1)
                    testErr = testpred - self.test_data[:,2] +  np.mean(self.test_data[:,2])
                    self.test_rmse_list.append(np.linalg.norm(testErr)/np.sqrt(test_num))

                    print("Training Loss:",totalloss,"\t","Training RMSE:",self.train_rmse_list[-1],"\t","Test RMSE:",self.test_rmse_list[-1])
    # 画出函数曲线
    def plt_PMF_result(self):
        plt.plot(range(self.max_epoch), self.train_rmse_list, marker='o', label='Training Data')
        plt.plot(range(self.max_epoch), self.test_rmse_list, marker='v', label='Test Data')
        plt.title('The MovieLens Dataset Learning Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show()

    def rating(self,user_i,movie_j):
        pre_rating=np.sum(np.multiply(self.U[user_i, :], self.V[movie_j, :]))
        return pre_rating

pmf=PMF()
pmf.train()
pmf.plt_PMF_result()
print("User:100 give movie:100 rating:",pmf.rating(100,100))








