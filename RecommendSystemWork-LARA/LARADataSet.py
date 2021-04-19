from torch.utils.data.dataset import Dataset
import DataSet as dataUtil
import numpy as np
import random
from sklearn.preprocessing import normalize


class LARA(Dataset):
    def __init__(self,transform=None):
        self.item_attribute=dataUtil.getMovieGenresMatrix()
        self.item_id_list=dataUtil.getMovieIDList()
        self.user_attr_positive_matrix, self.user_attr_negative_matrix, self.positive_list, self.negative_list=dataUtil.getUserMatrix(dataUtil.trian_data_path)
        self.item_attribute = normalize(self.item_attribute ,norm='l1',axis=1)
        self.user_attr_positive_matrix=normalize(self.user_attr_positive_matrix,norm='l1',axis=1)
        self.user_attr_negative_matrix=normalize(self.user_attr_negative_matrix,norm='l1',axis=1)



    def __len__(self):
        return len(self.item_id_list)

    def __getitem__(self, index):
        while(True):
            item_attr_vec = self.item_attribute[self.item_id_list[index]]
            positive_user_list = self.positive_list[self.item_id_list[index]]
            negative_user_list = self.negative_list[self.item_id_list[index]]
            if len(positive_user_list) == 0 or len(negative_user_list) == 0:
                index=random.randint(0,len(self.item_id_list)-1)
                continue
            try:
                positive_user_id = self.positive_list[self.item_id_list[index]].pop(0)

                self.positive_list[self.item_id_list[index]].append(positive_user_id)

                negative_user_id = self.negative_list[self.item_id_list[index]].pop(0)

                self.negative_list[self.item_id_list[index]].append(negative_user_id)

                user_attr_positive_vec = self.user_attr_positive_matrix[positive_user_id]
                user_attr_negative_vec = self.user_attr_negative_matrix[negative_user_id]
            except:
                print(positive_user_id,negative_user_list)
                index = random.randint(0, len(self.item_id_list))
                continue
            break
        return np.array(item_attr_vec),np.array(user_attr_positive_vec),np.array(user_attr_negative_vec)

class LARATestDataSet(Dataset):
    def __init__(self):
        self.item_attribute = dataUtil.getMovieGenresMatrix()
        self.item_id_list = dataUtil.getMovieIDList()
        self.user_attr_positive_matrix, self.user_attr_negative_matrix, self.positive_list, self.negative_list = dataUtil.getUserMatrix(
            dataUtil.test_data_path)
        self.item_attribute = normalize(self.item_attribute, norm='l1', axis=1)
        self.user_attr_positive_matrix = normalize(self.user_attr_positive_matrix, norm='l1', axis=1)
        self.user_attr_negative_matrix = normalize(self.user_attr_negative_matrix, norm='l1', axis=1)


    def __len__(self):
        return len(self.item_id_list)

    def __getitem__(self, index):
        while (True):
            item_attr_vec = self.item_attribute[self.item_id_list[index]]

            positive_user_list = self.positive_list[self.item_id_list[index]]
            negative_user_list = self.negative_list[self.item_id_list[index]]
            if len(positive_user_list) == 0 or len(negative_user_list) == 0:
                index = random.randint(0, len(self.item_id_list) - 1)
                continue
            try:
                positive_user_id = self.positive_list[self.item_id_list[index]].pop(0)
                self.positive_list[self.item_id_list[index]].append(positive_user_id)

                negative_user_id = self.negative_list[self.item_id_list[index]].pop(0)
                self.negative_list[self.item_id_list[index]].append(negative_user_id)

                user_attr_positive_vec = self.user_attr_positive_matrix[positive_user_id]
                user_attr_negative_vec = self.user_attr_negative_matrix[negative_user_id]

            except:
                print(positive_user_id, negative_user_list)
                index = random.randint(0, len(self.item_id_list))
                continue
            break
        return np.array(item_attr_vec), np.array(user_attr_positive_vec), np.array(user_attr_negative_vec)

