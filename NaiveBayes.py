import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import os

# Đường dẫn đến thư mục bạn muốn tạo
folder_path = 'Output_Cay_Quyet_Dinh'

# Kiểm tra xem thư mục đã tồn tại hay chưa
if not os.path.exists(folder_path):
    # Nếu thư mục không tồn tại, tạo thư mục mới
    os.makedirs(folder_path)

class NaiveBayes:
    def __init__(self, table, values):
        self.table = table
        self.values = values
        self.analyzed = []
        self.valid_values = []
    
    def get_sup(self, column, x, c, mode):
        x_sup = len(self.table.loc[(self.table['class ckd'] == mode) & (self.table[column] == x)])
        x_count = len(self.table.loc[self.table[column] == x])
        return (x_sup + 1)/(c + self.table[column].nunique())

    def analyze(self):
        count_True = len(self.table.loc[self.table['class ckd'] == True])
        count_False = len(self.table.loc[self.table['class ckd'] == False])
        for index, prop in enumerate(self.values):
            temp_column = []
            temp_valid = []
            if self.table.columns[index] == 'class ckd': 
                temp_valid = [None]
                temp_column = [None]
            else:
                for a, i in enumerate(prop):
                    if len(self.table.loc[self.table.iloc[:, index] == i]) == 0:
                        continue
                    else:
                        temp_valid.append(a)
                        sup_true = self.get_sup(self.table.columns[index], i, count_True, True)
                        sup_false = self.get_sup(self.table.columns[index], i, count_False, False)
                    temp_column.append([i, sup_true, sup_false])   
            self.valid_values.append(temp_valid)
            self.analyzed.append(temp_column)

    def scan_value(self, pos, row, side):
        if side == 'left':
            pos -= 1
        else:
            pos += 1
        if pos < 0 or pos >= len(self.values[row]):
            return -1
        else:
            next_value = self.values[row][pos]
            if(type(next_value) == str):
                if '<' in next_value or '>' in next_value:
                    return -1
            if pos in self.valid_values[row]:
                return pos
            else:
                return self.scan_value(pos, row, side)
        

    def guess_class(self, row):
        NB_true = 1
        NB_false = 1
        hv_fixed = False
        for index, value in enumerate(row):
            if index == 4: continue
            y = self.values[index].index(value)
            if(y not in self.valid_values[index]):
                hv_fixed = True
                y_left = self.scan_value(y, index, 'left')
                y_right = self.scan_value(y, index, 'right')
                if y_left == -1 and y_right == -1:
                    continue
                elif abs(y - y_left) <= abs(y - y_right) or y_right == -1:
                    y = y_left
                else:
                    y = y_right
            for sup_index in range(len(self.analyzed[index])):
                if self.values[index][y] == self.analyzed[index][sup_index][0]:
                    NB_true *= self.analyzed[index][sup_index][1]
                    NB_false *= self.analyzed[index][sup_index][2]
                    continue
        if NB_true > NB_false:
            return True, hv_fixed
        else:
            return False, hv_fixed


data_origin = pd.read_csv('ckd-dataset-v2.csv') # Dữ liệu tập huấn
full_data_scanning = pd.read_csv('ckd-dataset-v2.csv')
data_origin = data_origin.replace("Dec-20", "12 - 20")
full_data_scanning = full_data_scanning.replace("Dec-20", "12 - 20")
n = None    # Độ sâu của cây quyết định
if 'affected' in data_origin.columns:
    data_origin = data_origin.drop(columns='affected') 
    full_data_scanning = full_data_scanning.drop(columns='affected')
list_unique_columns_origin = []
with open('readList.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:  # Kiểm tra nếu dòng không trống
            # Tách chuỗi thành mảng dựa trên dấu phẩy
            parts = cleaned_line.split(',')
            if parts[0] == 'int':
                array = [int(x) for x in parts[1:]]  # Chuyển đổi thành số nguyên
            elif parts[0] == 'str':
                array = parts[1:]  # Không cần chuyển đổi, giữ nguyên chuỗi
            elif parts[0] == 'bool':
                array = [False, True]
            else:
                array = None  # Trường hợp không xác định, có thể xử lý tùy theo yêu cầu
            list_unique_columns_origin.append(array)
list_unique_columns_origin.pop(-2)

x = [i for i in range(5, 101)]
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []

# print(X_train)
# Tạo một đối tượng mô hình Naive Bayes
for index in range(1):
    for i in range(1):
        data_copy = data_origin.copy().sample(frac=100/100)
        nb_model = NaiveBayes(data_copy ,list_unique_columns_origin)
        nb_model.analyze()

        true_analysis = {True: 0, False:0}
        false_analysis = {True:0, False:0}
        lam_min_data = {True:0, False:0}
        out_data = 0

# for i in range(full_data_scanning.shape[0]):
#     NB_check = (nb_model.guess_class(full_data_scanning.iloc[i]))
#     unit_class = full_data_scanning.loc[i, 'class ckd']
#     if unit_class == NB_check:
#         break
        for i in range(full_data_scanning.shape[0]):
            row = full_data_scanning.iloc[i]
            analysised, out_tree_dot = nb_model.guess_class(row)
    # print(row['class ckd'])
            if analysised == None:
                out_data += 1
            else:
                if analysised == True:
                    if analysised == row['class ckd']:
                        true_analysis[True] += 1
                    else:
                        true_analysis[False] += 1
                else:
                    if analysised == row['class ckd']:
                        false_analysis[True] += 1
                    else:
                        false_analysis[False] += 1
                if out_tree_dot == True:
                    if analysised == row['class ckd']:
                        lam_min_data[True] += 1
                    else:
                        lam_min_data[False] += 1

        percent_trust = (true_analysis[True] + false_analysis[True])/(full_data_scanning.shape[0])*100
        percent_trust = round(percent_trust, 2)

        # for i in nb_model.analyzed:
        #     for j in i:
        #         print(j)
        #     print("-----------")
#         if index == 0:
#             y1.append(percent_trust)
#         elif index == 1:
#             y2.append(percent_trust)
#         elif index == 2:
#             y3.append(percent_trust)
#         elif index == 3:
#             y4.append(percent_trust)
#         else:
#             y5.append(percent_trust)

# plt.plot(x, y1, label='lần 1')
# plt.plot(x, y2, label='lần 2')
# plt.plot(x, y3, label='lần 3')
# plt.plot(x, y4, label='lần 4')
# plt.plot(x, y5, label='lần 5')

# plt.title('Naive Bayes')
# plt.xlabel('Số lượng tập huấn (%)')
# plt.ylabel('Độ chính xác (%)')

# plt.legend()

# plt.show()

with open('Output_Cay_Quyet_Dinh/naive_bayes.txt', 'w', encoding="utf-8") as file:
    file.write("-------------------------------------" +"\n")
    file.write("Kết quả khảo sát: (Chuẩn - Lệch)" +"\n")
    file.write("Nhãn đúng: " + str(true_analysis[True]) + " - " + str(true_analysis[False]) + '\n')
    file.write("Nhãn sai: " + str(false_analysis[True]) + " - " + str(false_analysis[False]) + '\n')
    file.write("Dữ liệu được làm mịn: " + str(lam_min_data[True]) + " - " + str(lam_min_data[False]) + '\n')
    file.write("Độ chính xác: " + str(percent_trust) + "%" + "\n")
    print("Đã in xong, kiểm tra file: naive_bayes.txt")
# def tinh_gia_tri_trung_binh_va_so_lan_xuat_hien_nhieu_nhat(lst):
#     # Tính giá trị trung bình
#     if len(lst) == 0:
#         gia_tri_trung_binh = 0
#     else:
#         gia_tri_trung_binh = sum(lst) / len(lst)
    
#     # Tìm số phần tử xuất hiện nhiều nhất
#     if len(lst) == 0:
#         phan_tu_nhieu_nhat = None
#         so_lan_xuat_hien_nhieu_nhat = 0
#     else:
#         counter = Counter(lst)
#         phan_tu_nhieu_nhat, so_lan_xuat_hien_nhieu_nhat = counter.most_common(1)[0]

#     return gia_tri_trung_binh, phan_tu_nhieu_nhat, so_lan_xuat_hien_nhieu_nhat
# my_list = y1 + y2 + y3 + y4 + y5
# gia_tri_trung_binh, phan_tu_nhieu_nhat, so_lan_xuat_hien_nhieu_nhat = tinh_gia_tri_trung_binh_va_so_lan_xuat_hien_nhieu_nhat(my_list)
# print("Giá trị trung bình của list là:", gia_tri_trung_binh)
# print("Phần tử xuất hiện nhiều nhất là:", phan_tu_nhieu_nhat)
# print("Số lần xuất hiện nhiều nhất là:", so_lan_xuat_hien_nhieu_nhat)