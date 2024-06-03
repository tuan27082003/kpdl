import pandas as pd
import numpy as np
import math

# Lớp cây quyết định
class DecisionTree:
    def __init__(self, value):
        self.value = value
        self.branch = {}
        self.result = None

# Lớp dữ liệu Entropy
class Entropy_temp:
    def __init__(self, value, info):
        self.value = value
        self.info = info

# Tính entropy
def entropy(list_data_count):
    temp_sum = sum(list_data_count)
    result = 0
    for i in list_data_count:
        if i == 0:
            continue
        else:
            result -= (i/temp_sum)*math.log2(i/temp_sum)
    return result

# Quét số lượng số liệu
def scan_data(data, bool, column_scan, column_unit, list_unique_columns):
    return len(data.loc[(data.loc[:, "class ckd"] == bool) & (data.iloc[:, column_scan] == list_unique_columns[column_scan][column_unit])])

# Tìm thuộc tính nhánh xây dựng cây
def find_best_column(data, list_unique_columns):
    entropy_S = entropy([len(data.loc[(data.loc[:, "class ckd"] == True)]), len(data.loc[(data.loc[:, "class ckd"] == False)])])
    max_gain = 0
    chosen_column = 0
    list_chosen_dict = {}
    temp_list = []
    temp_entropy_list = []
    temp_dict = {}
    # check = []
    for x in range(len(list_unique_columns)):
        temp_list.clear()
        temp_entropy_list.clear()
        temp_dict.clear()
        if data.columns[x] == 'class ckd':
            continue
        for i in range(len(list_unique_columns[x])):
            temp_solieu_1 = scan_data(data, True, x, i, list_unique_columns)
            temp_solieu_2 = scan_data(data, False, x, i, list_unique_columns)
            if temp_solieu_1 + temp_solieu_2 == 0: continue
            temp_list.append(temp_solieu_1 + temp_solieu_2)
            temp_en = entropy([temp_solieu_1, temp_solieu_2])
            temp_entropy_list.append(temp_en)
            temp_dict.update({list_unique_columns[x][i]: temp_en})
        temp_sum = sum(temp_list)
        total_check_entropy = 0
        for i in range(len(temp_list)):
            total_check_entropy += (temp_list[i]/temp_sum) * temp_entropy_list[i]
        temp_gain = abs(entropy_S - total_check_entropy)
        if temp_gain > max_gain or (temp_gain == max_gain and len(temp_dict) < len(list_chosen_dict)):
            max_gain = temp_gain
            chosen_column = x
            list_chosen_dict.clear()
            list_chosen_dict.update(temp_dict)
    return Entropy_temp(chosen_column, list_chosen_dict)

# Xây dựng cây
def slove_data_to_decision_tree(data, list_unique_columns, node, temp_node, level, n):
    if level == n:
        column_position = data.columns.get_loc(node.value)
        for unit in list_unique_columns[column_position]:
            so_luong_true = len(data.loc[(data.loc[:, "class ckd"] == True) & (data.iloc[:, column_position] == unit)])
            so_luong_false = len(data.loc[(data.loc[:, "class ckd"] == False) & (data.iloc[:, column_position] == unit)])
            if so_luong_true + so_luong_false == 0: continue
            node.branch.update({unit: DecisionTree('class ckd')})
            if so_luong_true > so_luong_false:
                node.branch[unit].result = True
            else:
                node.branch[unit].result = False           
        return
    # else: print(level)
    small_list = list_unique_columns.copy()
    del small_list[temp_node.value]
    for key, value in temp_node.info.items():
        # print(key)
        if value == 0:
            node.branch.update({key: DecisionTree('class ckd')})
            node.branch[key].result = data.loc[data[node.value] == key, 'class ckd'].values[0]
        else:
            # tao mot def de quy
            x = data.loc[data[data.columns[temp_node.value]] == key]
            x = x.drop(columns=data.columns[temp_node.value])
            temp_node_children = find_best_column(x, small_list)
            node.branch.update({key: DecisionTree(x.columns[temp_node_children.value])})
            slove_data_to_decision_tree(x, small_list, node.branch[key], temp_node_children, level + 1, n)

# Đọc dữ liệu từ tệp CSV (phần duy nhất được sửa vị trí truy cập file)
data_origin = pd.read_csv('<file>') # Dữ liệu tập huấn
full_data_scanning = pd.read_csv('ckd-dataset-v2.csv')
data_origin = data_origin.replace("Dec-20", "12 - 20")
n = None    # Độ sâu của cây quyết định
if 'affected' in data_origin.columns:
    data_origin = data_origin.drop(columns='affected')    # Xóa dòng này cũng được nhưng class_ckd = affected (tức là độ chính xác là 100%)

# Danh sach phan tu khac nhau trong tung cot
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
                array = [True, False]
            else:
                array = None  # Trường hợp không xác định, có thể xử lý tùy theo yêu cầu
            list_unique_columns_origin.append(array)
list_unique_columns_origin.pop(-2)

# Xây dựng cây quyết định
temp_node_started = find_best_column(data_origin, list_unique_columns_origin)
root = DecisionTree(data_origin.columns[temp_node_started.value])
slove_data_to_decision_tree(data_origin,list_unique_columns_origin, root, temp_node_started, 1, n)

# Hàm mịn dữ liệu
def scan_value(value, row, check_dict, side):
    value_index = row.index(value)
    if side == 'left':
        value_index -= 1
    else:
        value_index += 1
    if value_index < 0 or value_index >= len(row):
        return None, -1
    else:
        next_value = row[value_index]
    if '<' in next_value or '>' in next_value:
        return None, -1
    if next_value in check_dict:
        return next_value, 1
    else:
        got_value, find_position = scan_value(next_value, row, check_dict, side)
        if find_position == -1:
            return got_value, -1
        else:
            return got_value, find_position + 1    

# Hàm quét cây
def scaning_tree(row, node, isChanged):
    row_value = row[node.value]
    if row_value not in node.branch:
        left_value, closen_left = scan_value(row_value, list_unique_columns_origin[data_origin.columns.get_loc(node.value)], node.branch, 'left')
        right_value, closen_right = scan_value(row_value, list_unique_columns_origin[data_origin.columns.get_loc(node.value)], node.branch, 'right')
        isChanged = True
        if (closen_left < closen_right and closen_left != -1) or closen_right == -1:
            row_value = left_value
        elif closen_left >= closen_right and closen_right != -1 or closen_left == -1:
            row_value = right_value
        else:
            return None, isChanged
    if node.branch[row_value].result == None:
        return scaning_tree(row, node.branch[row_value], isChanged)
    else:
        return node.branch[row_value].result, isChanged

def print_decision_tree(node, file=None, indent=''):
    if node:
        if file is None:
            if node.value == 'class ckd':
                print(indent + str(node.value) + " " + str(node.result))
            else:
                print(indent + str(node.value))
        else:
            if node.value == 'class ckd':
                file.write(indent + str(node.value) + " " + str(node.result) + '\n')
            else:
                file.write(indent + str(node.value) + '\n')
        if node.branch:
            for key, value in node.branch.items():
                if file is None:
                    print(indent + '--' + str(key))
                else:
                    file.write(indent + '--' + str(key) + '\n')
                print_decision_tree(value, file, indent + '\t')

# Mở một file để ghi
with open('decision_tree.txt', 'w', encoding="utf-8") as file:
    file.write("Độ sâu cây : " + str(n) + "\n")
    # In cây vào file
    print_decision_tree(root, file)
    print("Đã in xong, kiểm tra file: decision_tree.txt")

true_analysis = {True: 0, False:0}
false_analysis = {True:0, False:0}
lam_min_data = {True:0, False:0}
out_data = 0
data_output = {
    'Ket qua trong Decision Tree' : [],
    'Ket qua thuc' : [],
    'Co lam min' : []
}

# Quét tập dữ liệu
for i in range(full_data_scanning.shape[0]):
    row = data_origin.iloc[i]
    analysised, out_tree_dot = scaning_tree(row, root, False)
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
    data_output['Ket qua trong Decision Tree'].append(analysised)
    data_output['Ket qua thuc'].append(row['class ckd'])
    data_output['Co lam min'].append(out_tree_dot)

# In danh sach
percent_trust = (true_analysis[True] + false_analysis[True])/(data_origin.shape[0])*100
percent_trust = round(percent_trust, 2)

with open('decision_tree.txt', 'a', encoding="utf-8") as file:
    file.write("-------------------------------------" +"\n")
    file.write("Kết quả khảo sát: (Chuẩn - Lệch)" +"\n")
    file.write("Nhãn đúng: " + str(true_analysis[True]) + " - " + str(false_analysis[False]) + '\n')
    file.write("Nhãn sai: " + str(false_analysis[True]) + " - " + str(false_analysis[False]) + '\n')
    file.write("Dữ liệu được làm mịn: " + str(lam_min_data[True]) + " - " + str(lam_min_data[False]) + '\n')
    file.write("Dữ liệu ngoại lệ: " + str(out_data)+ "\n")
    file.write("Độ chính xác: " + str(percent_trust) + "%" + "\n")
df = pd.DataFrame(data_output)
df.to_csv('Decision-Tree-result.csv', index=False)

