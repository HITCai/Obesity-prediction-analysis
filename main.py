import pandas as pd
from sklearn.model_selection import train_test_split

# 数据预处理
def data_preprocessing(data):
    # 缺失值处理：使用众数填充缺失值
    data = data.fillna(data.mode().iloc(0))
    # 重复值处理：删除重复值
    data.drop_duplicates(inplace=True)
    # 特征工程
    # 将属性值均转化成数值类型，方便构建特征向量
    # 将Gender的值转化成0,1
    data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'Female' else 1)
    # 将family_history_with_overweight的值转化成0,1
    data['family_history_with_overweight'] = data['family_history_with_overweight'].apply(
        lambda x: 0 if x == 'no' else 1)
    # 将FAVC的值转化成0,1
    data['FAVC'] = data['FAVC'].apply(lambda x: 0 if x == 'no' else 1)
    # 将CAEC的值转化成1~4
    data.CAEC.replace(to_replace={'no': 1,
                                  'Sometimes': 2,
                                  'Frequently': 3,
                                  'Always': 4}, inplace=True)
    # 将SMOKE的值转化成0,1
    data['SMOKE'] = data['SMOKE'].apply(lambda x: 0 if x == 'no' else 1)
    # 将SCC的值转化成0,1
    data['SCC'] = data['SCC'].apply(lambda x: 0 if x == 'no' else 1)
    # 将CALC的值转化成1~4
    data.CALC.replace(to_replace={'no': 1,
                                  'Sometimes': 2,
                                  'Frequently': 3,
                                  'Always': 4}, inplace=True)
    # 将MTRANS的值转化成1~5
    data.MTRANS.replace(to_replace={'Bike': 1,
                                    'Motorbike': 2,
                                    'Walking': 3,
                                    'Automobile': 4,
                                    'Public_Transportation': 5}, inplace=True)
    # 将肥胖类型转化成不同等级0~6
    data.NObeyesdad.replace(to_replace={'Insufficient': 0, 'Normal_Weight': 1,
                                        'Overweight_Level_I': 2, 'Overweight_Level_II': 3,
                                        'Obesity_Type_I': 4, 'Obesity_Type_II': 5,
                                        'Obesity_Type_III': 6}, inplace=True)

    return data

# 计算基尼指数
def gini(data):
    # 获取标签列
    data_label = data.iloc[:,-1]
    # 统计标签列每种类型的数量
    label_num = data_label.value_counts()

    res = 0

    for x in label_num.keys():
        p_k = label_num[x]/len(data_label)
        res = p_k ** 2

    return 1-res

# 计算每个特征取值的基尼指数
def gini_index(data , a):
    # 统计特征a的频数
    feature_class = data[a].value_counts()
    # 初始化一个空列表来存储每个特征取值及其 基尼指数*比重
    res = []
    # 遍历特征a的每个取值
    for feature in feature_class.keys():
        # 计算当前特征取值的权重
        weight = feature_class[feature]/len(data)
        # 计算基尼指数
        gini_value = gini(data.loc[data[a] == feature])
        gini_value1 = gini(data.loc[data[a] != feature])
        # 添加到列表
        res.append([feature, weight * gini_value + (1 - weight) * gini_value1])

    # 按照基尼指数的大小，从小到大排序
    res = sorted(res,key=lambda x:x[-1])
    # 返回基尼指数最小的特征取值及其基尼指数
    return res[0]


# 获取标签最多的那一类
# 在构建决策树的过程中，当子集中的样本属于同一类别时，选择该类别作为叶子节点的分类标签。
# 确保在决策树的叶子节点上，选择的标签是最多的那一类，使得决策树对该子集的预测更为准确
def get_most_label(data):
    data_label = data.iloc[:,-1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

# 挑选最优特征，即基尼指数最小的特征
def get_best_feature(data):
    # 获取数据集中所有特征的列表，除了最后一列（标签列）
    features = data.columns[:-1]
    # 初始化一个空字典，存储每个特征的最优切分点
    res = {}

    # 遍历所有特征
    for a in features:
        # 对每个特征调用gini_index函数，获取最优切分点
        temp = gini_index(data, a) #temp是列表，[feature_value, gini]
        res[a] = temp

    # 对字典按照基尼指数从小到大排序
    res = sorted(res.items(),key=lambda x:x[1][1])
    # 返回具有最小基尼指数的特征及其最有切分点
    # return res[0][0], res[0][1][0]
    return res[0][0],res[0][1][0]



# def drop_exist_feature(data,best_feature,best_feature_value):
#     data = data[data[best_feature] != best_feature_value]
#     return data


# # 创建决策树
# def create_tree(data):
#     # 获取最后一列
#     data_label = data.iloc[:,-1]
#
#     #如果最后一列只有一个种类
#     if len(data_label.value_counts()) == 1:
#         return data_label.values[0]
#     # 如果每列都只有一种值
#     if all(len(data[i].value_counts()) == 1 for i in data.iloc[:,:-1].columns):
#         return get_most_label(data)
#
#     # 根据基尼指数得到最优划分
#     best_feature,best_feature_value = get_best_feature(data)
#
#     # 用字典保存树
#     Tree = {best_feature:{}}
#
#     Tree[best_feature][best_feature_value] = create_tree(drop_exist_feature(data,best_feature,best_feature_value,1)[1])
#     Tree[best_feature]['Others'] = create_tree(drop_exist_feature(data,best_feature,best_feature_value,2)[1])
#
#     return Tree

class TreeNode:
    def __init__(self, feature=None,value=None, label=None, left=None, right=None):
        self.feature = feature # 特征
        self.value = value  # 特征值
        self.label = label  # 当前节点的分类标签
        self.left = left  # 左子树
        self.right = right  # 右子树

# 创建决策树
def create_tree(data):
    # 如果数据集中的样本属于同一类别，返回一个叶子节点
    if len(data['NObeyesdad'].unique()) == 1:
        return TreeNode(label=data['NObeyesdad'].iloc[0])

    # 如果所有特征都已经用完，返回一个叶子节点，标签为样本数最多的类别
    if len(data.columns) == 1:
        return TreeNode(label=get_most_label(data))

    # 挑选最优特征和切分点
    best_feature, best_feature_value = get_best_feature(data)

    # 根据最优特征和切分点划分数据集
    left_data = data[data[best_feature] == best_feature_value]
    right_data = data[data[best_feature] != best_feature_value]

    # 递归构建左右子树
    left_subtree = create_tree(left_data)
    right_subtree = create_tree(right_data)

    # 返回当前节点
    return TreeNode(feature= best_feature,value=best_feature_value, left=left_subtree, right=right_subtree)


# 预测
# def predict(tree_node, test_data):
#     sum = 0
#     TP = 0
#     FP = 0
#     for index,row in test_data.iterrows():
#         result = row['NObeyesdad']
#         line = row.drop('NObeyesdad',axis = 0)
#         while tree_node.label is None:
#             feature_value = line.get(tree_node.value, None)
#
#             if feature_value is None:
#                 # 如果样本中没有当前节点的特征值，返回默认标签（或者根据实际情况进行处理）
#                 return get_most_label(data)
#
#             # 根据样本的特征值，决定是往左子树还是右子树走
#             if feature_value == tree_node.value:
#                 tree_node = tree_node.left
#             else:
#                 tree_node = tree_node.right
#
#         # 返回叶子节点的分类标签
#         predict_result = tree_node.label
#
#         if result == predict_result:
#             sum += 1
#
#         if predict_result != 0 and predict_result != 1 and result == predict_result:
#             TP += 1
#
#         if predict_result != 0 and predict_result != 1 and result != predict_result:
#             FP += 1
#
#     return sum,TP,FP

def predict(tree_node, test_data):
    sum = 0
    TP = 0
    FP = 0

    for index, row in test_data.iterrows():
        result = row['NObeyesdad']

        current_node = tree_node  # 使用 tree_node 作为当前节点

        while current_node.label is None:
            # feature_value = line[current_node.value]
            feature_value = row[current_node.feature]

            if feature_value is None:
                # 如果样本中没有当前节点的特征值，返回默认标签（或者根据实际情况进行处理）
                return get_most_label(data)

            # 根据样本的特征值，决定是往左子树还是右子树走
            if feature_value == current_node.value:
                current_node = current_node.left
            else:
                current_node = current_node.right

        # 返回叶子节点的分类标签
        predict_result = current_node.label

        if result == predict_result:
            sum += 1

        if predict_result != 0 and predict_result != 1 and result == predict_result:
            TP += 1

        if predict_result != 0 and predict_result != 1 and result != predict_result:
            FP += 1

    return sum, TP, FP

# 评价
def evaluate(sum,TP,FP,test_data):
    lenth = len(test_data)

    # 准确度
    Accuracy = sum / lenth
    # 精确度
    Precision = TP / (TP + FP)

    return Accuracy,Precision











if __name__ == '__main__':

    # 导入数据集
    data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    # data = pd.read_csv('test2.csv')

    # 数据预处理
    data = data_preprocessing(data)

    # 划分训练集和测试集
    train_data,test_data = train_test_split(data,test_size = 0.2,random_state = 42)

    # 创建决策树
    Tree_Node = create_tree(train_data)

    # 测试
    sum,TP,FP = predict(Tree_Node,test_data)

    # 评价
    Accuracy,Precision = evaluate(sum,TP,FP,test_data)
    print('模型准确度为：',Accuracy)
    print('模型精确度为：',Precision)



