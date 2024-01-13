from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.functions import col

conf = SparkConf().setAppName("bigdata").setMaster("local[*]")

# 创建Spark会话
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# 加载数据
data = spark.read.csv("ObesityDataSet_raw_and_data_sinthetic.csv", header=True, inferSchema=True)

# 数据预处理
# 删除重复值
data = data.dropDuplicates()

# 特征工程
# 转换NObeyesdad肥胖等级为0-6
label_indexer = StringIndexer(inputCol="NObeyesdad", outputCol="label")
data = label_indexer.fit(data).transform(data)

# 转换其他分类变量为数值
categorical_columns = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC", "Gender", "CAEC", "MTRANS", "CALC"]
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(data) for column in categorical_columns]
pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)

# 对所有的索引列进行独热编码
encoder_columns = [column+"_index" for column in categorical_columns]
encoder = OneHotEncoder(inputCols=encoder_columns, outputCols=[column+"_encoded" for column in encoder_columns])
data = encoder.fit(data).transform(data)


# feature_columns = ["Gender_index", "Age", "Height", "Weight", "family_history_with_overweight_index",
#                    "FAVC_index", "FCVC", "NCP", "CAEC_index", "SMOKE_index", "CH2O", "SCC_index",
#                    "FAF", "TUE", "CALC_index", "MTRANS_index"]

# 特征列列表
feature_columns = ["Gender_index_encoded", "Age", "Height", "Weight", "family_history_with_overweight_index_encoded",
                   "FAVC_index_encoded", "FCVC", "NCP", "CAEC_index_encoded", "SMOKE_index_encoded", "CH2O", "SCC_index_encoded",
                   "FAF", "TUE", "CALC_index_encoded", "MTRANS_index_encoded"]

# 创建特征向量
assembler = VectorAssembler(inputCols=feature_columns + [column+"_encoded" for column in encoder_columns], outputCol="features")
data = assembler.transform(data)

# 删除原始列，仅保留索引列和特征向量列
selected_columns = ["label", "features"]
data = data.select(selected_columns)

# 将数据拆分为训练集和测试集
train_data, test_data = data.randomSplit([0.8, 0.2], seed=41)

# 训练DecisionTree模型
# featuresCol=特征列 labelCol=标签列 maxDepth=决策树最大深度 maxBins=最大拆分特征数
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=30, maxBins=2000)
# 调用fit方法以train_data为训练集训练模型
model = dt.fit(train_data)


# 在测试集上进行预测
predictions = model.transform(test_data)

# 评估模型
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# 计算 True Positive (TP)
TP = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
# 计算 False Positive (FP)
FP = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
# 计算 True Negative (TN)
TN = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
# 计算 False Negative (FN)
FN = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()



# 打印模型准确度
print("模型准确度:", accuracy)
# 计算 Precision
precision = TP / (TP + FP)
print("模型精确度:", precision)
# 计算 Recall
recall = TP / (TP + FN)
print("模型召回率:", recall)
# 计算 F1-Score
f1_score = 2 * (precision * recall) / (precision + recall)
print("模型F1分数:", f1_score)

# 混淆矩阵
conf_matrix = predictions.groupBy("label", "prediction").count()
conf_matrix.show()


# 停止Spark会话
spark.stop()
