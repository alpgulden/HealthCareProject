import sys
import pandas as pd
import numpy as np
import pyspark
from datetime import datetime
from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.functions import udf, count, collect_list, col, expr, when, mean, round
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import VectorAssembler


from pyspark.ml.classification import GBTClassifier
from pyspark.ml.regression import GBTRegressor

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

sc = SparkContext()

spark = SparkSession.builder.appName('term').getOrCreate()

def eval_row(s):
    from pyspark.ml.linalg import DenseVector
    from pyspark.ml.linalg import SparseVector
    from pyspark.sql import Row
    return eval(s)

EXP_all_featured_data = sc.textFile(sys.argv[1])

EXP_all_featured_data = EXP_all_featured_data.map(lambda x : eval_row(x))

EXP_all_featured_data = EXP_all_featured_data.toDF()

EXP_train, EXP_test = EXP_all_featured_data.randomSplit([0.7,0.3])

LOS_all_featured_data = sc.textFile(sys.argv[2])

LOS_all_featured_data = LOS_all_featured_data.map(lambda x : eval_row(x))

LOS_all_featured_data = LOS_all_featured_data.toDF()

LOS_train, LOS_test = LOS_all_featured_data.randomSplit([0.7,0.3])


evaluator = MulticlassClassificationEvaluator(
    labelCol="EXPIRE_FLAG", predictionCol="prediction", metricName="accuracy")

reg_time = datetime.now()
gbt_reg = GBTRegressor(labelCol='EXPIRE_FLAG')

paramGrid = (ParamGridBuilder()
             .addGrid(gbt_reg.maxDepth, [5, 6, 7])
             .addGrid(gbt_reg.maxIter, [10, 20, 30])
             .build())
cv_reg = CrossValidator(estimator=gbt_reg, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
# Run cross validations.
cv_reg_model = cv_reg.fit(EXP_train)
gbt_reg_cv_predictions = cv_reg_model.transform(EXP_test)
los_accuracy = evaluator.evaluate(gbt_reg_cv_predictions)
reg_time = (datetime.now() - reg_time).seconds

print("Accuracy = %g" % los_accuracy)
print("Grid time on GBT regression = %g" % reg_time)

class_time = datetime.now()
gbt_class = GBTClassifier(labelCol='EXPIRE_FLAG')
evaluator = MulticlassClassificationEvaluator(
    labelCol="EXPIRE_FLAG", predictionCol="prediction", metricName="accuracy")

paramGrid = (ParamGridBuilder()
             .addGrid(gbt_class.maxDepth, [11, 12, 13])
             .addGrid(gbt_class.maxIter, [20, 25, 30])
             .build())
cv_class = CrossValidator(estimator=gbt_class, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
# Run cross validations.
cv_class_model = cv_class.fit(EXP_train)
gbt_class_cv_predictions = cv_class_model.transform(EXP_test)
exp_accuracy = evaluator.evaluate(gbt_class_cv_predictions)
class_time = (datetime.now() - class_time).seconds

print("Accuracy = %g" % exp_accuracy)
print("Grid time on GBT regression = %g" % class_time)

