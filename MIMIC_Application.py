import sys
import pandas as pd
import numpy as np
import pyspark
import seaborn as sns
import matplotlib.pyplot as plt  
from datetime import datetime
from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.functions import udf, count, collect_list, col, expr, when, mean, round
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

sc = SparkContext()

spark = SparkSession.builder.appName('term').getOrCreate()

# Convert the RDD Row format string to the Row object
def eval_row(s):
    from pyspark.ml.linalg import DenseVector
    from pyspark.ml.linalg import SparseVector
    from pyspark.sql import Row
    return eval(s)

# Reading the classification featured data
EXP_all_featured_data = sc.textFile('All_features_Classification.txt/')

EXP_all_featured_data = EXP_all_featured_data.map(lambda x : eval_row(x))

EXP_all_featured_data = EXP_all_featured_data.toDF()

# Split to Training and Testing sets
EXP_train, EXP_test = EXP_all_featured_data.randomSplit([0.7,0.3])


# Reading the regression featured data
LOS_all_featured_data = sc.textFile('All_features_Regression.txt/')

LOS_all_featured_data = LOS_all_featured_data.map(lambda x : eval_row(x))

LOS_all_featured_data = LOS_all_featured_data.toDF()

# Split to Training and Testing sets
LOS_train, LOS_test = LOS_all_featured_data.randomSplit([0.7,0.3])



# GBT Classifier Implementation
model_start = datetime.now()
gbt = GBTClassifier(labelCol='EXPIRE_FLAG', maxDepth=13, maxIter=30)
exp_model = gbt.fit(EXP_train)
exp_predictions = exp_model.transform(EXP_test)
exp_evaluator = MulticlassClassificationEvaluator(
    labelCol="EXPIRE_FLAG", predictionCol="prediction", metricName="accuracy")

accuracy = exp_evaluator.evaluate(exp_predictions)
model_time = datetime.now() - model_start

preds_and_labels = exp_predictions.select('prediction', 'EXPIRE_FLAG')

rdd_preds_and_labels = preds_and_labels.rdd.map(lambda x : (x[0], float(x[1])))

matrics = MulticlassMetrics(rdd_preds_and_labels)

confusion_matrix = matrics.confusionMatrix().toArray()

true_positive = confusion_matrix[0][0]
false_positive = confusion_matrix[0][1]
false_negative = confusion_matrix[1][0]
true_negative = confusion_matrix[1][1]

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * (precision * recall) / (precision + recall)

print("Model is: ", exp_model)
print("Model time used %g seconds" % model_time.seconds)

ax= plt.subplot()
sns.heatmap(confusion_matrix, annot=True, ax = ax, fmt='g', cmap="YlGnBu");

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Live', 'Death']); 
ax.yaxis.set_ticklabels(['Live', 'Death']);

print("Accuracy is: %g " % accuracy)
print("Precision is: %g " % precision)
print("Recall is: %g " % recall)
print("F1 score is: %g " % f1)



# GBT Regression Implementation
model_start = datetime.now()
gbt = GBTRegressor(labelCol='LOS', maxDepth=6, maxIter=30)
los_model = gbt.fit(LOS_train)
los_predictions = los_model.transform(LOS_test)

los_evaluator = RegressionEvaluator(labelCol="LOS", predictionCol="prediction", metricName="rmse")
rmse = los_evaluator.evaluate(los_predictions)

los_evaluator = RegressionEvaluator(labelCol="LOS", predictionCol="prediction", metricName="mae")
mae = los_evaluator.evaluate(los_predictions)

model_time = datetime.now() - model_start

preds_and_labels = los_predictions.select('prediction', 'LOS')

print("Model is: ", los_model)
print("Model time used %g seconds" % model_time.seconds)
print("Root Mean Square Error is: %g " % rmse)
print("Mean Absolute Error: %g" % mae)



# Feature Explainations:

# avg_temp, avg_hr, avg_bp, avg_coma, avg_urine, 
# avg_nitrogen, avg_wbc, avg_bico, avg_sodium, 
# avg_potassium, avg_bilirubin, PF_ratio, gender(0 for M, 1 for F), los, age
# ICD9_List 0 (Infectious And Parasitic Diseases)
# ICD9_List 1 (Neoplasms)
# ICD9_List 2 (Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders)
# ICD9_List 3 (Diseases Of The Blood And Blood-Forming Organs)
# ICD9_List 4 (Mental Disorders)
# ICD9_List 5 (Diseases Of The Nervous System And Sense Organs)
# ICD9_List 6 (Diseases Of The Circulatory System)
# ICD9_List 7 (Diseases Of The Respiratory System)
# ICD9_List 8 (Diseases Of The Digestive System)
# ICD9_List 9 (Diseases Of The Genitourinary System)
# ICD9_List 10(Complications Of Pregnancy, Childbirth, And The Puerperium)
# ICD9_List 11(Diseases Of The Skin And Subcutaneous Tissue)
# ICD9_List 12(Diseases Of The Musculoskeletal System And Connective Tissue)
# ICD9_List 13(Congenital Anomalies)
# ICD9_List 14(Symptoms)
# ICD9_List 15(Nonspecific Abnormal Findings)
# ICD9_List 16(Ill-Defined And Unknown Causes Of Morbidity And Mortality)
# ICD9_List 17(Injury And Poisoning)
# ICD9_List 18(Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services)
# ICD9_List 19(Supplementary Classification Of External Causes Of Injury And Poisoning)

feature_dictionary = {'avg_temp' : 'Body Temperature', 
     'avg_heart_rate' : 'Heart Rate', 
     'avg_bp' : 'Blood Pressure', 
     'avg_coma_scale' : 'Glasgow Coma Scale', 
     'avg_urine_output' : 'Urine Output', 
     'avg_nitrogen' : 'Serum Urea Nitrogen Level', 
     'avg_wbc' : 'White Blood Cells Count', 
     'avg_bico' : 'Serum Bicarbonate Level', 
     'avg_sodium' : 'Sodium Level', 
     'avg_potassium' : 'Potassium Level', 
     'avg_bilirubin' : 'Bilirubin Level', 
     'PF_ratio' : 'PaO2 / FiO2 ratio', 
     'GENDER' : '0 for Male, 1 for Female', 'AGE' : "Patient's Age", 
     'ICD9_LIST[0]' : 'Infectious And Parasitic Diseases', 
     'ICD9_LIST[1]' : 'Neoplasms', 
     'ICD9_LIST[2]' : 'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders', 
     'ICD9_LIST[3]' : 'Diseases Of The Blood And Blood-Forming Organs', 
     'ICD9_LIST[4]' : 'Mental Disorders', 
     'ICD9_LIST[5]' : 'Diseases Of The Nervous System And Sense Organs',
     'ICD9_LIST[6]' : 'Diseases Of The Circulatory System', 
     'ICD9_LIST[7]' : 'Diseases Of The Respiratory System', 
     'ICD9_LIST[8]' : 'Diseases Of The Digestive System', 
     'ICD9_LIST[9]' : 'Diseases Of The Genitourinary System', 
     'ICD9_LIST[10]' : 'Complications Of Pregnancy, Childbirth, And The Puerperium', 
     'ICD9_LIST[11]' : 'Diseases Of The Skin And Subcutaneous Tissue', 
     'ICD9_LIST[12]' : 'Diseases Of The Musculoskeletal System And Connective Tissue', 
     'ICD9_LIST[13]' : 'Congenital Anomalies', 
     'ICD9_LIST[14]' : 'Symptoms', 
     'ICD9_LIST[15]' : 'Nonspecific Abnormal Findings', 
     'ICD9_LIST[16]' : 'Ill-Defined And Unknown Causes Of Morbidity And Mortality', 
     'ICD9_LIST[17]' : 'Injury And Poisoning', 
     'ICD9_LIST[18]' : 'Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services', 
     'ICD9_LIST[19]' : 'Supplementary Classification Of External Causes Of Injury And Poisoning'}



los_vec = ['avg_temp', 'avg_heart_rate', 'avg_bp', 'avg_coma_scale', 
    'avg_urine_output', 'avg_nitrogen', 'avg_wbc', 'avg_bico', 
    'avg_sodium', 'avg_bilirubin', 'PF_ratio', 'GENDER', 'AGE',
    'ICD9_LIST[0]', 'ICD9_LIST[1]', 'ICD9_LIST[2]', 'ICD9_LIST[3]', 
    'ICD9_LIST[4]', 'ICD9_LIST[5]', 'ICD9_LIST[6]', 'ICD9_LIST[7]', 
    'ICD9_LIST[8]', 'ICD9_LIST[9]', 'ICD9_LIST[10]', 'ICD9_LIST[11]', 
    'ICD9_LIST[12]', 'ICD9_LIST[13]', 'ICD9_LIST[14]', 'ICD9_LIST[15]', 
    'ICD9_LIST[16]', 'ICD9_LIST[17]', 'ICD9_LIST[18]', 'ICD9_LIST[19]']

exp_vec = ['avg_temp', 'avg_heart_rate', 'avg_bp', 'avg_coma_scale', 
    'avg_urine_output', 'avg_nitrogen', 'avg_wbc', 'avg_bico', 
    'avg_sodium', 'avg_bilirubin', 'PF_ratio', 'GENDER', 'LOS', 'AGE',
    'ICD9_LIST[0]', 'ICD9_LIST[1]', 'ICD9_LIST[2]', 'ICD9_LIST[3]', 
    'ICD9_LIST[4]', 'ICD9_LIST[5]', 'ICD9_LIST[6]', 'ICD9_LIST[7]', 
    'ICD9_LIST[8]', 'ICD9_LIST[9]', 'ICD9_LIST[10]', 'ICD9_LIST[11]', 
    'ICD9_LIST[12]', 'ICD9_LIST[13]', 'ICD9_LIST[14]', 'ICD9_LIST[15]', 
    'ICD9_LIST[16]', 'ICD9_LIST[17]', 'ICD9_LIST[18]', 'ICD9_LIST[19]']


los_assembler = VectorAssembler(inputCols=los_vec, outputCol='features')

exp_assembler = VectorAssembler(inputCols=exp_vec, outputCol='features')

# Promt User to imput patients information
def collectInfo(patient, vec):
    
    i = 0
    while i < len(vec):
        try:
            item = vec[i]
            temp = float(input("Patient's {} is: \n{}\n".format(item, feature_dictionary.get(item))))
            
            if temp == 999:
                i = 999
                
            patient.append(temp)
            i = i + 1
        except:
            print("Numbers Only")
    
    return np.array(patient)


def prediction_app():
    
    select = int(input("Press 1 for Length of Stay prediction \nPress 2 for Mortality prediction\n"))

    patient = []

    if select == 1:
        
        print("Predicting Patient's Length of Stay, Please Enter Patient's Information:")
        
        patient = collectInfo(patient, los_vec)

        patient_df = pd.DataFrame(patient.reshape(-1,33), columns=los_vec)

        vec_spark = spark.createDataFrame(patient_df)

        feature_vec = los_assembler.transform(vec_spark).select('features')

        pred = los_model.transform(feature_vec)
        
        return pred

    if select == 2:

        print("Predicting Patient's Mortality, Please Enter Patient's Information:")
        
        patient = collectInfo(patient, exp_vec)

        patient_df = pd.DataFrame(patient.reshape(-1,34), columns=exp_vec)

        vec_spark = spark.createDataFrame(patient_df)

        feature_vec = exp_assembler.transform(vec_spark).select('features')

        pred = exp_model.transform(feature_vec)
        
        return pred
    
    if select == 0:
        print("Exiting...")
        return

    else:
        print("Please enter only 1 or 2")
        prediction_app()

demo = prediction_app()
print(demo.toPandas()['prediction'][0])


