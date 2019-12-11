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

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor

from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext()

spark = SparkSession.builder.appName('term').getOrCreate()

def countICD(lst):
    rtn = []
    for i in range(1,21):
        rtn.append(lst.count("{}".format(i)))
    return rtn


def cal_age(now, dob):
    age = now.year - dob.year - ((now.month, now.day) < (dob.month, dob.day))

    if age > 86:
        return 87
    else:
        return age

def genderFM(gender):
    return 0 if gender == 'M' else 1

def icd_9Group(code):
    
    code = code[0:3]
    
    if code[0] == "E":
        return 20
    if code[0] == "V":
        return 19
    
    code = int(code)
    
    if 1 <= code <= 139:
        return 1
    
    if 140 <= code <= 239:
        return 2
    
    if 240 <= code <= 279:
        return 3
    
    if 280 <= code <= 289:
        return 4
    
    if 290 <= code <= 319:
        return 5
    
    if 320 <= code <= 389:
        return 6
    
    if 390 <= code <= 459:
        return 7
    
    if 460 <= code <= 519:
        return 8
    
    if 520 <= code <= 579:
        return 9
    
    if 580 <= code <= 629:
        return 10
    
    if 630 <= code <= 679:
        return 11
    
    if 680 <= code <= 709:
        return 12
    
    if 710 <= code <= 739:
        return 13
    
    if 740 <= code <= 759:
        return 14
    
    if 780 <= code <= 789:
        return 15
    
    if 790 <= code <= 796:
        return 16
    
    if 797 <= code <= 799:
        return 17
    
    if 800 <= code <= 999:
        return 18

udf_icd_9Group = udf(icd_9Group)

# Change temp F to C
def F_to_C(f):
    if f:
        return (f - 32) * 5 / 9
    else:
        return None

udf_F_to_C = udf(F_to_C)

# Merge two columns temp
def merge_temp(t1, t2):
    
    if t1 and t2:
        return (t1 + t2) / 2
    
    else:
        return t1 if t1 else t2

udf_merge_temp = udf(merge_temp)

# Merge two columns fio2
def merge_fio2(v1, v2):
    
    if v1 and v2:
        return (v1 + v2) / 2
    
    else:
        return v1 if v1 else v2

udf_merge_fio2 = udf(merge_fio2)


# Reading data into dataframe
df_diagnoses_icd = spark.read.csv(sys.argv[1],header=True,inferSchema=True)
df_icu_stay = spark.read.csv(sys.argv[2],header=True,inferSchema=True)
df_admin = spark.read.csv(sys.argv[3],header=True,inferSchema=True)
df_patients = spark.read.csv(sys.argv[4],header=True,inferSchema=True)

df_diagnoses_icd = df_diagnoses_icd.dropna(subset=('ICD9_CODE'))

start = datetime.now()

df_admin = df_admin.select(['SUBJECT_ID', 'HADM_ID', 
                            'ADMITTIME', 'DEATHTIME', 
                            'DIAGNOSIS', 'HOSPITAL_EXPIRE_FLAG'])

df_los = df_admin.join(df_icu_stay['HADM_ID', 'LOS'], 
                       df_icu_stay.HADM_ID == df_admin.HADM_ID)\
                 .select(df_admin['*'], df_icu_stay['LOS'])

df_mix = df_los.join(df_patients['SUBJECT_ID', 'DOB', 'GENDER'],
                     df_patients.SUBJECT_ID == df_los.SUBJECT_ID)\
               .select(df_los['*'], df_patients['DOB'], df_patients['GENDER'])


df_diagnoses_group_icd = df_diagnoses_icd.withColumn("ICD9_GROUP", udf_icd_9Group('ICD9_CODE'))
group_icd = df_diagnoses_group_icd.groupby('SUBJECT_ID').agg(collect_list('ICD9_GROUP'))

df_mix = df_mix.join(group_icd, group_icd['SUBJECT_ID'] == df_mix['SUBJECT_ID'])\
               .select(df_mix['*'], group_icd['collect_list(ICD9_GROUP)'].alias('ICD9_LIST'))


mix = df_mix.rdd

mix_temp = mix.map(lambda x : (x[0], x[1], cal_age(x[2], x[7]), x[4], x[5], x[6], genderFM(x[8]), countICD(x[9])))

df_mix_temp = mix_temp.toDF(schema=['SUBJECT_ID', 'HADM_ID', 'AGE', 
                                    'DIAGNOSIS', 'GENDER', 'LOS', 
                                    'EXPIRE_FLAG', 'ICD9_LIST'])


df_mix_temp = df_mix_temp.select(['SUBJECT_ID', 'GENDER', 'LOS', 'AGE', 'EXPIRE_FLAG'] + [expr('ICD9_LIST[{}]'.format(x)) for x in range(0, 20)])

df_mix_temp = df_mix_temp.dropna()



# Feature Items Lists
glasgow_coma_scale = [723, 454, 184, 223900, 223901, 220739]
systolic_blood_pressure = [51, 442, 455, 6701, 220179, 220050]
heart_rate = [211, 220045]
body_temp = [678, 223761, 676, 223762]
pao2_fio2_ratio = [50821, 50816, 223835, 3420, 3422, 190]
urine_output = [40055, 43175, 40069, 40094, 40715, 40473, 
                40085, 40057, 40056, 40405, 40428, 40086, 
                40096, 40651, 226559, 226560, 226561, 226584, 
                226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]
serum_urea_nitrogen = [51006]
white_blood_cells_count = [51300, 51301]
serum_bicarbonate = [50882]
sodium = [950824, 50983]
potassium = [50822, 50971]
bilirubin = [50885]

chartevents_features = [723, 454, 184, 223900, 223901, 220739, 
                       51, 442, 455, 6701, 220179, 220050, 
                       211, 220045, 678, 223761, 676, 223762, 
                       223835, 3420, 3422, 190]
outputevents_features = [40055, 43175, 40069, 40094, 40715, 40473, 
                         40085, 40057, 40056, 40405, 40428, 40086, 
                         40096, 40651, 226559, 226560, 226561, 226584, 
                         226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]
labevents_features = [50821, 50816, 51006, 51300, 51301, 50882, 
                      950824, 50983, 50822, 50971, 50885]

# Read data to dataframe
df_charts = spark.read.csv(sys.argv[5],header=True,inferSchema=True)
df_labs = spark.read.csv(sys.argv[6],header=True,inferSchema=True)
df_outputs = spark.read.csv(sys.argv[7],header=True,inferSchema=True)

# Selecting data only with feature items
df_charts_features = df_charts.filter(col('ITEMID').isin(chartevents_features))

df_labs_features = df_labs.filter(col('ITEMID').isin(labevents_features))

df_outputs_features = df_outputs.filter(col('ITEMID').isin(outputevents_features))


#------------Body Temp--------------

df_FtoC = df_charts_features.filter((col('ITEMID') == 678) | (col('ITEMID') == 223761))\
        .withColumn('VALUENUM_C', udf_F_to_C('VALUENUM'))\
        .select('SUBJECT_ID', 'ITEMID', 'VALUENUM_C')
df_C = df_charts_features.filter((col('ITEMID') == 676) | (col('ITEMID') == 223762))
# .select(col('valueuom')).distinct()

df_FtoC = df_FtoC.groupBy('SUBJECT_ID').agg(mean('VALUENUM_C').alias('avg_temp_F'))
df_C = df_C.groupBy('SUBJECT_ID').agg(mean('VALUENUM').alias('avg_temp_C'))

df_body_temp = df_FtoC.join(df_C, on='SUBJECT_ID', how='outer')\
    .withColumn('avg_temp', round(udf_merge_temp(col('avg_temp_F'), col('avg_temp_C')), 2))\
    .select('SUBJECT_ID', 'avg_temp')

#------------Heart Rate--------------

df_hr = df_charts_features.filter((col('ITEMID') == 211) | (col('ITEMID') == 220045))

df_hr = df_hr.groupBy('SUBJECT_ID').agg(round(mean('VALUENUM'), 2).alias('avg_heart_rate'))


#------------BP--------------

df_bp = df_charts_features.filter(col('ITEMID').isin(systolic_blood_pressure))

df_bp = df_bp.groupBy('SUBJECT_ID').agg(round(mean('VALUENUM'), 2).alias('avg_bp'))


#------------Coma Scale--------------

df_coma = df_charts_features.filter(col('ITEMID').isin(glasgow_coma_scale))

df_coma = df_coma.groupBy('SUBJECT_ID').agg(round(mean('VALUENUM'), 2).alias('avg_coma_scale'))


#------------Urine--------------

df_urine = df_outputs_features.filter(col('ITEMID').isin(urine_output))

df_urine = df_urine.groupBy('SUBJECT_ID').agg(round(mean('VALUE'), 2).alias('avg_urine_output'))


#------------Nitrogen--------------

df_N = df_labs_features.filter(col('ITEMID') == 51006)

df_N = df_N.groupBy('SUBJECT_ID').agg(round(mean('VALUE'), 2).alias('avg_nitrogen'))


#------------White Blood Cells--------------

df_wbc = df_labs_features.filter(col('ITEMID').isin(white_blood_cells_count))

df_wbc = df_wbc.groupBy('SUBJECT_ID').agg(round(mean('VALUE'), 2).alias('avg_wbc'))


#------------Bicarbonate--------------

df_bico = df_labs_features.filter(col('ITEMID') == 50882)

df_bico = df_bico.groupBy('SUBJECT_ID').agg(round(mean('VALUE'), 2).alias('avg_bico'))


#------------Sodium--------------

df_sodium = df_labs_features.filter(col('ITEMID').isin(sodium))

df_sodium = df_sodium.groupBy('SUBJECT_ID').agg(round(mean('VALUE'), 2).alias('avg_sodium'))


#------------Potassium--------------

df_potassium = df_labs_features.filter(col('ITEMID').isin(potassium))

df_potassium = df_potassium.groupBy('SUBJECT_ID').agg(round(mean('VALUE'), 2).alias('avg_potassium'))


#------------Bilirubin--------------

df_bilirubin = df_labs_features.filter(col('ITEMID') == 50885)

df_bilirubin = df_bilirubin.groupBy('SUBJECT_ID').agg(round(mean('VALUE'), 2).alias('avg_bilirubin'))


#------------PaO2/FiO2--------------

df_PO2 = df_labs_features.filter(col('ITEMID') == 50821)

df_PO2 = df_PO2.groupBy('SUBJECT_ID').agg(mean('VALUE').alias('avg_po2'))

df_FiO2_190 = df_charts_features.filter(col('ITEMID') == 190)

df_FiO2_190 = df_FiO2_190.groupBy('SUBJECT_ID').agg(mean('VALUE').alias('avg_fio0'))

df_FiO2 = df_charts_features.filter(col('ITEMID').isin([223835, 3420, 3422])).withColumn('VALUE', col('VALUE') / 100)

df_FiO2 = df_FiO2.groupBy('SUBJECT_ID').agg(mean('VALUE').alias('avg_fio1'))

df_FiO2 = df_FiO2.join(df_FiO2_190, on='SUBJECT_ID', how='outer')

df_FiO2 = df_FiO2.withColumn('avg_fio2', 
                             round(udf_merge_fio2(col('avg_fio0'), 
                                                  col('avg_fio1')), 2)).select('SUBJECT_ID', 'avg_fio2')

df_PF_ratio = df_PO2.join(df_FiO2, on='SUBJECT_ID', how='inner')

df_PF_ratio = df_PF_ratio\
    .withColumn('PF_ratio',round(col('avg_po2') / col('avg_fio2'), 2))\
    .select('SUBJECT_ID', 'PF_ratio')


# Merge all features into one dataframe
df_processed_features = df_body_temp.join(df_hr, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_bp, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_coma, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_urine, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_N, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_wbc, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_bico, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_sodium, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_potassium, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_bilirubin, on='SUBJECT_ID', how='inner')

df_processed_features = df_processed_features.join(df_PF_ratio, on='SUBJECT_ID', how='inner')

# df_processed_features.show(10)

df_all_features = df_processed_features.join(df_mix_temp, on='SUBJECT_ID', how='inner')

df_all_features = df_all_features.dropna()

# df_all_features.show(10)

assembler = VectorAssembler(inputCols=['avg_temp', 'avg_heart_rate', 'avg_bp', 'avg_coma_scale', 
                                        'avg_urine_output', 'avg_nitrogen', 'avg_wbc', 'avg_bico', 
                                        'avg_sodium', 'avg_bilirubin', 'PF_ratio', 'GENDER', 'AGE',
                                        'ICD9_LIST[0]', 'ICD9_LIST[1]', 'ICD9_LIST[2]', 'ICD9_LIST[3]', 
                                        'ICD9_LIST[4]', 'ICD9_LIST[5]', 'ICD9_LIST[6]', 'ICD9_LIST[7]', 
                                        'ICD9_LIST[8]', 'ICD9_LIST[9]', 'ICD9_LIST[10]', 'ICD9_LIST[11]', 
                                        'ICD9_LIST[12]', 'ICD9_LIST[13]', 'ICD9_LIST[14]', 'ICD9_LIST[15]', 
                                        'ICD9_LIST[16]', 'ICD9_LIST[17]', 'ICD9_LIST[18]', 'ICD9_LIST[19]'], outputCol='features')

featured_data = assembler.transform(df_all_features).select('features', 'LOS')

# featured_data.show(10)

print("Finished assembler")

featured_data = featured_data.dropna()

print("Finished dropna")

train,test = featured_data.randomSplit([0.7,0.3])

print("Finished randomSplit")

processing = datetime.now()
processing_time = (processing - start).seconds
print("Processing time = {}".format(processing_time))

featured_data.rdd.saveAsTextFile(sys.argv[8])



classifiers = [LinearRegression(labelCol='LOS'), 
               GeneralizedLinearRegression(labelCol='LOS'), 
               GBTRegressor(labelCol='LOS'), 
               DecisionTreeRegressor(labelCol='LOS'),
               RandomForestRegressor(labelCol='LOS')]

for classifier in classifiers:
    model_start = datetime.now()
    model = classifier.fit(train)
    predictions = model.transform(test)

    evaluator = RegressionEvaluator(labelCol="LOS", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    evaluator = RegressionEvaluator(labelCol="LOS", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)

    evaluator = RegressionEvaluator(labelCol="LOS", predictionCol="prediction", metricName="mae")
    mae = evaluator.evaluate(predictions)
    model_time = datetime.now() - model_start

    print("Model is: ", model)
    print("Root Mean Squared Error = %g" % rmse)
    print("R2 = %g" % r2)
    print("mae = %g" % mae)
    print("Model time used %g" % model_time.seconds)


end = datetime.now()
train_test_time = (end - processing).seconds

print("Processing time = {}".format(processing_time))
print("Training & testing time = {}".format(train_test_time))

