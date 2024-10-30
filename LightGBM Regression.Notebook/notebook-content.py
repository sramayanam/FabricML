# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

triazines = spark.read.format("libsvm").load(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/triazines.scale.svmlight"
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# print some basic info
print("records read: " + str(triazines.count()))
print("Schema: ")
triazines.printSchema()
display(triazines.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

train, test = triazines.randomSplit([0.85, 0.15], seed=1)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from synapse.ml.lightgbm import LightGBMRegressor

model = LightGBMRegressor(
    objective="quantile", alpha=0.2, learningRate=0.3, numLeaves=31, dataTransferMode="bulk"
).fit(train)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(model.getFeatureImportances())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

scoredData = model.transform(test)
display(scoredData)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from synapse.ml.train import ComputeModelStatistics

metrics = ComputeModelStatistics(
    evaluationMetric="regression", labelCol="label", scoresCol="prediction"
).transform(scoredData)
display(metrics)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.read.format("parquet").load(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/lightGBMRanker_train.parquet"
)
# print some basic info
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()
display(df.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from synapse.ml.lightgbm import LightGBMRanker

features_col = "features"
query_col = "query"
label_col = "labels"
lgbm_ranker = LightGBMRanker(
    labelCol=label_col,
    featuresCol=features_col,
    groupCol=query_col,
    predictionCol="preds",
    leafPredictionCol="leafPreds",
    featuresShapCol="importances",
    repartitionByGroupingColumn=True,
    numLeaves=32,
    numIterations=200,
    evalAt=[1, 3, 5],
    metric="ndcg",
    dataTransferMode="bulk"
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

lgbm_ranker_model = lgbm_ranker.fit(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dt = spark.read.format("parquet").load(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/lightGBMRanker_test.parquet"
)
predictions = lgbm_ranker_model.transform(dt)
predictions.limit(10).toPandas()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
