import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df = df.drop("customerID")

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    df = df.fillna({"TotalCharges": 0})
    categorical_cols = ['gender', 'PhoneService', 'InternetService']

    indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index", handleInvalid="keep") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "_Index", outputCol=col + "_Vec") for col in categorical_cols]

    for stage in indexers + encoders:
        df = stage.fit(df).transform(df)

    feature_cols = [col + "_Vec" for col in categorical_cols] + ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)
    df = StringIndexer(inputCol="Churn", outputCol="label").fit(df).transform(df)

    with open(f"{output_dir}/task1_preprocessing_summary.txt", "w") as f:
        f.write(" Preprocessing complete.\n")
        f.write(f"Features used: {feature_cols}\n")
        f.write(f"Total rows after preprocessing: {df.count()}\n")
    
    return df.select("features", "label")

# Task 2: Train and Evaluate a Logistic Regression Model
def train_logistic_regression_model(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression()
    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    with open(f"{output_dir}/task2_logistic_regression_results.txt", "w") as f:
        f.write(" Logistic Regression Model Evaluation\n")
        f.write(f"AUC (Area Under ROC): {auc:.4f}\n")

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    model = selector.fit(df)
    selected_indices = model.selectedFeatures

    # Full list of input features used in VectorAssembler (same as Task 1)
    categorical_cols = ['gender', 'PhoneService', 'InternetService']
    encoded_cols = [col + "_Vec" for col in categorical_cols]
    feature_names = encoded_cols + ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    # OneHotEncoder expands each _Vec into multiple columns — estimate expansion
    # We'll fit a sample to get accurate column mapping
    from pyspark.ml.linalg import DenseVector
    sample = df.select("features").head()[0]
    total_feature_count = len(sample)

    # Assume order is preserved: first OneHotEncoded vectors, then numeric columns
    # Since we don't know exact dimensions per _Vec, we'll use an approximation
    selected_feature_names = []
    current_index = 0
    for col in encoded_cols:
        # Assume each one-hot vector has 2-3 categories (adjust if needed)
        for i in range(3):  
            if current_index in selected_indices:
                selected_feature_names.append(f"{col}[{i}]")
            current_index += 1
    # Now check numeric columns
    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        if current_index in selected_indices:
            selected_feature_names.append(col)
        current_index += 1

    with open(f"{output_dir}/task3_feature_selection.txt", "w") as f:
        f.write(" Top 5 features selected using Chi-Square (with original names):\n")
        for name in selected_feature_names:
            f.write(f"- {name}\n")

# Task 4: Hyperparameter Tuning and Model Comparison
def tune_and_compare_models(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    output_lines = []

    models = {
        "LogisticRegression": (LogisticRegression(), ParamGridBuilder()
                               .addGrid(LogisticRegression.regParam, [0.01, 0.1])
                               .build()),
        "DecisionTree": (DecisionTreeClassifier(), ParamGridBuilder()
                         .addGrid(DecisionTreeClassifier.maxDepth, [3, 5, 10])
                         .build()),
        "RandomForest": (RandomForestClassifier(), ParamGridBuilder()
                         .addGrid(RandomForestClassifier.numTrees, [10, 20])
                         .build()),
        "GBT": (GBTClassifier(), ParamGridBuilder()
                .addGrid(GBTClassifier.maxIter, [10, 20])
                .build())
    }

    for name, (model, paramGrid) in models.items():
        output_lines.append(f"\n Tuning {name}...")
        cv = CrossValidator(estimator=model,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=5)
        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel
        predictions = best_model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        output_lines.append(f" {name} Best AUC: {auc:.4f}")
        output_lines.append(f"  Best Params: {best_model.extractParamMap()}\n")

    with open(f"{output_dir}/task4_model_comparison.txt", "w") as f:
        f.write("\n".join(output_lines))

# Run all tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
