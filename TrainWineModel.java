package net.example.wine;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.*;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.sql.*;

public class TrainWineModel {
  public static void main(String[] args) {
    if (args.length < 3) {
      System.err.println("Usage: TrainWineModel <train.csv> <valid.csv> <modelOut>");
      System.exit(1);
    }
    String trainPath = args[0], validPath = args[1], modelOut = args[2];

    SparkSession spark = SparkSession.builder()
            .config(new SparkConf().setAppName("WineQualityTrain"))
            .getOrCreate();

    Dataset<Row> train = spark.read().option("header", "true").option("inferSchema", "true").csv(trainPath);
    Dataset<Row> valid = spark.read().option("header", "true").option("inferSchema", "true").csv(validPath);

    String[] features = java.util.Arrays.stream(train.columns()).filter(c -> !c.equals("quality")).toArray(String[]::new);
    VectorAssembler assembler = new VectorAssembler().setInputCols(features).setOutputCol("features");

    RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("quality").setFeaturesCol("features").setNumTrees(100);

    Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, rf});

    ParamMap[] grid = new ParamGridBuilder()
            .addGrid(rf.maxDepth(), new int[]{5, 10, 15})
            .addGrid(rf.numTrees(), new int[]{50, 100, 150})
            .build();

    TrainValidationSplit tvs = new TrainValidationSplit()
            .setEstimator(pipeline)
            .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("quality").setMetricName("f1"))
            .setEstimatorParamMaps(grid)
            .setTrainRatio(0.8);

    PipelineModel best = (PipelineModel) tvs.fit(train).bestModel();

    double f1 = new MulticlassClassificationEvaluator().setLabelCol("quality").setMetricName("f1")
                    .evaluate(best.transform(valid));
    System.out.printf("Validation F1: %.4f%n", f1);

    best.write().overwrite().save(modelOut);
    spark.stop();
  }
}