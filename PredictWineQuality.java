package net.example.wine;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.*;

public class PredictWineQuality {
  public static void main(String[] args) {
    if (args.length < 2) {
      System.err.println("Usage: PredictWineQuality <test.csv> <modelPath>");
      System.exit(1);
    }
    String testPath = args[0], modelPath = args[1];

    SparkSession spark = SparkSession.builder()
            .config(new SparkConf().setAppName("WineQualityPredict"))
            .getOrCreate();

    Dataset<Row> test = spark.read().option("header", "true").option("inferSchema", "true").csv(testPath);
    PipelineModel model = PipelineModel.load(modelPath);

    double f1 = new MulticlassClassificationEvaluator().setLabelCol("quality").setMetricName("f1")
                    .evaluate(model.transform(test));
    System.out.printf("F1 score: %.4f%n", f1);
    spark.stop();
  }
}