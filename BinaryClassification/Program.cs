﻿using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Common;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "sentiment_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            TrainCatalogBase.TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            Evaluate(mlContext, model, splitDataView.TestSet);

            UseModelWithSingleItem(mlContext, model, "This was a horrible meal.");
            UseModelWithSingleItem(mlContext, model, "This was an amazing meal.");
            UseModelWithSingleItem(mlContext, model, "I didn't like this meal.");
            UseModelWithSingleItem(mlContext, model, "I love this spaghetti.");
            UseModelWithSingleItem(mlContext, model, "I don't like this spaghetti.");

            UseLoadedModelWithBatchItems(mlContext);
            Console.ReadLine();
        }

        public static TrainCatalogBase.TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainCatalogBase.TrainTestData splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            Util.SaveModelAsFile(_modelPath, mlContext, model);
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model, string sentimentText)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = sentimentText
            };

            var resultprediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void UseLoadedModelWithBatchItems(MLContext mlContext)
        {IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal."
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                },
                new SentimentData
                {
                    SentimentText = "I like this project."
                },
                new SentimentData
                {
                    SentimentText = "This project became hard to work with."
                }
            };

            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            IDataView sentimentStreamingDataView = mlContext.Data.LoadFromEnumerable(sentiments);
            IDataView predictions = loadedModel.Transform(sentimentStreamingDataView);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");
            
            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            IEnumerable<(SentimentData sentiment, SentimentPrediction prediction)> sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));
            
            foreach ((SentimentData sentiment, SentimentPrediction prediction) item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Positive" : "Negative")} | Probability: {item.prediction.Probability} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
        }

    }
}