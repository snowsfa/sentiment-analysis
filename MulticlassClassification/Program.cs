using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Common;

namespace MulticlassClassification
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "sentiment_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "sentiment_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<SentimentData, SentimentPrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<SentimentData>(_trainDataPath, hasHeader: true);
            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
            Evaluate();
            PredictIssue();

            Console.ReadLine();
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _trainedModel.CreatePredictionEngine<SentimentData, SentimentPrediction>(_mlContext);

            SentimentData issue = new SentimentData()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            return trainingPipeline;
        }

        public static void Evaluate()
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<SentimentData>(_testDataPath, hasHeader: true);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.AccuracyMacro:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            Util.SaveModelAsFile(_modelPath, _mlContext, _trainedModel);
        }

        private static void PredictIssue()
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = _mlContext.Model.Load(stream);
            }

            SentimentData singleIssue = new SentimentData() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };

            _predEngine = loadedModel.CreatePredictionEngine<SentimentData, SentimentPrediction>(_mlContext);
            var prediction = _predEngine.Predict(singleIssue);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }
    }
}
