using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SentimentAnalysis
{
    class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        // [ColumnName("Probability")]
        public float Probability { get; set; }

        // [ColumnName("Score")]
        public float Score { get; set; }
    }
}
