using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Common
{
    public abstract class Util
    {
        public static void SaveModelAsFile(String modelPath, MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("The model is saved to {0}", modelPath);
        }
    }
}
