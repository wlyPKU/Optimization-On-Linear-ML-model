package CoordinateDescent;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import it.unimi.dsi.fastutil.objects.ObjectSet;
import math.SparseMap;
import math.DenseVector;
import Utils.LabeledData;
import Utils.Utils;
import java.util.*;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class Lasso {
    private double lassoLoss(List<LabeledData> list, DenseVector model, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double predictValue = model.dot(labeledData.data);
            loss += 1 / 2 * Math.pow(labeledData.label - predictValue, 2);
        }
        for(Double v: model.values){
            loss += lambda * (v > 0? v : -v);
        }
        return loss;
    }
    public double test(List<LabeledData> list, DenseVector model) {
        double residual = 0;
        for (LabeledData labeledData : list) {
            double dot_prod = model.dot(labeledData.data);
            residual += Math.pow(labeledData.label - dot_prod, 2);
        }
        return residual;
    }

    public void train(SparseMap[] features, List<LabeledData> labeledData,
                      DenseVector model, double lambda, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;
        int sampleSize = features[featureDim].map.size();
        double featureSquare[] = new double[featureDim];
        double residual[] = new double[sampleSize];
        for(int i = 0; i < featureDim; i++){
            featureSquare[i] = 0;
            for(Double v: features[i].map.values()){
                featureSquare[i] += v * v;
            }
            if(featureSquare[i] == 0){
                featureSquare[i] = Double.MAX_VALUE;
            }
        }
        int i = 0;
        for(Double y : features[featureDim].map.values()){
            residual[i] = y;
            i++;
        }
        for (i = 0; i < 300; i ++) {
            long startTrain = System.currentTimeMillis();
            for(int j = 0; j < featureDim; j++){
                double oldValue = model.values[j];
                double updateValue = 0;

                ObjectIterator<Int2DoubleMap.Entry> iter =  features[j].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double xj = entry.getDoubleValue();
                    updateValue += xj * (xj * model.values[j] + residual[idx]);

                }

                updateValue /= featureSquare[j];
                model.values[j] = updateValue;
                model.values[j] = Utils.soft_threshold(lambda / featureSquare[j], model.values[j]);

                iter =  features[j].map.int2DoubleEntrySet().iterator();

                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double value = entry.getDoubleValue();
                    residual[idx] -= (model.values[j] - oldValue) * value;

                }
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = lassoLoss(trainCorpus, model, lambda);
            double accuracy = test(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testResidual=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }


    public static void train(SparseMap[] corpus, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        int dim = corpus.length;
        Lasso lassoCD = new Lasso();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector model = new DenseVector(dim);
        for(int i = 0; i < dim; i++){
            model.values[i] = 0;
        }
        long start = System.currentTimeMillis();
        lassoCD.train(corpus, labeledData, model, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }


    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: CoordinateDescent.Lasso FeatureDim SampleDim train_path lamda trainRatio");
        int featureDim = Integer.parseInt(argv[0]);
        int sampleDim = Integer.parseInt(argv[1]);
        String path = argv[2];
        double lambda = Double.parseDouble(argv[3]);
        double trainRatio = 0.5;
        if(argv.length >= 5){
            trainRatio = Double.parseDouble(argv[4]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long startLoad = System.currentTimeMillis();
        SparseMap[] features = Utils.LoadLibSVMByFeature(path, featureDim, sampleDim, trainRatio);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        //TODO Need to think how to min hash numeric variables
        train(features, labeledData, lambda, trainRatio);
    }
}
