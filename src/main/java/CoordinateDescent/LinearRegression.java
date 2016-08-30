package CoordinateDescent;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.*;
import Utils.*;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class LinearRegression extends model.LinearRegression{

    public void train(SparseMap[] features, List<LabeledData> labeledData,
                      DenseVector model, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;
        int sampleSize = features[featureDim].map.size();
        double featureSquare[] = new double[featureDim];
        double residual[] = new double[sampleSize];
        for(int i = 0; i < featureDim; i++){
            for(Double v: features[i].map.values()){
                featureSquare[i] += v * v;
            }
        }
        ObjectIterator<Int2DoubleMap.Entry> iter =  features[featureDim].map.int2DoubleEntrySet().iterator();
        while (iter.hasNext()) {
            Int2DoubleMap.Entry entry = iter.next();
            int idx = entry.getIntKey();
            double y = entry.getDoubleValue();
            residual[idx] = y;
        }

        DenseVector oldModel = new DenseVector(featureDim);
        for (int i = 0; i < 300; i ++) {
            long startTrain = System.currentTimeMillis();
            for(int j = 0; j < featureDim; j++){
                double oldValue = model.values[j];
                double updateValue = 0;
                iter =  features[j].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double xj = entry.getDoubleValue();
                    double tmpValue = xj * (residual[idx] + xj * model.values[j]);
                    updateValue += tmpValue;
                }

                updateValue /= featureSquare[j];
                model.values[j] = updateValue;
                iter =  features[j].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double xj = entry.getDoubleValue();
                    residual[idx] -= (model.values[j] - oldValue) * xj;
                }

            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = test(trainCorpus, model);
            double accuracy = test(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testResidual=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
            double []trainAccuracy = Utils.LinearAccuracy(trainCorpus, model);
            double []testAccuracy = Utils.LinearAccuracy(testCorpus, model);
            System.out.println("Train Accuracy:");
            Utils.printAccuracy(trainAccuracy);
            System.out.println("Test Accuracy:");
            Utils.printAccuracy(testAccuracy);
            if(converage(oldModel, model)){
                break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDim);
        }
    }


    public static void train(SparseMap[] corpus, List<LabeledData> labeledData, double trainRatio) {
        int dimension = corpus.length;
        LinearRegression linearRegressionCD = new LinearRegression();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector model = new DenseVector(dimension);
        for(int i = 0; i < dimension; i++){
            model.values[i] = 0;
        }
        long start = System.currentTimeMillis();
        linearRegressionCD.train(corpus, labeledData, model, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }


    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: CoordinateDescent.LinearRegression FeatureDim SampleDim train_path [trainRatio]");
        int featureDim = Integer.parseInt(argv[0]);
        int sampleDim = Integer.parseInt(argv[1]);
        String path = argv[2];
        double trainRatio = 0.5;
        if(argv.length >= 4){
            trainRatio = Double.parseDouble(argv[3]);
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
        train(features, labeledData, trainRatio);
    }
}
