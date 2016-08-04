package CoordinateDescent;

import math.SparseMap;
import math.DenseVector;
import Utils.LabeledData;
import Utils.Utils;
import java.util.List;
import java.util.Map;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class LinearRegression {
    public double test(List<LabeledData> list, DenseVector model) {
        double residual = 0;
        for (LabeledData labeledData : list) {
            double dot_prod = model.dot(labeledData.data);
            residual += Math.pow(labeledData.label - dot_prod, 2);
        }

        return residual;
    }

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
            if(featureSquare[i] == 0){
                featureSquare[i] = Double.MAX_VALUE;
            }
        }
        int i = 0;
        for(Double y : features[featureDim].map.values()){
            residual[i] = y;
            i++;
        }
        for (i = 0; i < 30; i ++) {
            long startTrain = System.currentTimeMillis();
            for(int j = 0; j < featureDim; j++){
                double oldValue = model.values[j];
                double updateValue = 0;
                for(Map.Entry<Integer, Double> m: features[j].map.entrySet()){
                    int idx = m.getKey();
                    double xj = m.getValue();
                    double tmpValue = xj * (residual[idx] + xj * model.values[j]);
                    updateValue += tmpValue;
                }

                updateValue /= featureSquare[j];
                model.values[j] = updateValue;
                for(Map.Entry<Integer, Double> m: features[j].map.entrySet()){
                    int idx = m.getKey();
                    residual[idx] -= (model.values[j] - oldValue) * m.getValue();
                }

            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = test(trainCorpus, model);
            double accuracy = test(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testResidual=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }


    public static void train(SparseMap[] corpus, List<LabeledData> labeledData, double trainRatio) {
        int dim = corpus.length;
        LinearRegression linearRegressionCD = new LinearRegression();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector model = new DenseVector(dim);
        for(int i = 0; i < dim; i++){
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
        //TODO Need to think how to min hash numeric variables
        train(features, labeledData, trainRatio);
    }
}