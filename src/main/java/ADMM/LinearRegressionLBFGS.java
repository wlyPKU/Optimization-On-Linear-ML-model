package ADMM;

import Utils.*;
import math.DenseVector;
import math.SparseMap;
import java.util.List;

//TODO: To be checked ...
/**
 * Created by 王羚宇 on 2016/7/24.
 */
//According to https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
public class LinearRegressionLBFGS extends model.LinearRegression{
    public void train(SparseMap[] features, List<LabeledData> labeledData,
                      ADMMState model, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;
        double rho = 1e-4;
        double maxRho = 5;

        int lbfgsNumIteration = 10;
        int lbfgsHistory = 10;

        double rel_par = 1.0;
        double x_hat[] = new double[model.featureNum];

        DenseVector oldModel = new DenseVector(featureDim);

        for (int i = 0; i < 300; i ++) {
            long startTrain = System.currentTimeMillis();
            //Update x;
            LBFGS.train(model, lbfgsNumIteration, lbfgsHistory, rho, i, trainCorpus, "LinearRegression");

            //Update z
            for(int j = 0; j < featureDim; j++) {
                x_hat[j] = rel_par * model.x.values[j] + (1 - rel_par) * model.z.values[j];
                model.z.values[j] = x_hat[j] + model.u.values[j];
            }

            //Update u
            for(int j = 0; j < featureDim; j++) {
                model.u.values[j] +=  (x_hat[j] - model.z.values[j]);
            }
            //rho = Math.min(rho * 1.1, maxRho);

            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = test(trainCorpus, model.x);
            double accuracy = test(testCorpus, model.x);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testResidual=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
            double []trainAccuracy = Utils.LinearAccuracy(trainCorpus, model.x);
            double []testAccuracy = Utils.LinearAccuracy(testCorpus, model.x);
            System.out.println("Train Accuracy:");
            Utils.printAccuracy(trainAccuracy);
            System.out.println("Test Accuracy:");
            Utils.printAccuracy(testAccuracy);
            //rho = Math.min(rho * 1.1, maxRho);
            if(converage(oldModel, model.x)){
                break;
            }
            System.arraycopy(model.x.values, 0, oldModel.values, 0, featureDim);
        }
    }

    public static void train(SparseMap[] corpus, List<LabeledData> labeledData, double trainRatio) {
        int dimension = corpus.length;
        LinearRegressionLBFGS lrADMM = new LinearRegressionLBFGS();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        ADMMState model = new ADMMState(dimension);
        long start = System.currentTimeMillis();
        lrADMM.train(corpus, labeledData, model, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.LinearRegressionLBFGS FeatureDim SampleDim train_path [trainRatio]");
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
