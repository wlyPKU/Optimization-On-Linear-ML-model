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
    public void train(int featureDim, List<LabeledData> labeledData,
                      ADMMState model, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        double rho = 1e-4;
        double maxRho = 5;

        int lbfgsNumIteration = 10;
        int lbfgsHistory = 10;

        double rel_par = 1.0;
        double x_hat[] = new double[model.featureNum];

        DenseVector oldModel = new DenseVector(featureDim);

        for (int i = 0; i < 100; i ++) {
            long startTrain = System.currentTimeMillis();
            //Update x;
            LBFGS.train(model, lbfgsNumIteration, lbfgsHistory, rho, i, trainCorpus, "LinearRegressionModelParallel");

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
            if(converge(oldModel, model.x)){
                //break;
            }
            System.arraycopy(model.x.values, 0, oldModel.values, 0, featureDim);
        }
    }

    public static void train(int featureDim, List<LabeledData> labeledData, double trainRatio) {
        LinearRegressionLBFGS lrADMM = new LinearRegressionLBFGS();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        ADMMState model = new ADMMState(featureDim);
        long start = System.currentTimeMillis();
        lrADMM.train(featureDim, labeledData, model, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.LinearRegressionLBFGS FeatureDim train_path [trainRatio]");
        int featureDim = Integer.parseInt(argv[0]);
        String path = argv[1];
        double trainRatio = 0.5;
        if(argv.length >= 3){
            trainRatio = Double.parseDouble(argv[2]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long startLoad = System.currentTimeMillis();
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(featureDim, labeledData, trainRatio);
    }
}
