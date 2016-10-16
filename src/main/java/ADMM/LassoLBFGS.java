package ADMM;

import Utils.*;
import math.DenseVector;
import math.SparseMap;
import java.util.List;

//According to https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
/**
 * Created by 王羚宇 on 2016/7/24.
 */
//https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
//https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
//https://web.stanford.edu/~boyd/papers/pdf/admm_slides.pdf
//https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
//https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
//http://www.simonlucey.com/lasso-using-admm/
//http://users.ece.gatech.edu/~justin/CVXOPT-Spring-2015/resources/14-notes-admm.pdf
public class LassoLBFGS extends model.Lasso{
    public void train(int featureDim, List<LabeledData> labeledData,
                      ADMMState model, double lambda, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        double rho = 1e-4;
        double maxRho = 5;

        int lbfgsNumIteration = 10;
        int lbfgsHistory = 10;

        double x_hat[] = new double[model.featureNum];
        DenseVector oldModel = new DenseVector(featureDim);
        for (int i = 0; ; i ++) {
            long startTrain = System.currentTimeMillis();
            //Update x;
            LBFGS.train(model, lbfgsNumIteration, lbfgsHistory, rho, i, trainCorpus, "LassoLBFGS");
            double rel_par = 1.0;

            //Update z
            for(int id = 0; id < featureDim; id++){
                x_hat[id] = rel_par * model.x.values[id] + (1 - rel_par) * model.z.values[id];
                //z=Soft_threshold(lambda/rho,x+u);
                model.z.values[id] = Utils.soft_threshold(lambda/ rho, x_hat[id]
                        + model.u.values[id]);
            }

            //Update u
            for(int j = 0; j < featureDim; j++) {
                //u=u+(x_hat-z)
                model.u.values[j] += (x_hat[j] - model.z.values[j]);
            }

            //rho = Math.min(rho * 1.1, maxRho);

            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = lassoLoss(trainCorpus, model.x, lambda);
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

    public static void train(int featureDim, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        LassoLBFGS lassoLBFGS = new LassoLBFGS();
        ADMMState model = new ADMMState(featureDim);
        long start = System.currentTimeMillis();
        lassoLBFGS.train(featureDim, labeledData, model, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.LassoLBFGS FeatureDim train_path lambda trainRatio");
        int featureDim = Integer.parseInt(argv[0]);
        String path = argv[1];
        double lambda = Double.parseDouble(argv[2]);
        double trainRatio = 0.5;
        if(argv.length >= 4){
            trainRatio = Double.parseDouble(argv[3]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long startLoad = System.currentTimeMillis();
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(featureDim, labeledData, lambda, trainRatio);
    }
}
