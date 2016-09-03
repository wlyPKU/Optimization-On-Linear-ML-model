package ADMM;

import Utils.*;
import math.DenseVector;

import java.util.List;

//TODO: To be checked...
//According to the angel ADMM logistic regression
/**
 * Created by 王羚宇 on 2016/7/24.
 */
//https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
//https://web.stanford.edu/~boyd/papers/pdf/admm_slides.pdf
//https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
//https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
//http://www.simonlucey.com/lasso-using-admm/
//http://users.ece.gatech.edu/~justin/CVXOPT-Spring-2015/resources/14-notes-admm.pdf

//https://web.stanford.edu/~boyd/papers/admm/logreg-l1/logreg.html
public class LogisticRegression extends model.LogisticRegression{

    public void train(int featureDim, List<LabeledData> labeledData,
                      ADMMState model, double lambda, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        double rho = 1e-4;
        double maxRho = 5;

        DenseVector oldModel = new DenseVector(featureDim);
        //Parameter:
        int lbfgsNumIteration = 10;
        int lbfgsHistory = 10;
        double rel_par = 1.0;
        double x_hat[] = new double[model.featureNum];
        for (int i = 0; i < 100; i ++) {
            long startTrain = System.currentTimeMillis();
            //Update x;
            LBFGS.train(model, lbfgsNumIteration, lbfgsHistory, rho, i, trainCorpus, "logisticRegression");
            //Update z
            for(int j = 0; j < featureDim; j++) {
                //Z=Soft_threshold(lambda/(rho*N),x_hat+u);
                x_hat[j] = rel_par * model.x.values[j] + (1 - rel_par) * model.z.values[j];
                model.z.values[j] = Utils.soft_threshold(lambda / rho, x_hat[j] + model.u.values[j]);
            }

            //Update u
            for(int j = 0; j < featureDim; j++) {
                //u=u+(x_hat-z)
                model.u.values[j] = model.u.values[j] + (x_hat[j] - model.z.values[j]);
            }
            //rho = Math.min(maxRho, rho * 1.1);

            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = logLoss(trainCorpus, model.x, model.z,lambda);
            double trainAuc = auc(trainCorpus, model.x);
            double testAuc = auc(testCorpus, model.x);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc +
                    " trainTime=" + trainTime + " testTime=" + testTime);
            if(converage(oldModel, model.x)){
                //break;
            }
            System.arraycopy(model.x.values, 0, oldModel.values, 0, featureDim);
        }
    }


    public static void train(int featureDim, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        LogisticRegression lrADMM = new LogisticRegression();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        ADMMState model = new ADMMState(featureDim);
        long start = System.currentTimeMillis();
        lrADMM.train(featureDim, labeledData, model, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.LogisticRegression FeatureDim train_path lambda trainRatio");
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
