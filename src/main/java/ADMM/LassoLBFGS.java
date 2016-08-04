package ADMM;

import Utils.*;
import math.SparseMap;
import math.DenseVector;
import java.util.List;

//TODO: To be checked ...
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
public class LassoLBFGS {
    private double lassoLoss(List<LabeledData> list, DenseVector model_x, DenseVector model_z, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double predictValue = model_x.dot(labeledData.data);
            loss += 1 / 2 * Math.pow(labeledData.label - predictValue, 2);
        }
        for(Double v: model_z.values){
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
                      ADMMState model, double lambda, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;
        int i;
        double rho = 1e-4;
        double maxRho = 5;

        int lbfgsNumIteration = 10;
        int lbfgsHistory = 10;

        double x_hat[] = new double[model.featureNum];
        for (i = 0; i < 300; i ++) {
            long startTrain = System.currentTimeMillis();
            //Update x;
            LBFGS.train(model, lbfgsNumIteration, lbfgsHistory, rho, i, trainCorpus, "lasso");
            double rel_par = 1.0;

            //Update z
            for(int id = 0; id < featureDim; id++){
                x_hat[id] = rel_par * model.x.values[id] + (1 - rel_par) * model.z.values[id];
                //z=Soft_threshold(lambda/rho,x+u);
                model.z.values[id] = Utils.soft_threshold(lambda / rho, x_hat[id]
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

            double loss = lassoLoss(trainCorpus, model.x, model.z, lambda);
            double accuracy = test(testCorpus, model.x);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testResidual=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
            //rho = Math.min(rho * 1.1, maxRho);
        }
    }

    public static void train(SparseMap[] corpus, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        int dim = corpus.length;
        LassoLBFGS lassoLBFGS = new LassoLBFGS();
        ADMMState model = new ADMMState(dim);
        long start = System.currentTimeMillis();
        lassoLBFGS.train(corpus, labeledData, model, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.LassoLBFGS FeatureDim SampleDim train_path lambda trainRatio");
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
