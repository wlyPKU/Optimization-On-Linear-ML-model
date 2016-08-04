package ADMM;

import Utils.*;
import math.SparseMap;
import math.DenseVector;
import GradientDescent.Lasso;
import GradientDescent.SVM;
import java.util.List;

//TODO: To be checked ...
/**
 * Created by 王羚宇 on 2016/7/24.
 */
//According to https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
public class LinearRegressionLBFGS {
    public double test(List<LabeledData> list, DenseVector model) {
        double residual = 0;
        for (LabeledData labeledData : list) {
            double dot_prod = model.dot(labeledData.data);
            residual += Math.pow(labeledData.label - dot_prod, 2);
        }
        return residual;
    }
    public static void trainWithMinHash(List<LabeledData> corpus, int K, int b, double lambda) {
        int dim = corpus.get(0).data.dim;
        long startMinHash = System.currentTimeMillis();
        List<LabeledData> hashedCorpus = SVM.minhash(corpus, K, dim, b);
        long minHashTime = System.currentTimeMillis() - startMinHash;
        dim = hashedCorpus.get(0).data.dim;
        corpus = hashedCorpus;
        System.out.println("Utils.MinHash takes " + minHashTime + " ms" + " the dimension is " + dim);

        Lasso lasso = new Lasso();
        DenseVector modelOfU = new DenseVector(dim);
        DenseVector modelOfV = new DenseVector(dim);
        long start = System.currentTimeMillis();
        lasso.train(corpus, modelOfU, modelOfV, lambda);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public void train(SparseMap[] features, List<LabeledData> labeledData,
                      ADMMState model, double trainRatio) {
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

        double rel_par = 1.0;
        double x_hat[] = new double[model.featureNum];

        for (i = 0; i < 30; i ++) {
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
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = test(trainCorpus, model.x);
            double accuracy = test(testCorpus, model.x);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testResidual=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
            //rho = Math.min(rho * 1.1, maxRho);
        }
    }


    public static void train(SparseMap[] corpus, List<LabeledData> labeledData, double trainRatio) {
        int dim = corpus.length;
        LinearRegressionLBFGS lrADMM = new LinearRegressionLBFGS();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        ADMMState model = new ADMMState(dim);
        long start = System.currentTimeMillis();
        lrADMM.train(corpus, labeledData, model, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.LinearRegression FeatureDim SampleDim train_path [trainRatio]");
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
