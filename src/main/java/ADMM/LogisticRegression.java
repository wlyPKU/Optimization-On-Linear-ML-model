package ADMM;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseMap;
import math.DenseVector;
import GradientDescent.Lasso;
import java.util.List;
import Utils.ADMMState;
import Utils.LBFGS;
import GradientDescent.SVM;

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
public class LogisticRegression {
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
    public void train(DenseMap[] features, List<LabeledData> labeledData,
                      ADMMState model, double lambda, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;
        int sampleNum = trainCorpus.size();
        int i;
        double rho = 1e-4;
        double maxRho = 5;
        //Parameter:
        int lbfgsNumIteration = 50;
        int lbfgsHistory = 10;
        for (i = 0; i < 30; i ++) {
            long startTrain = System.currentTimeMillis();
            //Update x;
            LBFGS.train(model, lbfgsNumIteration, lbfgsHistory, rho, model.z.values, i, trainCorpus);
            //Update z
            for(int j = 0; j < featureDim; j++) {
                //Z=Soft_threshold(lambda/(rho*N),x+u);
                model.z.values[j] = Utils.soft_threshold(lambda / (rho * sampleNum), model.x.values[j]
                        + model.u.values[j]);
            }

            //Update u
            for(int j = 0; j < featureDim; j++) {
                //u=u+(B-C)
                model.u.values[j] = model.u.values[j] + (model.x.values[j] - model.z.values[j]);
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = test(trainCorpus, model.x);
            double accuracy = test(testCorpus, model.x);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testResidual=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }


    public static void train(DenseMap[] corpus, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        int dim = corpus.length;
        LogisticRegression lrADMM = new LogisticRegression();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        ADMMState model = new ADMMState(dim);
        long start = System.currentTimeMillis();
        lrADMM.train(corpus, labeledData, model, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.LogisticRegression FeatureDim SampleDim train_path lamda trainRatio");
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
        DenseMap[] features = Utils.LoadLibSVMByFeature(path, featureDim, sampleDim, trainRatio);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        //TODO Need to think how to min hash numeric variables
        train(features, labeledData, lambda, trainRatio);
    }
}
