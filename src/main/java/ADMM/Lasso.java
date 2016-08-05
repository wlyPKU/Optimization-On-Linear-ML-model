package ADMM;

import Utils.*;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.SparseMap;
import math.DenseVector;
import java.util.List;
import java.util.Map;

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
public class Lasso {
    private double lassoLoss(List<LabeledData> list, DenseVector model_x, DenseVector model_z, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double predictValue = model_x.dot(labeledData.data);
            loss += 1.0 / 2.0 * Math.pow(labeledData.label - predictValue, 2);
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
        //Initialize the second part of B
        //Calculate (A^Tb)
        double []part2OfB = new double[featureDim];
        double []tmpPart2OfB = new double[featureDim];
        double [][]tmpPart1OfB = new double[featureDim][featureDim];
        for(int r = 0; r < featureDim; r++){
            tmpPart2OfB[r] = 0;
            ObjectIterator<Int2DoubleMap.Entry> iter =  features[r].map.int2DoubleEntrySet().iterator();
            while (iter.hasNext()) {
                Int2DoubleMap.Entry entry = iter.next();
                int idx = entry.getIntKey();
                double value = entry.getDoubleValue();
                tmpPart2OfB[r] += value * features[featureDim].map.get(idx);
            }
        }
        //Store A^T*A
        for(i = 0; i < featureDim; i++){
            for(int j = 0; j < featureDim; j++){
                tmpPart1OfB[i][j] = features[j].multiply(features[i]);
            }
        }
        double x_hat[] = new double[model.featureNum];
        for (i = 0; i < 100; i ++) {
            //Calculate (A^Tb+rho*(z-u))
            for(int r = 0; r < featureDim; r++) {
                part2OfB[r] = tmpPart2OfB[r] + rho * (model.z.values[r] - model.u.values[r]);
            }
            long startTrain = System.currentTimeMillis();
            //Update x;
            for(int j = 0; j < featureDim; j++){
                model.x.values[j] = 0;
                for (int ite = 0; ite < featureDim; ite++) {
                    //Calculate (A^T*A+rho*I)_j_ite
                    //double part1OfB_j_ite = features[j].multiply(features[ite]);
                    double part1OfB_j_ite = tmpPart1OfB[ite][j];
                    if (j == ite) {
                        part1OfB_j_ite += rho * 1;
                    }
                    model.x.values[j] += part1OfB_j_ite * part2OfB[ite];
                }
            }
            //https://github.com/afbujan/admm_lasso/blob/master/lasso_admm.py
            double rel_par = 1.0;
            for(int id = 0; id < featureDim; id++){
                x_hat[id] = rel_par * model.x.values[id] + (1 - rel_par) * model.z.values[id];
            }
            //Update z
            for(int j = 0; j < featureDim; j++) {
                //z=Soft_threshold(lambda/rho,x+u);
                model.z.values[j] = Utils.soft_threshold(lambda / rho, x_hat[j]
                        + model.u.values[j]);
            }

            //Update u
            for(int j = 0; j < featureDim; j++) {
                //u=u+(x_hat-z)
                model.u.values[j] += (x_hat[j] - model.z.values[j]);
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = lassoLoss(trainCorpus, model.x, model.z, lambda);
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
        }
    }


    public static void train(SparseMap[] corpus, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        int dim = corpus.length;
        Lasso lassoADMM = new Lasso();
        ADMMState model = new ADMMState(dim);
        long start = System.currentTimeMillis();
        lassoADMM.train(corpus, labeledData, model, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.Lasso FeatureDim SampleDim train_path lamda trainRatio");
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
