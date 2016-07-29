package ADMM;

import GradientDescent.*;
import Utils.*;
import it.unimi.dsi.fastutil.doubles.DoubleComparator;
import math.DenseMap;
import math.DenseVector;

import java.util.Collections;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
public class SVM {
    public double auc(List<LabeledData> list, DenseVector model) {
        int length = list.size();
        System.out.println(length);
        double[] scores = new double[length];
        double[] labels = new double[length];

        int cnt = 0;
        for (LabeledData labeledData: list) {
            double score = model.dot(labeledData.data);

            scores[cnt] = score;
            labels[cnt] = labeledData.label;
            cnt ++;
        }

        Sort.quickSort(scores, labels, 0, length, new DoubleComparator() {

            public int compare(double i, double i1) {
                if (Math.abs(i - i1) < 10e-12) {
                    return 0;
                } else {
                    return i - i1 > 10e-12 ? 1 : -1;
                }
            }

            public int compare(Double o1, Double o2) {
                if (Math.abs(o1 - o2) < 10e-12) {
                    return 0;
                } else {
                    return o1 - o2 > 10e-12 ? 1 : -1;
                }
            }
        });

        long M = 0, N = 0;
        for (int i = 0; i < scores.length; i ++) {
            if (labels[i] == 1.0)
                M ++;
            else
                N ++;
        }

        double sigma = 0.0;
        for (long i = M + N - 1; i >= 0; i --) {
            if (labels[(int) i] == 1.0) {
                sigma += i;
            }
        }

        double auc = (sigma - (M + 1) * M / 2) / (M * N);
        System.out.println("sigma=" + sigma + " M=" + M + " N=" + N);
        return auc;
    }
    /*
    public static void trainWithMinHash(List<LabeledData> corpus, int K, int b, double lambda) {
        int dim = corpus.get(0).data.dim;
        long startMinHash = System.currentTimeMillis();
        List<LabeledData> hashedCorpus = GradientDescent.SVM.minhash(corpus, K, dim, b);
        long minHashTime = System.currentTimeMillis() - startMinHash;
        dim = hashedCorpus.get(0).data.dim;
        corpus = hashedCorpus;
        System.out.println("Utils.MinHash takes " + minHashTime + " ms" + " the dimension is " + dim);

        GradientDescent.Lasso lasso = new GradientDescent.Lasso();
        DenseVector modelOfU = new DenseVector(dim);
        DenseVector modelOfV = new DenseVector(dim);
        long start = System.currentTimeMillis();
        lasso.train(corpus, modelOfU, modelOfV, lambda);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    */
    public void train(int featureDim, List<LabeledData> labeledData,
                      ADMMState model, double lambda, double trainRatio) {
        Collections.shuffle(labeledData);

        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        double rho = 1e-4;
        double maxRho = 5;
        //Parameter:
        int lbfgsNumIteration = 10;
        int lbfgsHistory = 10;
        for (int i = 0; i < 100; i ++) {
            long startTrain = System.currentTimeMillis();
            //Update x;
            LBFGS.train(model, lbfgsNumIteration, lbfgsHistory, rho, model.z.values, i, trainCorpus, "SVM");
            //Update z
            for(int j = 0; j < featureDim; j++) {
                //Z=(1/(1/lambda + rho * N))*(x+u);
                model.z.values[j] = (rho / (1.0 / lambda + rho * featureDim)) * (model.x.values[j] + model.u.values[j]);
            }

            //Update u
            for(int j = 0; j < featureDim; j++) {
                //u=u+(B-C)
                model.u.values[j] = model.u.values[j] + (model.x.values[j] - model.z.values[j]);
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();
            double loss = SVMLoss(trainCorpus, model.x);
            double trainAuc = auc(trainCorpus, model.x);
            double testAuc = auc(testCorpus, model.x);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc +
                    " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }

    private double SVMLoss(List<LabeledData> list, DenseVector model) {
        double loss = 0.0;
        for (LabeledData labeledData : list) {
            double dotProd = model.dot(labeledData.data);
            loss += Math.max(0, 1 - dotProd * labeledData.label);
        }
        return loss / list.size();
    }
    public double test(List<LabeledData> list, DenseVector model) {
        int N_RIGHT = 0;
        int N_TOTAL = 0;
        for (LabeledData labeledData : list) {
            double dot_prod = model.dot(labeledData.data);

            if (dot_prod * labeledData.label >= 0) {
                N_RIGHT++;
            }

            N_TOTAL++;
        }

        return 1.0 * N_RIGHT / N_TOTAL;
    }
    public static void train(int featureDim, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        SVM svmADMM = new SVM();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        ADMMState model = new ADMMState(featureDim);
        long start = System.currentTimeMillis();
        svmADMM.train(featureDim, labeledData, model, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.SVM FeatureDim train_path lamda trainRatio");
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
        //TODO Need to think how to min hash numeric variables
        train(featureDim, labeledData, lambda, trainRatio);
    }
}
