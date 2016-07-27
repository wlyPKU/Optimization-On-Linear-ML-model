package GradientDescent;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.Collections;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class LinearRegression {
    public double linearLoss(List<LabeledData> list, DenseVector model) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double predictValue = model.dot(labeledData.data);
            loss += 1 / 2 * Math.pow(labeledData.label - predictValue, 2);
        }
        return loss;
    }

    public void sgdOneEpoch(List<LabeledData> list, DenseVector model, double lr) {
        for (LabeledData labeledData: list) {
            double scala = labeledData.label - model.dot(labeledData.data);
            model.plusGradient(labeledData.data, + scala * lr);
        }
    }
    public double test(List<LabeledData> list, DenseVector model) {
        double residual = 0;
        for (LabeledData labeledData : list) {
            double dot_prod = model.dot(labeledData.data);
            residual += Math.pow(labeledData.label - dot_prod, 2);
        }

        return residual;
    }
    public void train(List<LabeledData> corpus, DenseVector model) {
        Collections.shuffle(corpus);
        int size = corpus.size();
        int end = (int) (size * 0.5);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        for (int i = 0; i < 30; i ++) {
            long startTrain = System.currentTimeMillis();
            //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            sgdOneEpoch(trainCorpus, model, 0.005);
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = linearLoss(trainCorpus, model);
            double accuracy = test(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testAuc=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }


    public static void train(List<LabeledData> corpus) {
        int dim = corpus.get(0).data.dim;
        LinearRegression lr = new LinearRegression();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        lr.train(corpus, model);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void trainWithMinHash(List<LabeledData> corpus, int K, int b) {
        int dim = corpus.get(0).data.dim;
        long startMinHash = System.currentTimeMillis();
        List<LabeledData> hashedCorpus = SVM.minhash(corpus, K, dim, b);
        long minHashTime = System.currentTimeMillis() - startMinHash;
        dim = hashedCorpus.get(0).data.dim;
        corpus = hashedCorpus;
        System.out.println("Utils.MinHash takes " + minHashTime + " ms" + " the dimension is " + dim);

        LinearRegression lr = new LinearRegression();
        DenseVector model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        lr.train(corpus, model);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: GradientDescent.LinearRegression dim train_path [true|false] K b");
        int dim = Integer.parseInt(argv[0]);
        String path = argv[1];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");

        boolean minhash = Boolean.parseBoolean(argv[2]);
        if (minhash) {
            System.out.println("Training with minhash method.");
            int K = Integer.parseInt(argv[3]);
            int b = Integer.parseInt(argv[4]);
            trainWithMinHash(corpus, K, b);
        } else {
            train(corpus);
        }
    }
}
