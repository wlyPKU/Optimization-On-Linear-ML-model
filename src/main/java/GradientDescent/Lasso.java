package GradientDescent;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.Collections;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class Lasso {
    public double lassoLoss(List<LabeledData> list, DenseVector model, double lamda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double predictValue = model.dot(labeledData.data);
            loss += 1 / 2 * Math.pow(labeledData.label - predictValue, 2);
        }
        for(Double v: model.values){
            loss += lamda * (v > 0? v : -v);
        }
        return loss;
    }
    public void sgdOneEpoch(List<LabeledData> list, DenseVector modelOfU,
                            DenseVector modelOfV, double lr, double lambda) {
        for (LabeledData labeledData: list) {
            double scala = labeledData.label - modelOfU.dot(labeledData.data)
                    + modelOfV.dot(labeledData.data);
            modelOfU.plusBy(labeledData.data, (-lambda + scala) * lr);
            modelOfV.plusBy(labeledData.data, (-lambda - scala) * lr);
            modelOfU.positiveValueOrZero(labeledData.data);
            modelOfV.positiveValueOrZero(labeledData.data);
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
    public void train(List<LabeledData> corpus, DenseVector modelOfU,
                      DenseVector modelOfV, double lambda) {
        Collections.shuffle(corpus);
        int size = corpus.size();
        int end = (int) (size * 0.5);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        DenseVector model = new DenseVector(modelOfU.dim);

        for (int i = 0; i < 30; i ++) {
            long startTrain = System.currentTimeMillis();
            //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            sgdOneEpoch(trainCorpus, modelOfU, modelOfV, 0.005, lambda);
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            for(int j = 0; j < model.dim; j++){
                model.values[j] = modelOfU.values[j] - modelOfV.values[j];
            }
            double loss = lassoLoss(trainCorpus, model, lambda);
            double accuracy = test(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testAuc=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }


    public static void train(List<LabeledData> corpus, double lambda) {
        int dim = corpus.get(0).data.dim;
        Lasso lasso = new Lasso();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector modelOfU = new DenseVector(dim);
        DenseVector modelOfV = new DenseVector(dim);

        long start = System.currentTimeMillis();
        lasso.train(corpus, modelOfU, modelOfV, lambda);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
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

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: GradientDescent.Lasso dim train_path lamda [true|false] K b");
        int dim = Integer.parseInt(argv[0]);
        String path = argv[1];
        double lambda = Double.parseDouble(argv[2]);
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");

        boolean minhash = Boolean.parseBoolean(argv[3]);
        //TODO Need to think how to min hash numeric variables
        if (minhash) {
            System.out.println("Training with minhash method.");
            int K = Integer.parseInt(argv[4]);
            int b = Integer.parseInt(argv[5]);
            trainWithMinHash(corpus, K, b, lambda);
        } else {
            train(corpus, lambda);
        }
    }
}
