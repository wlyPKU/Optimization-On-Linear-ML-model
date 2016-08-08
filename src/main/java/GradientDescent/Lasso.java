package GradientDescent;

import Utils.*;
import math.DenseVector;
import java.util.*;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class Lasso extends model.Lasso{

    private void sgdOneEpoch(List<LabeledData> list, DenseVector modelOfU,
                            DenseVector modelOfV, double lr, double lambda) {
        for (LabeledData labeledData: list) {
            double scala = labeledData.label - modelOfU.dot(labeledData.data)
                    + modelOfV.dot(labeledData.data);
            modelOfU.plusGradient(labeledData.data, scala * lr);
            modelOfU.allPlusBy(- lr * lambda);
            modelOfV.plusGradient(labeledData.data, - scala * lr);
            modelOfV.allPlusBy(- lr * lambda);
            modelOfU.positiveValueOrZero();
            modelOfV.positiveValueOrZero();
        }
    }

    public void train(List<LabeledData> corpus, DenseVector modelOfU,
                      DenseVector modelOfV, double lambda) {
        Collections.shuffle(corpus);
        int size = corpus.size();
        int end = (int) (size * 0.5);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        DenseVector model = new DenseVector(modelOfU.dim);

        for (int i = 0; i < 100; i ++) {
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
            System.out.println("loss=" + loss + " Test Loss =" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
            double []trainAccuracy = Utils.LinearAccuracy(trainCorpus, model);
            double []testAccuracy = Utils.LinearAccuracy(testCorpus, model);
            System.out.println("Train Accuracy:");
            Utils.printAccuracy(trainAccuracy);
            System.out.println("Test Accuracy:");
            Utils.printAccuracy(testAccuracy);
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


    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: GradientDescent.Lasso dim train_path lambda");
        int dim = Integer.parseInt(argv[0]);
        String path = argv[1];
        double lambda = Double.parseDouble(argv[2]);
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(corpus, lambda);
    }
}
