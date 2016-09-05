package GradientDescent;

import Utils.*;
import math.DenseVector;
import java.util.*;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class LinearRegression extends model.LinearRegression{

    static double trainRatio = 0.5;
    private void sgdOneEpoch(List<LabeledData> list, DenseVector model, double lr) {
        for (LabeledData labeledData: list) {
            double scala = labeledData.label - model.dot(labeledData.data);
            model.plusGradient(labeledData.data, + scala * lr);
        }
    }
    public void train(List<LabeledData> corpus, DenseVector model) {
        Collections.shuffle(corpus);
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        DenseVector oldModel = new DenseVector(model.dim);
        for (int i = 0; i < 100; i ++) {
            long startTrain = System.currentTimeMillis();
            //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            sgdOneEpoch(trainCorpus, model, 0.005);
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = test(trainCorpus, model);
            double accuracy = test(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testAuc=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
            double []trainAccuracy = Utils.LinearAccuracy(trainCorpus, model);
            double []testAccuracy = Utils.LinearAccuracy(testCorpus, model);
            System.out.println("Train Accuracy:");
            Utils.printAccuracy(trainAccuracy);
            System.out.println("Test Accuracy:");
            Utils.printAccuracy(testAccuracy);
            if(converge(oldModel, model)){
                //break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
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


    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: GradientDescent.LinearRegression dim train_path [trainRatio]");
        int dim = Integer.parseInt(argv[0]);
        String path = argv[1];
        long startLoad = System.currentTimeMillis();
        if(argv.length >= 3){
            trainRatio = Double.parseDouble(argv[2]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");

        train(corpus);
    }
}
