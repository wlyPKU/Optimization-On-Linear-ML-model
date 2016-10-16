package GradientDescent;

import Utils.*;
import math.DenseVector;
import java.util.*;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class Lasso extends model.Lasso{

    static double trainRatio = 0.5;
    private void sgdOneEpoch(List<LabeledData> list, DenseVector modelOfU,
                            DenseVector modelOfV, double lr, double lambda) {
        double modelPenalty = -lr * lambda / list.size();
        for (LabeledData labeledData: list) {
            double scala = labeledData.label - modelOfU.dot(labeledData.data)
                    + modelOfV.dot(labeledData.data);
            modelOfU.plusSparse(labeledData.data, modelPenalty);
            modelOfU.plusGradient(labeledData.data, scala * lr);
            modelOfV.plusSparse(labeledData.data, modelPenalty);
            modelOfV.plusGradient(labeledData.data, - scala * lr);
            modelOfU.positiveOrZero(labeledData.data);
            modelOfV.positiveOrZero(labeledData.data);
        }
    }

    public void train(List<LabeledData> corpus, DenseVector modelOfU,
                      DenseVector modelOfV, double lambda) {
        Collections.shuffle(corpus);
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        DenseVector model = new DenseVector(modelOfU.dim);

        DenseVector oldModel = new DenseVector(model.dim);

        for (int i = 0; ; i ++) {
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
            System.out.println("loss=" + loss + " TestLoss=" + accuracy +
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
        System.out.println("Usage: GradientDescent.LassoLBFGS dim train_path lambda [trainRatio]");
        int dim = Integer.parseInt(argv[0]);
        String path = argv[1];
        double lambda = Double.parseDouble(argv[2]);
        long startLoad = System.currentTimeMillis();
        if(argv.length >= 4){
            trainRatio = Double.parseDouble(argv[3]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(corpus, lambda);
    }
}
