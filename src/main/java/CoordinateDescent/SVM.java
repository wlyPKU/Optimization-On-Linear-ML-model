package CoordinateDescent;

import math.*;
import Utils.*;
import java.util.*;

/**
 * Created by leleyu on 2016/6/30.
 */
//http://www.tuicool.com/m/articles/RRZvYb
//https://github.com/acharuva/svm_cd/blob/master/svm_cd.py
public class SVM extends model.SVM{

    static double trainRatio = 0.5;
    public void train(List<LabeledData> corpus, DenseVector model, double lambda) {
        Collections.shuffle(corpus);

        int size = corpus.size();
        int end = (int) (size * trainRatio);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);

        //TODO https://github.com/acharuva/svm_cd/blob/master/svm_cd.py
        double []Q = new double[trainCorpus.size()];
        int index = 0;
        for(LabeledData l: trainCorpus){
            Q[index] = 0;
            if(l.data.values == null){
                //binary
                Q[index] = l.data.indices.length;
            }else{
                for(double v: l.data.values) {
                    Q[index] += v * v;
                }
            }
            index++;
        }

        double []alpha = new double[trainCorpus.size()];
        Arrays.fill(alpha, 0);

        double C = 1.0 / (2.0 * lambda);
        DenseVector oldModel = new DenseVector(model.values.length);
        for (int i = 0; i < 300; i ++) {
            long startTrain = System.currentTimeMillis();
            //Coordinate Descent
            int j = 0;
            for (LabeledData labeledData : trainCorpus) {
                double G = model.dot(labeledData.data) * labeledData.label - 1;
                double alpha_old = alpha[j];
                double PG = 0;
                if(alpha[j] == 0){
                    PG = Math.min(G, 0);
                }else if(alpha[j] == C){
                    PG = Math.max(G, 0);
                }else if(alpha[j] > 0 && alpha[j] < C){
                    PG = G;
                }
                if(PG != 0) {
                    alpha[j] = Math.min(Math.max(0, alpha[j] - G / Q[j]), C);
                    int r = 0;
                    for (Integer idx : labeledData.data.indices) {
                        if (labeledData.data.values == null) {
                            model.values[idx] += (alpha[j] - alpha_old) * labeledData.label;
                        } else {
                            model.values[idx] += (alpha[j] - alpha_old) * labeledData.label * labeledData.data.values[r];
                            r++;
                        }
                    }
                }
                j++;
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();
            double loss = SVMLoss(trainCorpus, model, lambda);
            double trainAuc = auc(trainCorpus, model);
            double testAuc = auc(testCorpus, model);
            double trainAccuracy = accuracy(trainCorpus, model);
            double testAccuracy = accuracy(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc
                    + " trainAccuracy=" + trainAccuracy + " testAccuracy=" + testAccuracy
                    + " trainTime=" + trainTime + " testTime=" + testTime);
            if(converage(oldModel, model)){
                //break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
        }
    }

    public static void train(List<LabeledData> corpus, double lambda) {
        int dim = corpus.get(0).data.dim;
        SVM svmCD = new SVM();
        DenseVector model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        svmCD.train(corpus, model, lambda);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: CoordinateDescent.SVM dim train_path lambda [trainRatio]");
        int dim = Integer.parseInt(argv[0]);
        String path = argv[1];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        double lambda = Double.parseDouble(argv[2]);
        if(argv.length >= 4){
            trainRatio = Double.parseDouble(argv[3]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(corpus, lambda);
    }
}
