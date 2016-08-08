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

    public void train(List<LabeledData> corpus, DenseVector model, double lambda) {
        Collections.shuffle(corpus);

        int size = corpus.size();
        int end = (int) (size * 0.5);
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
        for(int j = 0; j < alpha.length;j++){
            alpha[j] = 0;
        }
        double C = 1 / (2.0 * lambda);
        for (int i = 0; i < 100; i ++) {
            long startTrain = System.currentTimeMillis();
            //Coordinate Descent
            int j = 0;
            for (LabeledData labeledData : trainCorpus) {
                double G = model.dot(labeledData.data) * labeledData.label - 1;
                double alpha_old = alpha[j];
                alpha[j] = Math.min(Math.max(0, alpha[j] - G / Q[j]), C);
                int r = 0;
                for(Integer idx : labeledData.data.indices){
                    if(labeledData.data.values == null){
                        model.values[idx] += (alpha[j] - alpha_old) * labeledData.label;
                    }
                    else{
                        model.values[idx] += (alpha[j] - alpha_old) * labeledData.label * labeledData.data.values[r];
                        r++;
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
            System.out.println("Iter " + i + " loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc
                    + " trainAccuracy=" + trainAccuracy + " testAccuracy=" + testAccuracy
                    + " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }

    private static List<LabeledData> minhash(List<LabeledData> trainCorpus, int K, int dim, int b) {
        MinHash hash = new MinHash(K, dim, b);
        int hashedDim = hash.getHashedDim();
        List<LabeledData> hashedCorpus = new ArrayList<LabeledData>();
        for (LabeledData labeledData : trainCorpus) {
            int[] bits = hash.generateMinHashBits(labeledData.data);
            SparseVector vector = new SparseVector(hashedDim, bits);
            LabeledData hashedData = new LabeledData(vector, labeledData.label);
            hashedCorpus.add(hashedData);
        }
        return hashedCorpus;
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

    private static void trainWithMinHash(List<LabeledData> corpus, int K, int b, double lambda) {

        int dimension = corpus.get(0).data.dim;
        long startMinHash = System.currentTimeMillis();
        List<LabeledData> hashedCorpus = minhash(corpus, K, dimension, b);
        long minHashTime = System.currentTimeMillis() - startMinHash;

        dimension = hashedCorpus.get(0).data.dim;
        System.out.println("MinHash takes " + minHashTime + " ms" + " the dimension is " + dimension);

        corpus = hashedCorpus;
        SVM svmcd = new SVM();
        DenseVector model = new DenseVector(dimension);
        long start = System.currentTimeMillis();
        svmcd.train(corpus, model, lambda);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: CoordinateDescent.SVM dim train_path lamda [true|false] K b");
        int dim = Integer.parseInt(argv[0]);
        String path = argv[1];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        double lamda = Double.parseDouble(argv[2]);
        boolean minhash = Boolean.parseBoolean(argv[3]);
        if (minhash) {
            System.out.println("Training with minhash method.");
            int K = Integer.parseInt(argv[4]);
            int b = Integer.parseInt(argv[5]);
            trainWithMinHash(corpus, K, b, lamda);
        } else {
            train(corpus, lamda);
        }
    }


}
