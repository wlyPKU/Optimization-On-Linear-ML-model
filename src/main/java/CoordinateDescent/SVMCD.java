package CoordinateDescent;

import math.DenseVector;
import math.SparseVector;
import Utils.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by leleyu on 2016/6/30.
 */
//http://www.tuicool.com/m/articles/RRZvYb
//https://github.com/acharuva/svm_cd/blob/master/svm_cd.py
public class SVMCD {

    public double SVMLoss(List<LabeledData> list, DenseVector model) {
        double loss = 0.0;
        for (LabeledData labeledData : list) {
            double dotProd = model.dot(labeledData.data);
            loss += Math.max(0, 1 - dotProd * labeledData.label);
        }
        return loss / list.size();
    }

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
        double C = 1 / (2 * lambda);
        for (int i = 0; i < 30; i ++) {
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
            }
            double loss = SVMLoss(trainCorpus, model);
            double accuracy = test(testCorpus, model);
            System.out.println( "loss = " + loss + " accuracy = " + accuracy);
        }
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

    public static List<LabeledData> minhash(List<LabeledData> trainCorpus, int K, int dim, int b) {
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
        SVMCD svmcd = new SVMCD();
        DenseVector model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        svmcd.train(corpus, model, lambda);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void trainWithMinHash(List<LabeledData> corpus, int K, int b, double lambda) {

        int dim = corpus.get(0).data.dim;
        long startMinHash = System.currentTimeMillis();
        List<LabeledData> hashedCorpus = minhash(corpus, K, dim, b);
        long minHashTime = System.currentTimeMillis() - startMinHash;

        dim = hashedCorpus.get(0).data.dim;
        System.out.println("Utils.MinHash takes " + minHashTime + " ms" + " the dimension is " + dim);

        corpus = hashedCorpus;
        SVMCD svmcd = new SVMCD();
        DenseVector model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        svmcd.train(corpus, model, lambda);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: GradientDescent.SVM dim train_path lamda [true|false] K b");
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
