package GradientDescent;

import Utils.*;
import it.unimi.dsi.fastutil.doubles.DoubleComparator;
import math.DenseVector;
import math.SparseVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by leleyu on 2016/6/30.
 */
public class SVM {
  public double accuracy(List<LabeledData> labeledData, DenseVector model){
    double result = 0;
    for(LabeledData l : labeledData) {
      double predictValue = model.dot(l.data);
      if (predictValue * l.label >= 0){
        result ++;
      }
    }
    return result / labeledData.size();
  }
  private double auc(List<LabeledData> list, DenseVector model) {
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
  private double SVMLoss(List<LabeledData> list, DenseVector model, double lambda) {
    double loss = 0.0;
    for (LabeledData labeledData : list) {
      double dotProd = model.dot(labeledData.data);
      loss += Math.max(0, 1 - dotProd * labeledData.label);
    }
    for(Double v: model.values){
      loss += lambda * v * v;
    }
    return loss;
  }

  private double sgdOneEpoch(List<LabeledData> list, DenseVector model, double lr, double lambda) {
    int N_UPDATES = 0;
    int N_TOTAL = 0;

    for (LabeledData labeledData : list) {
      N_TOTAL++;
      //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf Pg 3.
      /* Model pennalty */
      for(int i = 0; i < model.values.length; i++){
        model.values[i] -= 2 * lr * lambda * model.values[i];
      }
      double dotProd = model.dot(labeledData.data);
      if (1 - dotProd * labeledData.label > 0) {
        /* residual pennalty */
        model.plusGradient(labeledData.data, lr * labeledData.label);
        N_UPDATES++;
      }
    }

    return 1.0 * N_UPDATES / N_TOTAL;
  }

  public void train(List<LabeledData> corpus, DenseVector model, double lambda) {
    Collections.shuffle(corpus);

    int size = corpus.size();
    int end = (int) (size * 0.5);
    List<LabeledData> trainCorpus = corpus.subList(0, end);
    List<LabeledData> testCorpus = corpus.subList(end, size);

    for (int i = 0; i < 100; i ++) {
      long startTrain = System.currentTimeMillis();
      //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
      double ratio = sgdOneEpoch(trainCorpus, model, 0.005, lambda);
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
    SVM svm = new SVM();
    DenseVector model = new DenseVector(dim);
    long start = System.currentTimeMillis();
    svm.train(corpus, model, lambda);

    long cost = System.currentTimeMillis() - start;
    System.out.println(cost + " ms");
  }

  private static void trainWithMinHash(List<LabeledData> corpus, int K, int b, double lambda) {

    int dim = corpus.get(0).data.dim;
    long startMinHash = System.currentTimeMillis();
    List<LabeledData> hashedCorpus = minhash(corpus, K, dim, b);
    long minHashTime = System.currentTimeMillis() - startMinHash;

    dim = hashedCorpus.get(0).data.dim;
    System.out.println("MinHash takes " + minHashTime + " ms" + " the dimension is " + dim);

    corpus = hashedCorpus;
    SVM svm = new SVM();
    DenseVector model = new DenseVector(dim);
    long start = System.currentTimeMillis();
    svm.train(corpus, model, lambda);

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
