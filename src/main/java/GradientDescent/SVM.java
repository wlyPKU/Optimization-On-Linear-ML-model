package GradientDescent;

import Utils.LabeledData;
import Utils.MinHash;
import Utils.Utils;
import math.DenseVector;
import math.SparseVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by leleyu on 2016/6/30.
 */
public class SVM {

  public double SVMLoss(List<LabeledData> list, DenseVector model) {
    double loss = 0.0;
    for (LabeledData labeledData : list) {
      double dotProd = model.dot(labeledData.data);
      loss += Math.max(0, 1 - dotProd * labeledData.label);
    }
    return loss / list.size();
  }

  public double sgdOneEpoch(List<LabeledData> list, DenseVector model, double lr, double lamda) {
    int N_UPDATES = 0;
    int N_TOTAL = 0;

    for (LabeledData labeledData : list) {
      N_TOTAL++;
      //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf Pg 3.
      /* Model pennalty */
      model.plusBy(labeledData.data, -lr);

      double dotProd = model.dot(labeledData.data);
      if (1 - dotProd * labeledData.label > 0) {
        /* residual pennalty */
        model.plusBy(labeledData.data, lr * labeledData.label);
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

    for (int i = 0; i < 30; i ++) {
      //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
      double ratio = sgdOneEpoch(trainCorpus, model, 0.005, lambda);
      double loss = SVMLoss(trainCorpus, model);
      double accuracy = test(testCorpus, model);
      System.out.println("ratio = " + ratio + " loss = " + loss + " accuracy = " + accuracy);
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

  public static void train(List<LabeledData> corpus, double lamda) {
    int dim = corpus.get(0).data.dim;
    SVM svm = new SVM();
    DenseVector model = new DenseVector(dim);
    long start = System.currentTimeMillis();
    svm.train(corpus, model, lamda);

    long cost = System.currentTimeMillis() - start;
    System.out.println(cost + " ms");
  }

  public static void trainWithMinHash(List<LabeledData> corpus, int K, int b, double lamda) {

    int dim = corpus.get(0).data.dim;
    long startMinHash = System.currentTimeMillis();
    List<LabeledData> hashedCorpus = minhash(corpus, K, dim, b);
    long minHashTime = System.currentTimeMillis() - startMinHash;

    dim = hashedCorpus.get(0).data.dim;
    System.out.println("Utils.MinHash takes " + minHashTime + " ms" + " the dimension is " + dim);

    corpus = hashedCorpus;
    SVM svm = new SVM();
    DenseVector model = new DenseVector(dim);
    long start = System.currentTimeMillis();
    svm.train(corpus, model, lamda);

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
