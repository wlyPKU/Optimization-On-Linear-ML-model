package GradientDescent;

import Utils.*;
import it.unimi.dsi.fastutil.doubles.DoubleComparator;
import math.DenseVector;
import java.util.Collections;
import java.util.List;

/**
 * Created by leleyu on 2016/7/1.
 */
public class LogisticRegression {
  @SuppressWarnings("unused")
  public double grad(double pre, double y) {
    double z = pre * y;
    if (z > 18) {
      return y * Math.exp(-z);
    } else if (z < -18) {
      return y;
    } else {
      return y / (1.0 + Math.exp(z));
    }
  }

  private double logLoss(List<LabeledData> list, DenseVector model, double lambda) {
    double loss = 0.0;
    for (LabeledData labeledData: list) {
      double p = model.dot(labeledData.data);
      double z = p * labeledData.label;
      if (z > 18) {
        loss += Math.exp(-z);
      } else if (z < -18) {
        loss += -z;
      } else {
        loss += Math.log(1 + Math.exp(-z));
      }
    }
    for(Double v : model.values){
      loss += lambda * (v > 0? v : -v);
    }
    return loss;
  }

  private void sgdOneEpoch(List<LabeledData> list, DenseVector modelOfU,
                          DenseVector modelOfV, double lr, double lambda) {
    for (LabeledData labeledData: list) {
      //Gradient=lamda+(1-1/(1+e^(-ywx))*y*xi
      //scala=(1-1/(1+e^(-ywx))*y
      double predictValue = modelOfU.dot(labeledData.data) - modelOfV.dot(labeledData.data);
      double tmpValue = 1.0 / (1.0 + Math.exp(labeledData.label * predictValue));
      double scala = tmpValue * labeledData.label;
      modelOfU.plusGradient(labeledData.data, + scala * lr);
      modelOfU.allPlusBy(- lr * lambda);
      modelOfU.positiveValueOrZero(labeledData.data);
      modelOfV.plusGradient(labeledData.data, - scala * lr);
      modelOfV.allPlusBy(- lr * lambda);
      modelOfV.positiveValueOrZero(labeledData.data);
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
    for (int i = 0; i < 300; i ++) {
      long startTrain = System.currentTimeMillis();
      //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
      sgdOneEpoch(trainCorpus, modelOfU, modelOfV, 0.005, lambda);
      long trainTime = System.currentTimeMillis() - startTrain;
      long startTest = System.currentTimeMillis();

      for(int j = 0; j < model.dim; j++){
        model.values[j] = modelOfU.values[j] - modelOfV.values[j];
      }
      double loss = logLoss(trainCorpus, model, lambda);

      double trainAuc = auc(trainCorpus, model);
      double testAuc = auc(testCorpus, model);
      long testTime = System.currentTimeMillis() - startTest;
      System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc +
              " trainTime=" + trainTime + " testTime=" + testTime);
    }
  }

  public static void train(List<LabeledData> corpus, double lambda) {
    int dim = corpus.get(0).data.dim;
    LogisticRegression lr = new LogisticRegression();
    //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
    DenseVector modelOfU = new DenseVector(dim);
    DenseVector modelOfV = new DenseVector(dim);
    long start = System.currentTimeMillis();
    lr.train(corpus, modelOfU, modelOfV, lambda);
    long cost = System.currentTimeMillis() - start;
    System.out.println(cost + " ms");
  }

  private static void trainWithMinHash(List<LabeledData> corpus,
                                      int K, int b, double lambda) {
    int dim = corpus.get(0).data.dim;
    long startMinHash = System.currentTimeMillis();
    List<LabeledData> hashedCorpus = SVM.minhash(corpus, K, dim, b);
    long minHashTime = System.currentTimeMillis() - startMinHash;
    dim = hashedCorpus.get(0).data.dim;
    corpus = hashedCorpus;
    System.out.println("Utils.MinHash takes " + minHashTime + " ms" + " the dimension is " + dim);

    LogisticRegression lr = new LogisticRegression();
    DenseVector modelOfU = new DenseVector(dim);
    DenseVector modelOfV = new DenseVector(dim);
    long start = System.currentTimeMillis();
    lr.train(corpus, modelOfU, modelOfV, lambda);
    long cost = System.currentTimeMillis() - start;
    System.out.println(cost + " ms");
  }

  public double test(List<LabeledData> list, DenseVector model) {
    int N_RIGHT = 0;
    int N_TOTAL = 0;
    for (LabeledData labeledData: list) {
      double z = model.dot(labeledData.data);
      double score = 1.0 / (1.0 + Math.exp(-z));
      if (score >= 0.5 && labeledData.label == 1)
        N_RIGHT ++;
      if (score < 0.5 && labeledData.label == -1)
        N_RIGHT ++;
      N_TOTAL ++;
    }
    return 1.0 * N_RIGHT / N_TOTAL;
  }

  private double auc(List<LabeledData> list, DenseVector model) {
    int length = list.size();
    double[] scores = new double[length];
    double[] labels = new double[length];

    int cnt = 0;
    for (LabeledData labeledData: list) {
      double z = model.dot(labeledData.data);
      double score = 1.0 / (1.0 + Math.exp(-z));

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

  public static void main(String[] argv) throws Exception {
    System.out.println("Usage: GradientDescent.LogisticRegression FeatureDim train_path lamda [true|false] K b");
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
