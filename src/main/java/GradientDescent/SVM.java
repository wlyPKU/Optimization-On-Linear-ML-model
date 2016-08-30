package GradientDescent;

import Utils.*;
import math.DenseVector;
import java.util.*;

/**
 * Created by leleyu on 2016/6/30.
 */
public class SVM extends model.SVM{
  private double sgdOneEpoch(List<LabeledData> list, DenseVector model, double lr, double lambda) {
    int N_UPDATES = 0;
    int N_TOTAL = 0;
    double modelPenalty = -2 * lr * lambda * list.size();
    for (LabeledData labeledData : list) {
      N_TOTAL++;
      //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf Pg 3.
      /* model pennalty */
      //model.value[i] -= model.value[i] * 2 * lr * lambda / N;
      model.penaltySparse(labeledData.data, modelPenalty);
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

    DenseVector oldModel = new DenseVector(model.values.length);
    for (int i = 0; i < 300; i ++) {
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

      if(converage(oldModel, model)){
        break;
      }
      System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
    }
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

  public static void main(String[] argv) throws Exception {
    System.out.println("Usage: GradientDescent.SVM dim train_path lambda");
    int dim = Integer.parseInt(argv[0]);
    String path = argv[1];
    long startLoad = System.currentTimeMillis();
    List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
    long loadTime = System.currentTimeMillis() - startLoad;
    System.out.println("Loading corpus completed, takes " + loadTime + " ms");
    double lambda = Double.parseDouble(argv[2]);
    train(corpus, lambda);
  }
}
