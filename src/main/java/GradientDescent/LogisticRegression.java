package GradientDescent;

import Utils.*;
import math.DenseVector;
import java.util.*;

/**
 * Created by leleyu on 2016/7/1.
 */
public class LogisticRegression extends model.LogisticRegression{

  private void sgdOneEpoch(List<LabeledData> list, DenseVector modelOfU,
                          DenseVector modelOfV, double lr, double lambda) {
    for (LabeledData labeledData: list) {
      //Gradient=lambda+(1-1/(1+e^(-ywx))*y*xi
      //scala=(1-1/(1+e^(-ywx))*y
      double predictValue = modelOfU.dot(labeledData.data) - modelOfV.dot(labeledData.data);
      double tmpValue = 1.0 / (1.0 + Math.exp(labeledData.label * predictValue));
      double scala = tmpValue * labeledData.label;
      modelOfU.plusGradient(labeledData.data, + scala * lr);
      modelOfU.allPlusBy(- lr * lambda);
      modelOfU.positiveValueOrZero();
      modelOfV.plusGradient(labeledData.data, - scala * lr);
      modelOfV.allPlusBy(- lr * lambda);
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
      double loss = logLoss(trainCorpus, model, lambda);

      double trainAuc = auc(trainCorpus, model);
      double testAuc = auc(testCorpus, model);
      long testTime = System.currentTimeMillis() - startTest;
      System.out.println("Iter " + i + " loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc +
              " trainTime=" + trainTime + " testTime=" + testTime);
    }
  }

  public static void train(List<LabeledData> corpus, double lambda) {
    int dimension = corpus.get(0).data.dim;
    LogisticRegression lr = new LogisticRegression();
    //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
    DenseVector modelOfU = new DenseVector(dimension);
    DenseVector modelOfV = new DenseVector(dimension);
    long start = System.currentTimeMillis();
    lr.train(corpus, modelOfU, modelOfV, lambda);
    long cost = System.currentTimeMillis() - start;
    System.out.println(cost + " ms");
  }

  public static void main(String[] argv) throws Exception {
    System.out.println("Usage: GradientDescent.LogisticRegression FeatureDim train_path lambda");
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
