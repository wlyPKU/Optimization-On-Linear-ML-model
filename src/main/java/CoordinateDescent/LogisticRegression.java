package CoordinateDescent;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.SparseMap;
import math.DenseVector;
import Utils.LabeledData;
import Utils.Utils;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
//Ref: http://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf
public class LogisticRegression extends model.LogisticRegression{

    public void train(SparseMap[] features, List<LabeledData> labeledData,
                      DenseVector model, double lambda, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;

        for (int i = 0; i < 100; i ++) {
            long startTrain = System.currentTimeMillis();
            //Cyclic Feature
            for(int fIdx = 0; fIdx < featureDim; fIdx++){
                double secondOrderL = 0;
                //Sum X^T(jk)D(kk)X(kj)  k=1,2,3...,sampleSize
                ObjectIterator<Int2DoubleMap.Entry> iter =  features[fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double value = entry.getDoubleValue();
                    LabeledData l = labeledData.get(idx);
                    double predictValue = model.dot(l.data);
                    double Dii = (1 / (1 + Math.exp( -l.label * predictValue)))
                            * (1 - (1 / (1 + Math.exp( -l.label * predictValue))));
                    secondOrderL += Dii * Math.pow(value, 2);
                }
                secondOrderL *= 1 / lambda;

                //First Order L:
                double firstOrderL = 0;
                iter =  features[fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double value = entry.getDoubleValue();
                    LabeledData l = labeledData.get(idx);
                    double predictValue = model.dot(l.data);
                    double tao = 1 / (1 + Math.exp( -l.label * predictValue));
                    firstOrderL += l.label * value * (tao - 1);
                }

                firstOrderL *= 1.0 / lambda;
                double d;
                if(firstOrderL + 1 <= secondOrderL * model.values[fIdx]){
                    d = - (firstOrderL + 1) / secondOrderL;
                }else if(firstOrderL - 1 >= secondOrderL * model.values[fIdx]){
                    d = - (firstOrderL - 1) / secondOrderL;
                }else{
                    d = 0;
                }
                double TOLERANCE = 1e-5;
                if(Math.abs(d) < TOLERANCE){
                    continue;
                }
                boolean findSolution = false;
                double BETA = 0.2;
                double step = 1;
                double SIGMA = 0.8;
                while(!findSolution){
                    DenseVector plusModel = new DenseVector(model);
                    plusModel.values[fIdx] +=  d * step;
                    double deltaFunctionValue = lossFunctionValue(trainCorpus, plusModel, lambda)
                            -lossFunctionValue(trainCorpus, model, lambda);
                    double compareValue = step * SIGMA *(firstOrderL * d+ getVectorLength(plusModel) - getVectorLength(model));
                    if(compareValue >= deltaFunctionValue){
                        findSolution = true;
                        model.values[fIdx] += step * d;
                    }
                    step *= BETA;
                }
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = logLoss(trainCorpus, model, lambda);
            double trainAuc = auc(trainCorpus, model);
            double testAuc = auc(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc +
                    " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }


    public static void train(SparseMap[] corpus, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        int dimension = corpus.length;
        LogisticRegression lrCD = new LogisticRegression();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector model = new DenseVector(dimension);
        for(int i = 0; i < dimension; i++){
            model.values[i] = 0;
        }
        long start = System.currentTimeMillis();
        lrCD.train(corpus, labeledData, model, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: CoordinateDescent.LogisticRegression FeatureDim SampleDim train_path lambda trainRatio");
        int featureDim = Integer.parseInt(argv[0]);
        int sampleDim = Integer.parseInt(argv[1]);
        String path = argv[2];
        double lambda = Double.parseDouble(argv[3]);
        double trainRatio = 0.5;
        if(argv.length >= 5){
            trainRatio = Double.parseDouble(argv[4]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long startLoad = System.currentTimeMillis();
        SparseMap[] features = Utils.LoadLibSVMByFeature(path, featureDim, sampleDim, trainRatio);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(features, labeledData, lambda, trainRatio);
    }
}