package CoordinateDescent;

import it.unimi.dsi.fastutil.doubles.DoubleComparator;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.SparseMap;
import math.DenseVector;
import Utils.LabeledData;
import Utils.Utils;
import Utils.Sort;
import java.util.List;
import java.util.Map;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
//Ref: http://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf
public class LogisticRegression {
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

    private double lossFunctionValue(List<LabeledData> labeledData,
                                DenseVector model, double lambda){
        double result = 0;
        for(LabeledData l: labeledData){
            double exp = Math.exp(- l.label * model.dot(l.data));
            double loss = Math.log(1 + exp);
            result += loss;
        }
        for(Double w : model.values){
            result += lambda * Math.abs(w);
        }
        return result;
    }

    private double getVectorLength(DenseVector model){
        double length = 0;
        for(Double w: model.values){
            length += w * w;
        }
        return Math.sqrt(length);
    }
    public void train(SparseMap[] features, List<LabeledData> labeledData,
                      DenseVector model, double lambda, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;

        for (int i = 0; i < 300; i ++) {
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
//      double accuracy = test(testCorpus, model);

            double trainAuc = auc(trainCorpus, model);
            double testAuc = auc(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc +
                    " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }


    public static void train(SparseMap[] corpus, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        int dim = corpus.length;
        LogisticRegression lrCD = new LogisticRegression();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector model = new DenseVector(dim);
        for(int i = 0; i < dim; i++){
            model.values[i] = 0;
        }
        long start = System.currentTimeMillis();
        lrCD.train(corpus, labeledData, model, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    private double auc(List<LabeledData> list, DenseVector model) {
        int length = list.size();
        System.out.println(length);
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
        System.out.println("Usage: CoordinateDescent.LogisticRegression FeatureDim SampleDim train_path lamda trainRatio");
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
        //TODO Need to think how to min hash numeric variables
        train(features, labeledData, lambda, trainRatio);
    }
}