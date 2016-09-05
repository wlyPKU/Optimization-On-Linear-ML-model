package CoordinateDescent;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.*;
import Utils.*;

import java.util.Arrays;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
//Ref: http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Pg. 7,14,18
public class LogisticRegressionSCD extends model.LogisticRegression{
    public void train(SparseMap[] features, List<LabeledData> labeledData,
                      DenseVector modelOfU, DenseVector modelOfV, double lambda, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;

        double predictValue[] = new double[trainCorpus.size()];

        DenseVector model = new DenseVector(featureDim);
        DenseVector oldModel = new DenseVector(featureDim);

        for (int i = 0; i < 100; i ++) {
            for(int idx = 0; idx < trainCorpus.size(); idx++){
                LabeledData l = trainCorpus.get(idx);
                predictValue[idx] = modelOfU.dot(l.data) - modelOfV.dot(l.data);
            }
            long startTrain = System.currentTimeMillis();
            //Update w+
            for(int fIdx = 0; fIdx < featureDim; fIdx++){
                //First Order L:
                double firstOrderL = 0;
                double oldValue = modelOfU.values[fIdx];
                ObjectIterator<Int2DoubleMap.Entry> iter =  features[fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < testBegin) {
                        double xj = entry.getDoubleValue();
                        LabeledData l = trainCorpus.get(idx);
                        double tao = 1 / (1 + Math.exp(-l.label * predictValue[idx]));
                        firstOrderL += l.label * xj * (tao - 1);
                    }
                }

                double Uj = 0.25 * 1 / lambda * trainCorpus.size();
                double updateValue = (1 + firstOrderL) / Uj;
                if(updateValue > modelOfU.values[fIdx]){
                    modelOfU.values[fIdx] = 0;
                }else{
                    modelOfU.values[fIdx] -= updateValue;
                }
                //Update predictValue
                iter =  features[fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < testBegin) {
                        double value = entry.getDoubleValue();
                        predictValue[idx] += value * (modelOfU.values[fIdx] - oldValue);
                    }
                }
            }
            //Update w-
            for(int fIdx = 0; fIdx < featureDim; fIdx++){
                //First Order L:
                double firstOrderL = 0;
                double oldValue = modelOfV.values[fIdx];
                ObjectIterator<Int2DoubleMap.Entry> iter =  features[fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < testBegin) {
                        double xj = entry.getDoubleValue();
                        LabeledData l = trainCorpus.get(idx);
                        double tao = 1 / (1 + Math.exp(-l.label * predictValue[idx]));
                        firstOrderL += l.label * xj * (tao - 1);
                    }
                }

                double Uj = 0.25 * 1 / lambda * trainCorpus.size();
                double updateValue = (1 - firstOrderL) / Uj;
                if(updateValue > modelOfV.values[fIdx]){
                    modelOfV.values[fIdx] = 0;
                }else{
                    modelOfV.values[fIdx] -= updateValue;
                }

                iter =  features[fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < testBegin) {
                        double value = entry.getDoubleValue();
                        predictValue[idx] -= value * (modelOfV.values[fIdx] - oldValue);
                    }
                }
            }
            for(int fIdx = 0; fIdx < featureDim; fIdx ++){
                model.values[fIdx] = modelOfU.values[fIdx] - modelOfV.values[fIdx];
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = logLoss(trainCorpus, model, lambda);

            double trainAuc = auc(trainCorpus, model);
            double testAuc = auc(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc +
                    " trainTime=" + trainTime + " testTime=" + testTime);

            if(converge(oldModel, model)){
                //break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDim);
        }
    }


    public static void train(SparseMap[] corpus, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        int dimension = corpus.length;
        LogisticRegressionSCD lrSCD = new LogisticRegressionSCD();
        //http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf 3197-3200+
        DenseVector modelOfU = new DenseVector(dimension);
        DenseVector modelOfV = new DenseVector(dimension);
        Arrays.fill(modelOfU.values, 0);
        Arrays.fill(modelOfV.values, 0);
        long start = System.currentTimeMillis();
        lrSCD.train(corpus, labeledData, modelOfU, modelOfV, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: CoordinateDescent.LogisticRegressionSCD FeatureDim SampleDim train_path lamda trainRatio");
        int featureDim = Integer.parseInt(argv[0]);
        int sampleDim = Integer.parseInt(argv[1]);
        String path = argv[2];
        double lambda = Double.parseDouble(argv[3]);
        if(lambda <= 0){
            System.out.println("Please input a correct lambda (>0)");
            System.exit(2);
        }
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