package parallelCD;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.*;
import Utils.*;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
//Ref: http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Pg. 7,14,18
public class LogisticRegressionSCD extends model.LogisticRegression{

    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static SparseMap[] features;
    private static DenseVector modelOfU;
    private static DenseVector modelOfV;
    private static double predictValue[];

    private static List<LabeledData> trainCorpus;

    private class updateWPositive implements Runnable
    {
        int from, to;
        private updateWPositive(int from, int to){
            this.from = from;
            this.to = to;

        }
        public void run() {
            localTrain();
        }
        private void localTrain() {
            for(int fIdx = from; fIdx < to; fIdx++){
                //First Order L:
                double firstOrderL = 0;
                double oldValue = modelOfU.values[fIdx];
                ObjectIterator<Int2DoubleMap.Entry> iter =  features[fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < trainCorpus.size()) {
                        double xj = entry.getDoubleValue();
                        LabeledData l = trainCorpus.get(idx);
                        double tao = 1 / (1 + Math.exp( -l.label * predictValue[idx]));
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
                    if(idx < trainCorpus.size()) {
                        double value = entry.getDoubleValue();
                        predictValue[idx] += value * (modelOfU.values[fIdx] - oldValue);
                    }
                }
            }
        }
    }

    private class updateWNegative implements Runnable
    {
        int from, to;
        private updateWNegative(int from, int to){
            this.from = from;
            this.to = to;

        }
        public void run() {
            localTrain();
        }
        private void localTrain() {
            for(int fIdx = from; fIdx < to; fIdx++){
                //First Order L:
                double firstOrderL = 0;
                double oldValue = modelOfV.values[fIdx];
                ObjectIterator<Int2DoubleMap.Entry> iter =  features[fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < trainCorpus.size()) {
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
                    if(idx < trainCorpus.size()) {
                        double value = entry.getDoubleValue();
                        predictValue[idx] -= value * (modelOfV.values[fIdx] - oldValue);
                    }
                }
            }
        }
    }


    private void trainCore(List<LabeledData> labeledData) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;

        predictValue = new double[trainCorpus.size()];

        DenseVector model = new DenseVector(featureDim);
        DenseVector oldModel = new DenseVector(featureDim);

        for (int i = 0; i < 100; i ++) {
            for(int idx = 0; idx < trainCorpus.size(); idx++){
                LabeledData l = trainCorpus.get(idx);
                predictValue[idx] = modelOfU.dot(l.data) - modelOfV.dot(l.data);
            }
            long startTrain = System.currentTimeMillis();
            //Update w+
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDim * threadID / threadNum;
                int to = featureDim * (threadID + 1) / threadNum;
                threadPool.execute(new updateWPositive(from, to));
            }
            threadPool.shutdown();
            while (!threadPool.isTerminated()) {
                try {
                    threadPool.awaitTermination(1, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    System.out.println("Waiting.");
                    e.printStackTrace();
                }
            }
            //Update w-
            ExecutorService newThreadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDim * threadID / threadNum;
                int to = featureDim * (threadID + 1) / threadNum;
                newThreadPool.execute(new updateWNegative(from, to));
            }
            newThreadPool.shutdown();
            while (!newThreadPool.isTerminated()) {
                try {
                    newThreadPool.awaitTermination(1, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    System.out.println("Waiting.");
                    e.printStackTrace();
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


    public static void train(List<LabeledData> labeledData) {
        int dimension = features.length;
        LogisticRegressionSCD lrSCD = new LogisticRegressionSCD();
        //http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf 3197-3200+
        modelOfU = new DenseVector(dimension);
        modelOfV = new DenseVector(dimension);
        Arrays.fill(modelOfU.values, 0);
        Arrays.fill(modelOfV.values, 0);
        long start = System.currentTimeMillis();
        lrSCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.LogisticRegressionSCD threadNum FeatureDim SampleDim train_path lamda trainRatio");
        threadNum = Integer.parseInt(argv[0]);
        int featureDim = Integer.parseInt(argv[1]);
        int sampleDim = Integer.parseInt(argv[2]);
        String path = argv[3];
        lambda = Double.parseDouble(argv[4]);
        if(lambda <= 0){
            System.out.println("Please input a correct lambda (>0)");
            System.exit(2);
        }

        if(argv.length >= 6){
            trainRatio = Double.parseDouble(argv[5]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long startLoad = System.currentTimeMillis();
        features = Utils.LoadLibSVMByFeature(path, featureDim, sampleDim, trainRatio);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}