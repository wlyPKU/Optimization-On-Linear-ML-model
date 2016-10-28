package parallelCD;

import Utils.LabeledData;
import Utils.Utils;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.DenseVector;
import math.SparseMap;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
// Block Coordinate Descent
//  model       每个线程独立(虽然更改同样一份model,predictValue,因此每个线程独立)
//  predictValue    每个线程独立
//Ref: http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Pg. 7,14,18
public class LogisticRegressionBCD extends model.LogisticRegression{
    private static long start;

    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;
    private static int sampleDimension;
    private static SparseMap[] features;
    private static DenseVector modelOfU;
    private static DenseVector modelOfV;
    private static double predictValue[][];


    private static List<LabeledData> trainCorpus;

    private class updateWPositive implements Runnable
    {
        int from, to;
        int threadID;
        private updateWPositive(int threadID, int from, int to){
            this.threadID = threadID;
            this.from = from;
            this.to = to;

        }
        public void run() {
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
                        double tao = 1 / (1 + Math.exp( -l.label * predictValue[threadID][idx]));
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
                        predictValue[threadID][idx] += value * (modelOfU.values[fIdx] - oldValue);
                    }
                }
            }
        }
    }

    private class updateWNegative implements Runnable
    {
        int threadID;
        int from, to;
        private updateWNegative(int threadID, int from, int to){
            this.threadID = threadID;
            this.from = from;
            this.to = to;

        }
        public void run() {
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
                        double tao = 1 / (1 + Math.exp(-l.label * predictValue[threadID][idx]));
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
                        predictValue[threadID][idx] -= value * (modelOfV.values[fIdx] - oldValue);
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

        predictValue = new double[threadNum][trainCorpus.size()];

        DenseVector model = new DenseVector(featureDimension);
        DenseVector oldModel = new DenseVector(featureDimension);

        long totalBegin = System.currentTimeMillis();


        for (int i = 0; ; i ++) {
            for(int idx = 0; idx < trainCorpus.size(); idx++){
                LabeledData l = trainCorpus.get(idx);
                double predictV = modelOfU.dot(l.data) - modelOfV.dot(l.data);
                for(int j = 0; j < threadNum; j++) {
                    predictValue[j][idx] = predictV;
                }
            }
            long startTrain = System.currentTimeMillis();
            //Update w+
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
                threadPool.execute(new updateWPositive(threadID, from, to));
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
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
                newThreadPool.execute(new updateWNegative(threadID, from, to));
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

            for(int fIdx = 0; fIdx < featureDimension; fIdx ++){
                model.values[fIdx] = modelOfU.values[fIdx] - modelOfV.values[fIdx];
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime " + trainTime + " ");

            testAndSummary(trainCorpus, testCorpus, model, lambda);



            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );
            if(converge(oldModel, model)){
                if(earlyStop)

                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDimension);

            long nowCost = System.currentTimeMillis() - start;
            if(nowCost > 300000) {
                break;
                //break;
            }

        }
    }


    public static void train(List<LabeledData> labeledData) {
        LogisticRegressionBCD lrSCD = new LogisticRegressionBCD();
        //http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf 3197-3200+
        modelOfU = new DenseVector(featureDimension);
        modelOfV = new DenseVector(featureDimension);
        Arrays.fill(modelOfU.values, 0);
        Arrays.fill(modelOfV.values, 0);
        start = System.currentTimeMillis();
        lrSCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.LogisticRegressionBCD threadNum FeatureDim SampleDim train_path lamda trainRatio");
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        sampleDimension = Integer.parseInt(argv[2]);
        String path = argv[3];
        lambda = Double.parseDouble(argv[4]);
        if(lambda <= 0){
            System.out.println("Please input a correct lambda (>0)");
            System.exit(2);
        }

        for(int i = 0; i < argv.length - 1; i++){
            if(argv[i].equals("EarlyStop")){
                earlyStop = Boolean.parseBoolean(argv[i + 1]);
            }
            if(argv[i].equals("TimeLimit")){
                maxTimeLimit = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("TrainRatio")){
                trainRatio = Double.parseDouble(argv[4]);
                if(trainRatio >= 1 || trainRatio <= 0){
                    System.out.println("Error Train Ratio!");
                    System.exit(1);
                }            }
        }
        System.out.println("ThreadNum " + threadNum);
        System.out.println("FeatureDimension " + featureDimension);
        System.out.println("SampleDimension " + sampleDimension);
        System.out.println("File Path " + path);
        System.out.println("Lambda " + lambda);
        System.out.println("TrainRatio " + trainRatio);
        System.out.println("TimeLimit " + maxTimeLimit);
        System.out.println("EarlyStop " + earlyStop);
        long startLoad = System.currentTimeMillis();
        features = Utils.LoadLibSVMByFeature(path, featureDimension, sampleDimension, trainRatio);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}