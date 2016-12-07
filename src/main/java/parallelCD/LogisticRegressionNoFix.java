package parallelCD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;
import math.SparseMap;
import math.SparseVector;

import java.lang.management.ManagementFactory;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
//Ref: http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Pg. 7,14,18

//  model       每个线程共享
//  predictValue    每个线程共享
//  可能会发生冲突
public class LogisticRegressionNoFix extends model.LogisticRegression{

    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;

    private static SparseVector[] features;

    private static DenseVector modelOfU;
    private static DenseVector modelOfV;
    private static DenseVector model;
    private static double predictValue[];

    private static double featureSquare[];

    private static List<LabeledData> trainCorpus;

    private class updateWPositive implements Runnable
    {
        int from, to;
        private updateWPositive(int from, int to){
            this.from = from;
            this.to = to;

        }
        public void run() {
            double C = Double.MAX_VALUE;
            if(lambda != 0){
                C = 1.0 / lambda;
            }
            int[] indices;
            int idx;
            double firstOrderL, oldValue, xj;
            double[] values;
            for(int fIdx = from; fIdx < to; fIdx++){
                indices = features[fIdx].indices;
                values = features[fIdx].values;
                if(featureSquare[fIdx] != 0) {
                    //First Order L:
                    firstOrderL = 0;
                    oldValue = modelOfV.values[fIdx];
                    for (int i = 0; i < indices.length; i++) {
                        idx = indices[i];
                        xj = values[i];
                        LabeledData l = trainCorpus.get(idx);
                        double tao = 1 / (1 + Math.exp(-l.label * predictValue[idx]));
                        firstOrderL += l.label * xj * (tao - 1);
                    }
                    firstOrderL *= C;
                    //误差的来源,0.25.
                    double Uj = 0.25 * C * featureSquare[fIdx];
                    double updateValue = (1 + firstOrderL) / Uj;
                    if (updateValue > modelOfU.values[fIdx]) {
                        modelOfU.values[fIdx] = 0;
                    } else {
                        modelOfU.values[fIdx] -= updateValue;
                    }
                    if(modelOfU.values[fIdx] - oldValue != 0) {
                        //Update predictValue
                        for (int i = 0; i < indices.length; i++) {
                            idx = indices[i];
                            predictValue[idx] += values[i] * (modelOfU.values[fIdx] - oldValue);
                        }
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
            double C = Double.MAX_VALUE;
            if(lambda != 0){
                C = 1.0 / lambda;
            }
            int[] indices;
            int idx;
            double firstOrderL, oldValue, xj;
            double[] values;
            for(int fIdx = from; fIdx < to; fIdx++){
                indices = features[fIdx].indices;
                values = features[fIdx].values;
                if(featureSquare[fIdx] != 0) {
                    //First Order L:
                    firstOrderL = 0;
                    oldValue = modelOfV.values[fIdx];
                    for (int i = 0; i < indices.length; i++) {
                        idx = indices[i];
                        xj = values[i];
                        LabeledData l = trainCorpus.get(idx);
                        double tao = 1 / (1 + Math.exp(-l.label * predictValue[idx]));
                        firstOrderL += l.label * xj * (tao - 1);
                    }
                    firstOrderL *= C;
                    double Uj = 0.25 * C * featureSquare[fIdx];
                    double updateValue = (1 - firstOrderL) / Uj;
                    if (updateValue > modelOfV.values[fIdx]) {
                        modelOfV.values[fIdx] = 0;
                    } else {
                        modelOfV.values[fIdx] -= updateValue;
                    }
                    if(modelOfV.values[fIdx] - oldValue != 0){
                        for (int i = 0; i < indices.length; i++) {
                            predictValue[indices[i]] -= values[i] * (modelOfV.values[fIdx] - oldValue);
                        }
                    }
                }
            }
        }
    }

    public class adjustPredictValueThread implements Runnable
    {
        int from, to;
        adjustPredictValueThread(int from, int to){
            this.from = from;
            this.to = to;
        }
        public void run() {
            for(int j = from; j < to; j++){
                LabeledData l = trainCorpus.get(j);
                predictValue[j] = model.dot(l.data);
            }
        }
    }

    private void adjustPredictValue(){
        ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
        for (int threadID = 0; threadID < threadNum; threadID++) {
            int from = trainCorpus.size() * threadID / threadNum;
            int to = trainCorpus.size() * (threadID + 1) / threadNum;
            threadPool.execute(new adjustPredictValueThread(from, to));
        }
        threadPool.shutdown();
        while (!threadPool.isTerminated()) {
            try {
                while (!threadPool.awaitTermination(1, TimeUnit.MILLISECONDS)) {
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private void trainCore(List<LabeledData> labeledData) {
        double startCompute = System.currentTimeMillis();
        //shuffle(labeledData);
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        trainCorpus = labeledData.subList(0, testBegin + 1);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        predictValue = new double[trainCorpus.size()];

        model = new DenseVector(featureDimension);
        DenseVector oldModel = new DenseVector(featureDimension);

        long totalBegin = System.currentTimeMillis();

        long totalIterationTime = 0;
        //Added according to the http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Page 18.
        featureSquare = new double[featureDimension];
        Arrays.fill(featureSquare, 0);
        for(int idx = 0; idx < featureDimension; idx++){
            for (double xj: features[idx].values) {
                featureSquare[idx] += xj * xj;
            }
        }
        System.out.println("[Prepare]Pre-computation takes " + (System.currentTimeMillis() - startCompute) + " ms totally");
        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model, lambda);
            long startTrain = System.currentTimeMillis();
            //Update w+
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
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
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
                newThreadPool.execute(new updateWNegative(from, to));
            }
            newThreadPool.shutdown();
            while (!newThreadPool.isTerminated()) {
                try {
                    newThreadPool.awaitTermination(1, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            for(int fIdx = 0; fIdx < featureDimension; fIdx ++){
                model.values[fIdx] = modelOfU.values[fIdx] - modelOfV.values[fIdx];
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("[Information]trainTime " + trainTime + " ");
            totalIterationTime += trainTime;
            System.out.println("[Information]totalTrainTime " + totalIterationTime);
            System.out.println("[Information]totalTime " + (System.currentTimeMillis() - totalBegin) );
            System.out.println("[Information]HeapUsed " + ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed()
                    / 1024 / 1024 + "M");
            System.out.println("[Information]MemoryUsed " + (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
                    / 1024 / 1024 + "M");

            if(modelType == 1) {
                if (totalIterationTime > maxTimeLimit) {
                    break;
                }
            }else if(modelType == 0){
                if(i > maxIteration){
                    break;
                }
            }
            if(converge(oldModel, model, trainCorpus, lambda)){
                if (modelType == 2)
                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDimension);
            if(diverge){
                System.out.println("[Warning]Diverge happens!");
                break;
            }
        }
    }


    public static void train(List<LabeledData> labeledData) {
        LogisticRegressionNoFix lrSCD = new LogisticRegressionNoFix();
        //http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf 3197-3200+
        modelOfU = new DenseVector(featureDimension);
        modelOfV = new DenseVector(featureDimension);
        Arrays.fill(modelOfU.values, 0);
        Arrays.fill(modelOfV.values, 0);
        long start = System.currentTimeMillis();
        lrSCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }

    private void shuffle(List<LabeledData> labeledData) {
        Collections.shuffle(labeledData);
        SparseMap[] tmpFeatures = Utils.LoadLibSVMFromLabeledData(labeledData, featureDimension, trainRatio);
        features = Utils.generateSpareVector(tmpFeatures);
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.LogisticRegression threadNum FeatureDim train_path lamda trainRatio");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        if(lambda <= 0){
            System.out.println("Please input a correct lambda (>0)");
            System.exit(2);
        }
        for(int i = 0; i < argv.length - 1; i++){
            if(argv[i].equals("Model")){
                //0: maxIteration  1: maxTime 2: earlyStop
                modelType = Integer.parseInt(argv[i + 1]);
            }
            if(argv[i].equals("TimeLimit")){
                maxTimeLimit = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("StopDelta")){
                stopDelta = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("MaxIteration")){
                maxIteration = Integer.parseInt(argv[i + 1]);
            }
            if(argv[i].equals("TrainRatio")){
                trainRatio = Double.parseDouble(argv[i+1]);
                if(trainRatio >= 1 || trainRatio <= 0){
                    System.out.println("Error Train Ratio!");
                    System.exit(1);
                }
            }
        }
        System.out.println("[Parameter]ThreadNum " + threadNum);
        System.out.println("[Parameter]StopDelta " + stopDelta);
        System.out.println("[Parameter]FeatureDimension " + featureDimension);
        System.out.println("[Parameter]File Path " + path);
        System.out.println("[Parameter]Lambda " + lambda);
        System.out.println("[Parameter]TrainRatio " + trainRatio);
        System.out.println("[Parameter]TimeLimit " + maxTimeLimit);
        System.out.println("[Parameter]ModelType " + modelType);
        System.out.println("[Parameter]Iteration Limit " + maxIteration);
        System.out.println("------------------------------------");
        long startLoad = System.currentTimeMillis();
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}