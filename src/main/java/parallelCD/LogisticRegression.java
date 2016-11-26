package parallelCD;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.*;
import Utils.*;

import java.lang.management.ManagementFactory;
import java.text.SimpleDateFormat;
import java.util.*;
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
public class LogisticRegression extends model.LogisticRegression{
    private static long start;

    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;
    private static SparseMap[] tmpFeatures;
    private static SparseMap[] features;
    private static DenseVector modelOfU;
    private static DenseVector modelOfV;
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
                firstOrderL *= C;
                //误差的来源,0.25.
                double Uj = 0.25 * C * featureSquare[fIdx];
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
            double C = Double.MAX_VALUE;
            if(lambda != 0){
                C = 1.0 / lambda;
            }
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
                firstOrderL *= C;
                double Uj = 0.25 * C * featureSquare[fIdx];
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
        shuffle(labeledData, tmpFeatures);
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        predictValue = new double[trainCorpus.size()];

        DenseVector model = new DenseVector(featureDimension);
        DenseVector oldModel = new DenseVector(featureDimension);

        long totalBegin = System.currentTimeMillis();

        long totalIterationTime = 0;
        //Added according to the http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Page 18.
        featureSquare = new double[featureDimension];
        Arrays.fill(featureSquare, 0);
        for(int idx = 0; idx < featureDimension; idx++){
            ObjectIterator<Int2DoubleMap.Entry> iter =  features[idx].map.int2DoubleEntrySet().iterator();
            while (iter.hasNext()) {
                Int2DoubleMap.Entry entry = iter.next();
                int labelIdx = entry.getIntKey();
                double featureValue = entry.getDoubleValue();
                if(labelIdx < trainCorpus.size()) {
                    featureSquare[idx] += featureValue * featureValue;
                }
            }
        }
        for (int i = 0; ; i ++) {
            System.out.println("Iteration " + i + " ---------------");
            testAndSummary(trainCorpus, testCorpus, model, lambda);

            for(int idx = 0; idx < trainCorpus.size(); idx++){
                LabeledData l = trainCorpus.get(idx);
                predictValue[idx] = modelOfU.dot(l.data) - modelOfV.dot(l.data);
            }
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
            if(converge(oldModel, model)){
                if (modelType == 2)
                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDimension);

        }
    }


    public static void train(List<LabeledData> labeledData) {
        LogisticRegression lrSCD = new LogisticRegression();
        //http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf 3197-3200+
        modelOfU = new DenseVector(featureDimension);
        modelOfV = new DenseVector(featureDimension);
        Arrays.fill(modelOfU.values, 0);
        Arrays.fill(modelOfV.values, 0);
        start = System.currentTimeMillis();
        lrSCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }

    private void shuffle(List<LabeledData> labeledData, SparseMap[] tmpFeatures) {
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < labeledData.size(); i++) {
            list.add(i);
        }
        Collections.shuffle(list);
        List<LabeledData> tmpSet = new ArrayList<LabeledData>();
        for (int i = 0; i < labeledData.size(); i++) {
            tmpSet.add(labeledData.get(list.get(i)));
        }
        labeledData.clear();
        for (int i = 0; i < tmpSet.size(); i++) {
            labeledData.add(tmpSet.get(i));
        }
        features = new SparseMap[featureDimension + 1];
        for (int i = 0; i <= featureDimension; i++) {
            features[i] = new SparseMap();
        }
        int map[] = new int[labeledData.size()];
        for(int i = 0; i < map.length; i++){
            map[i] = list.indexOf(i);
        }
        for (int i = 0; i < features.length; i++) {
            ObjectIterator<Int2DoubleMap.Entry> iter = tmpFeatures[i].map.int2DoubleEntrySet().iterator();
            while (iter.hasNext()) {
                Int2DoubleMap.Entry entry = iter.next();
                int idx = entry.getIntKey();
                double value = entry.getDoubleValue();
                if (map[idx] < trainRatio * labeledData.size())
                    features[i].add(map[idx], value);
            }
        }
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
        tmpFeatures = Utils.LoadLibSVMByFeature(path, featureDimension);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}