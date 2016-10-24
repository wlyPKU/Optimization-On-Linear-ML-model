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
//Ref: http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Pg. 7,14,18
//  数据并行
public class LogisticRegressionDataParallel extends model.LogisticRegression{
    private static long start;

    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static SparseMap[] features;
    private static DenseVector modelOfU;
    private static DenseVector modelOfV;
    private static DenseVector[] localModelOfU;
    private static DenseVector[] localModelOfV;
    private static double predictValue[];

    private static int featureDimension;
    private static int sampleDimension;
    private static List<LabeledData> trainCorpus;
    private static SparseMap[][] featuresSplits;

    private class updateWPositive implements Runnable
    {
        int threadID;
        private updateWPositive(int threadID){
            this.threadID = threadID;

        }
        public void run() {
            localTrain();
        }
        private void localTrain() {
            for(int fIdx = 0; fIdx < featureDimension; fIdx++){
                //First Order L:
                double firstOrderL = 0;
                double oldValue = localModelOfU[threadID].values[fIdx];
                ObjectIterator<Int2DoubleMap.Entry> iter =  featuresSplits[threadID][fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < trainCorpus.size()) {
                        double xj = entry.getDoubleValue();
                        LabeledData l = trainCorpus.get(idx);
                        double tao = 1.0 / (1.0 + Math.exp( -l.label * predictValue[idx]));
                        firstOrderL += l.label * xj * (tao - 1);
                    }
                }

                double Uj = 0.25 * 1 / lambda * trainCorpus.size();
                double updateValue = (1 + firstOrderL) / Uj;
                if(updateValue > localModelOfU[threadID].values[fIdx]){
                    localModelOfU[threadID].values[fIdx] = 0;
                }else{
                    localModelOfU[threadID].values[fIdx] -= updateValue;
                }
                //Update predictValue
                iter =  featuresSplits[threadID][fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < trainCorpus.size()) {
                        double value = entry.getDoubleValue();
                        predictValue[idx] += value * (localModelOfU[threadID].values[fIdx] - oldValue);
                    }
                }
            }
            modelOfU.plusDense(localModelOfU[threadID]);
        }
    }

    private class updateWNegative implements Runnable
    {
        int threadID;
        private updateWNegative(int threadID){
            this.threadID = threadID;
        }
        public void run() {
            for(int fIdx = 0; fIdx < featureDimension; fIdx++){
                //First Order L:
                double firstOrderL = 0;
                double oldValue = localModelOfV[threadID].values[fIdx];
                ObjectIterator<Int2DoubleMap.Entry> iter =  featuresSplits[threadID][fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < trainCorpus.size()) {
                        double xj = entry.getDoubleValue();
                        LabeledData l = trainCorpus.get(idx);
                        double tao = 1.0 / (1.0 + Math.exp(-l.label * predictValue[idx]));
                        firstOrderL += l.label * xj * (tao - 1);
                    }
                }

                double Uj = 0.25 * 1.0 / lambda * trainCorpus.size();
                double updateValue = (1 - firstOrderL) / Uj;
                if(updateValue > localModelOfV[threadID].values[fIdx]){
                    localModelOfV[threadID].values[fIdx] = 0;
                }else{
                    localModelOfV[threadID].values[fIdx] -= updateValue;
                }

                iter =   featuresSplits[threadID][fIdx].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    if(idx < trainCorpus.size()) {
                        double value = entry.getDoubleValue();
                        predictValue[idx] -= value * (localModelOfV[threadID].values[fIdx] - oldValue);
                    }
                }
            }
            modelOfV.plusDense(localModelOfV[threadID]);
        }

    }

    private void trainCore(List<LabeledData> labeledData) {
        int testBegin = (int)(sampleDimension * trainRatio);
        int testEnd = labeledData.size();
        trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        predictValue = new double[trainCorpus.size()];

        DenseVector model = new DenseVector(featureDimension);
        DenseVector oldModel = new DenseVector(featureDimension);

        long totalBegin = System.currentTimeMillis();


        for (int i = 0; ; i ++) {
            for(int idx = 0; idx < trainCorpus.size(); idx++){
                LabeledData l = trainCorpus.get(idx);
                predictValue[idx] = modelOfU.dot(l.data) - modelOfV.dot(l.data);
            }
            Arrays.fill(modelOfU.values, 0);
            Arrays.fill(modelOfV.values, 0);

            long startTrain = System.currentTimeMillis();
            //Update w+
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new updateWPositive(threadID));
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
                newThreadPool.execute(new updateWNegative(threadID));
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

            modelOfU.allDividedBy(threadNum);
            modelOfV.allDividedBy(threadNum);

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

            for(int idx = 0; idx < threadNum; idx++){
                System.arraycopy(modelOfU.values, 0, localModelOfU[idx].values, 0, featureDimension);
                System.arraycopy(modelOfV.values, 0, localModelOfV[idx].values, 0, featureDimension);
            }
            long nowCost = System.currentTimeMillis() - start;
            if(nowCost > 300000) {
                break;
                //break;
            }
        }
    }


    public static void train(List<LabeledData> labeledData) {
        LogisticRegressionDataParallel lrSCD = new LogisticRegressionDataParallel();
        //http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf 3197-3200+
        modelOfU = new DenseVector(featureDimension);
        modelOfV = new DenseVector(featureDimension);
        localModelOfU = new DenseVector[threadNum];
        localModelOfV = new DenseVector[threadNum];
        for(int i = 0; i < threadNum; i++){
            localModelOfU[i] = new DenseVector(featureDimension);
            localModelOfV[i] = new DenseVector(featureDimension);
        }
        Arrays.fill(modelOfU.values, 0);
        Arrays.fill(modelOfV.values, 0);
        start = System.currentTimeMillis();
        lrSCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.LogisticRegressionDataParallel threadNum FeatureDim SampleDim train_path lamda trainRatio");
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        sampleDimension= Integer.parseInt(argv[2]);
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
        features = Utils.LoadLibSVMByFeature(path, featureDimension, sampleDimension, trainRatio);
        featuresSplits = Utils.LoadLibSVMByFeatureSplit(path, featureDimension, sampleDimension, trainRatio, threadNum);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDimension);

        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}