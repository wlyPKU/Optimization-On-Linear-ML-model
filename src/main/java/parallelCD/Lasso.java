package parallelCD;

import Utils.LabeledData;
import Utils.Utils;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.DenseVector;
import math.SparseMap;

import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by WLY on 2016/9/4.
 */

//  model       每个线程共享
//  residual    每个线程共享
//  可能会发生冲突
public class Lasso extends model.Lasso {
    private static long start;

    private static double residual[];
    private static DenseVector model;
    private static double featureSquare[];
    private static SparseMap[] features;
    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;
    private static int sampleDimension;

    public class executeRunnable implements Runnable
    {
        int from, to;
        public executeRunnable(int from, int to){
            this.from = from;
            this.to = to;

        }
        public void run() {
            for(int j = from; j < to; j++){
                double oldValue = model.values[j];
                double updateValue = 0;

                ObjectIterator<Int2DoubleMap.Entry> iter =  features[j].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double xj = entry.getDoubleValue();
                    updateValue += xj * residual[idx];
                }
                updateValue /= featureSquare[j];
                model.values[j] += updateValue;
                model.values[j] = Utils.soft_threshold(lambda / featureSquare[j], model.values[j]);

                iter =  features[j].map.int2DoubleEntrySet().iterator();
                double deltaChange = model.values[j] - oldValue;
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double value = entry.getDoubleValue();
                    residual[idx] -= deltaChange * value;
                }
            }
        }
    }

    private void adjustResidual(DenseVector model, double[] residual){
        ObjectIterator<Int2DoubleMap.Entry> iter =  features[featureDimension].map.int2DoubleEntrySet().iterator();
        while (iter.hasNext()) {
            Int2DoubleMap.Entry entry = iter.next();
            int idx = entry.getIntKey();
            double y = entry.getDoubleValue();
            residual[idx] = y;
        }
        for(int j = 0; j < featureDimension; j++){
            iter =  features[j].map.int2DoubleEntrySet().iterator();
            while (iter.hasNext()) {
                Int2DoubleMap.Entry entry = iter.next();
                int idx = entry.getIntKey();
                double value = entry.getDoubleValue();
                residual[idx] -= model.values[j] * value;
            }
        }
    }

    private void trainCore(List<LabeledData> labeledData) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        featureSquare = new double[featureDimension];
        residual = new double[sampleDimension];
        for(int i = 0; i < featureDimension; i++){
            featureSquare[i] = 0;
            for(Double v: features[i].map.values()){
                featureSquare[i] += v * v;
            }
        }

        ObjectIterator<Int2DoubleMap.Entry> iter =  features[featureDimension].map.int2DoubleEntrySet().iterator();
        while (iter.hasNext()) {
            Int2DoubleMap.Entry entry = iter.next();
            int idx = entry.getIntKey();
            double y = entry.getDoubleValue();
            residual[idx] = y;
        }
        DenseVector oldModel = new DenseVector(featureDimension);

        long totalBegin = System.currentTimeMillis();

        long totalIterationTime = 0;

        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model, lambda);
            long startTrain = System.currentTimeMillis();
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
                threadPool.execute(new executeRunnable(from, to));
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

            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("[Information]trainTime " + trainTime);
            totalIterationTime += trainTime;
            System.out.println("[Information]totalTrainTime " + totalIterationTime);
            System.out.println("[Information]totalTime " + (System.currentTimeMillis() - totalBegin));
            System.out.println("[Information]HeapUsed " + ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed()
                    / 1024 / 1024 + "M");
            System.out.println("[Information]MemoryUsed " + (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
                    / 1024 / 1024 + "M");
            adjustResidual(model, residual);
            if(modelType == 1) {
                if (totalIterationTime > maxTimeLimit) {
                    break;
                }
            }else if(modelType == 0){
                if(i > maxIteration){
                    break;
                }
            }else if (modelType == 2){
                if(converge(oldModel, model)){
                    break;
                }
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDimension);
            if(diverge){
                System.out.println("[Warning]Diverge happens!");
                break;
            }
        }
    }

    public static void train(List<LabeledData> labeledData) {
        Lasso lassoModelParallelCD = new Lasso();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        model = new DenseVector(featureDimension);
        Arrays.fill(model.values, 0);
        start = System.currentTimeMillis();
        lassoModelParallelCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.Lasso threadNum FeatureDim SampleDim train_path lambda trainRatio");
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        sampleDimension = Integer.parseInt(argv[2]);
        String path = argv[3];
        lambda = Double.parseDouble(argv[4]);
        trainRatio = 0.5;
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
        System.out.println("[Parameter]SampleDimension " + sampleDimension);
        System.out.println("[Parameter]File Path " + path);
        System.out.println("[Parameter]Lambda " + lambda);
        System.out.println("[Parameter]TrainRatio " + trainRatio);
        System.out.println("[Parameter]TimeLimit " + maxTimeLimit);
        System.out.println("[Parameter]ModelType " + modelType);
        System.out.println("[Parameter]Iteration Limit " + maxIteration);
        System.out.println("------------------------------------");

        long startLoad = System.currentTimeMillis();
        features = Utils.LoadLibSVMByFeature(path, featureDimension, sampleDimension, trainRatio);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}
