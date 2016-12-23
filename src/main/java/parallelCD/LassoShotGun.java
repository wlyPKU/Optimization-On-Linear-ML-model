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
 * Created by WLY on 2016/9/4.
 */

//  model       每个线程共享
//  residual    每个线程共享
//  可能会发生冲突
public class LassoShotGun extends model.Lasso {

    private static double residual[];
    private static DenseVector model;
    private static double featureSquare[];
    private static SparseVector[] features;
    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;

    private List<LabeledData> trainCorpus;

    public class executeRunnable implements Runnable
    {
        int from, to;
        public executeRunnable(int from, int to){
            this.from = from;
            this.to = to;

        }
        public void run() {
            double oldValue, updateValue, xj;
            int idx;
            /*
            int[] sequence = new int[to - from];
            for(int i = 0; i < sequence.length; i++){
                sequence[i] = i + from;
            }
            Random random = new Random();
            for(int i = 0; i < sequence.length; i++){
                int p = random.nextInt(to - from);
                int tmp = sequence[i];
                sequence[i] = sequence[p];
                sequence[p] = tmp;
            }

            for(int j : sequence){
            */
            for(int j = from; j < to; j++){
                if(featureSquare[j] != 0) {
                    int indices[] = features[j].indices;
                    double values[] = features[j].values;
                    oldValue = model.values[j];
                    updateValue = 0;

                    for(int i = 0; i < features[j].indices.length; i++){
                        idx = indices[i];
                        xj = values[i];
                        updateValue += xj * residual[idx];
                    }
                    updateValue /= featureSquare[j];
                    model.values[j] += updateValue;
                    model.values[j] = Utils.soft_threshold(lambda / featureSquare[j], model.values[j]);
                    double deltaChange = model.values[j] - oldValue;
                    if (deltaChange != 0) {
                        for(int i = 0; i < features[j].indices.length; i++){
                            idx = indices[i];
                            xj = values[i];
                            residual[idx] -= xj * deltaChange;
                        }
                    }
                }
            }
        }
    }

    public class adjustResidualThread implements Runnable
    {
        int from, to;
        adjustResidualThread(int from, int to){
            this.from = from;
            this.to = to;
        }
        public void run() {
            for(int j = from; j < to; j++){
                LabeledData l = trainCorpus.get(j);
                residual[j] = l.label;
                for(int i = 0; i < l.data.indices.length; i++){
                    int index = l.data.indices[i];
                    double value = l.data.values == null? 1: l.data.values[i];
                    residual[j] -= value * model.values[index];
                }
            }
        }
    }

    private void adjustResidual(){
        ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
        for (int threadID = 0; threadID < threadNum; threadID++) {
            int from = trainCorpus.size() * threadID / threadNum;
            int to = trainCorpus.size() * (threadID + 1) / threadNum;
            threadPool.execute(new adjustResidualThread(from, to));
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

    @SuppressWarnings("unused")
    private void shuffle(List<LabeledData> labeledData) {
        Collections.shuffle(labeledData);
    }

    private double calculateMaxLambda(List<LabeledData> trainCorpus){
        double lambdaMax = 0;
        for(LabeledData l: trainCorpus){
            lambdaMax = Math.max(lambdaMax, Math.abs(l.label));
        }
        return lambdaMax;
    }
    private void trainCore(List<LabeledData> labeledData) {
        double startCompute = System.currentTimeMillis();
        shuffle(labeledData);
        SparseMap[] tmpFeatures = Utils.LoadLibSVMFromLabeledData(labeledData, featureDimension, trainRatio);
        features = Utils.generateSpareVector(tmpFeatures);
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        trainCorpus = labeledData.subList(0, testBegin + 1);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        featureSquare = new double[featureDimension];
        residual = new double[trainCorpus.size()];
        for(int i = 0; i < featureDimension; i++){
            featureSquare[i] = 0;
            for(Double v: features[i].values){
                featureSquare[i] += v * v;
            }
        }

        for(int i = 0; i < features[featureDimension].indices.length; i++){
            residual[features[featureDimension].indices[i]] = features[featureDimension].values[i];
        }
        DenseVector oldModel = new DenseVector(featureDimension);

        System.out.println("[Prepare]Pre-computation takes " + (System.currentTimeMillis() - startCompute) + " ms totally");

        long totalBegin = System.currentTimeMillis();

        long totalIterationTime = 0;

        double lambdaMax = calculateMaxLambda(trainCorpus);
        double lambdaMin = lambda;
        //double lambdaChangeIteration = Math.min(1 + trainCorpus.size()/ 2000, maxIteration);
        double lambdaChangeIteration = maxIteration / 2;
        double alpha = Math.pow(lambdaMax/lambdaMin, 1.0/(1.0*lambdaChangeIteration));
        System.out.println("Alpha " + alpha);

        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            if(i < lambdaChangeIteration) {
                lambda = lambdaMin * Math.pow(alpha, lambdaChangeIteration - i);
            }
            System.out.println("[Information]Lambda " + lambda);

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
            //if(threadNum != 1){
                adjustResidual();
            //}
            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("[Information]trainTime " + trainTime);
            totalIterationTime += trainTime;
            System.out.println("[Information]totalTrainTime " + totalIterationTime);
            System.out.println("[Information]totalTime " + (System.currentTimeMillis() - totalBegin));
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
        LassoShotGun lassoModelParallelCD = new LassoShotGun();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        model = new DenseVector(featureDimension);
        Arrays.fill(model.values, 0);
        long start = System.currentTimeMillis();
        lassoModelParallelCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.Lasso threadNum FeatureDimension train_path lambda trainRatio");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
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
