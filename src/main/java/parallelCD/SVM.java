package parallelCD;

/**
 * Created by WLY on 2016/9/4.
 */

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.lang.management.ManagementFactory;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

//http://www.tuicool.com/m/articles/RRZvYb
//https://github.com/acharuva/svm_cd/blob/master/svm_cd.py

//https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf
//  model           每个线程共享
//  predictValue    每个线程共享
//  可能会发生冲突
public class SVM extends model.SVM{

    private static double trainRatio = 0.5;
    private static double lambda = 0.5;
    private static int threadNum = 1;
    private static List<LabeledData> trainCorpus;
    private static int featureDimension;

    private static DenseVector model;
    private static double []alpha;
    private static double []Q;

    private static double [][]partModel;

    public class executeRunnable implements Runnable
    {
        int featureFrom, featureTo;
        double C;
        int threadID;
        public executeRunnable(int threadID, double C){
            this.featureFrom = featureDimension * threadID / threadNum;
            this.featureTo= featureDimension * (threadID + 1) / threadNum;
            this.C = C;
            this.threadID = threadID;
        }
        public void run() {
            for (int j = threadID * trainCorpus.size() / threadNum; j < (threadID + 1)* trainCorpus.size() / threadNum; j++) {
                if(Q[j] != 0) {
                    LabeledData labeledData = trainCorpus.get(j);
                    double G = model.dot(labeledData.data) * labeledData.label - 1;
                    double alpha_old = alpha[j];
                    double PG = 0;
                    if (alpha[j] == 0) {
                        PG = Math.min(G, 0);
                    } else if (alpha[j] == C) {
                        PG = Math.max(G, 0);
                    } else if (alpha[j] > 0 && alpha[j] < C) {
                        PG = G;
                    }
                    if (PG != 0) {
                        alpha[j] = Math.min(Math.max(0, alpha[j] - G / Q[j]), C);
                        double deltaAlpha = alpha[j] - alpha_old;
                        if (deltaAlpha != 0) {
                            if (labeledData.data.values == null) {
                                for (Integer idx : labeledData.data.indices) {
                                    model.values[idx] += deltaAlpha * labeledData.label;
                                }
                            } else {
                                for (int i = 0; i < labeledData.data.indices.length; i++) {
                                    int idx = labeledData.data.indices[i];
                                    double value = labeledData.data.values[i];
                                    model.values[idx] += deltaAlpha * labeledData.label * value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public class fixConflictThread implements Runnable
    {
        int from, to, threadID;
        fixConflictThread(int from, int to, int threadID){
            this.from = from;
            this.to= to;
            this.threadID = threadID;
        }
        public void run() {
            Arrays.fill(partModel[threadID], 0);
            for (int j = from; j < to; j++) {
                LabeledData labeledData = trainCorpus.get(j);
                int r = 0;
                for (Integer idx : labeledData.data.indices) {
                    if (labeledData.data.values == null) {
                        partModel[threadID][idx] += alpha[j]* labeledData.label;
                    } else {
                        partModel[threadID][idx] += alpha[j] * labeledData.label * labeledData.data.values[r];
                        r++;
                    }
                }
            }

        }
    }

    private void fixConflictError() {
        ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
        //Coordinate Descent
        for (int threadID = 0; threadID < threadNum; threadID++) {
            int from = trainCorpus.size() * threadID / threadNum;
            int to = trainCorpus.size() * (threadID + 1) / threadNum;
            threadPool.execute(new fixConflictThread(from, to, threadID));
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
        Arrays.fill(model.values, 0);
        for(int i = 0; i < threadNum; i++){
            for(int j = 0; j < featureDimension; j++){
                model.values[j] += partModel[i][j];
            }
        }
    }

    private void trainCore(List<LabeledData> corpus) {
        double startCompute = System.currentTimeMillis();
        //Collections.shuffle(corpus);
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        trainCorpus = corpus.subList(0, end + 1);
        List<LabeledData> testCorpus = corpus.subList(end, size);

        //https://github.com/acharuva/svm_cd/blob/master/svm_cd.py
        Q = new double[trainCorpus.size()];
        int index = 0;
        for(LabeledData l: trainCorpus){
            Q[index] = 0;
            if(l.data.values == null){
                //binary
                Q[index] = l.data.indices.length;
            }else{
                for(double v: l.data.values) {
                    Q[index] += v * v;
                }
            }
            index++;
        }
        alpha = new double[trainCorpus.size()];
        double C = Double.MAX_VALUE;
        if(lambda != 0){
            C = 1.0 / lambda;
        }
        DenseVector oldModel = new DenseVector(model.values.length);
        long totalBegin = System.currentTimeMillis();
        System.out.println("[Prepare]Pre-computation takes " + (System.currentTimeMillis() - startCompute) + " ms totally");
        partModel = new double[threadNum][featureDimension];

        long totalIterationTime = 0;
        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model, lambda);

            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            long startTrain = System.currentTimeMillis();
            //Coordinate Descent
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(threadID, C));
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
            //if(threadNum != 1) {
                fixConflictError();
           // }
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
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            if(diverge){
                System.out.println("[Warning]Diverge happens!");
                break;
            }
        }
    }

    public static void train(List<LabeledData> corpus) {
        SVM svmCD = new SVM();
        model = new DenseVector(featureDimension);
        long start = System.currentTimeMillis();
        svmCD.trainCore(corpus);
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.SVM threadNum dim train_path lambda [trainRatio]");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");
        lambda = Double.parseDouble(argv[3]);
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
        train(corpus);
    }
}
