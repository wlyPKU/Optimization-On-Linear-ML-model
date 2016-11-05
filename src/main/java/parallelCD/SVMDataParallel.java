package parallelCD;

/**
 * Created by WLY on 2016/9/4.
 */

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.Arrays;
import java.util.Collections;
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
public class SVMDataParallel extends model.SVM{
    private static long start;

    private static double trainRatio = 0.5;
    private static double lambda = 0.5;
    private static int threadNum = 1;
    private static List<LabeledData> trainCorpus;
    private static int featureDimension;

    private static DenseVector model;
    private static double []alpha;
    private static double []Q;

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
                LabeledData labeledData = trainCorpus.get(j);
                double G = model.dot(labeledData.data) * labeledData.label - 1;
                double alpha_old = alpha[j];
                double PG = 0;
                if(alpha[j] == 0){
                    PG = Math.min(G, 0);
                }else if(alpha[j] == C){
                    PG = Math.max(G, 0);
                }else if(alpha[j] > 0 && alpha[j] < C){
                    PG = G;
                }
                if(PG != 0) {
                    alpha[j] = Math.min(Math.max(0, alpha[j] - G / Q[j]), C);
                    int r = 0;
                    for (Integer idx : labeledData.data.indices) {
                            if (labeledData.data.values == null) {
                                model.values[idx] += (alpha[j] - alpha_old) * labeledData.label;
                            } else {
                                model.values[idx] += (alpha[j] - alpha_old) * labeledData.label * labeledData.data.values[r];
                                r++;
                            }
                    }
                }
            }
        }
    }

    private void fixConflictError() {
        Arrays.fill(model.values, 0);
        for (int j = 0; j < trainCorpus.size(); j++) {
            LabeledData labeledData = trainCorpus.get(j);
            int r = 0;
            for (Integer idx : labeledData.data.indices) {
                if (labeledData.data.values == null) {
                    model.values[idx] += alpha[j]* labeledData.label;
                } else {
                    model.values[idx] += alpha[j] * labeledData.label * labeledData.data.values[r];
                    r++;
                }

            }
        }
    }
    private void trainCore(List<LabeledData> corpus) {
        Collections.shuffle(corpus);
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        trainCorpus = corpus.subList(0, end);
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
        double C = 1.0 / (2.0 * lambda);
        DenseVector oldModel = new DenseVector(model.values.length);
        long totalBegin = System.currentTimeMillis();

        long totalIterationTime = 0;
        for (int i = 0; ; i ++) {
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
            fixConflictError();
            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("Iteration " + i);
            System.out.println("trainTime " + trainTime + " ");
            totalIterationTime += trainTime;
            System.out.println("totalIterationTime " + totalIterationTime);
            testAndSummary(trainCorpus, testCorpus, model, lambda);

            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );

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
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);

        }
    }

    public static void train(List<LabeledData> corpus) {
        SVMDataParallel svmCD = new SVMDataParallel();
        model = new DenseVector(featureDimension);
        start = System.currentTimeMillis();
        svmCD.trainCore(corpus);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.SVMModelParallel threadNum dim train_path lambda [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
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
        System.out.println("ThreadNum " + threadNum);
        System.out.println("StopDelta " + stopDelta);
        System.out.println("FeatureDimension " + featureDimension);
        System.out.println("File Path " + path);
        System.out.println("Lambda " + lambda);
        System.out.println("TrainRatio " + trainRatio);
        System.out.println("TimeLimit " + maxTimeLimit);
        System.out.println("ModelType " + modelType);
        System.out.println("Iteration Limit " + maxIteration);
        train(corpus);
    }
}
