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
public class SVMMultiCore extends model.SVM{
    private static long start;

    private static double trainRatio = 0.5;
    private static double lambda;
    private static int threadNum;
    private static int featureDimension;
    private static DenseVector model;
    private static double []alpha;
    private static double []deltaAlpha;
    private static double []Q;
    private static List<LabeledData> trainCorpus;

    public class executeRunnable implements Runnable
    {
        int localSampleStart, localSampleEnd;
        int threadID;
        double C;
        public executeRunnable(int threadID, double C){
            this.threadID = threadID;
            localSampleStart = trainCorpus.size() * threadID / threadNum;
            localSampleEnd= trainCorpus.size() * (threadID + 1) / threadNum;
            this.C = C;
        }
        public void run() {
            for (int j = localSampleStart; j < localSampleEnd; j++) {
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
                    alpha[j] = Math.min(Math.max(0, alpha[j] - G / (1.5 * Q[j])), C);
                    deltaAlpha[j] = alpha[j] - alpha_old;
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
        deltaAlpha = new double[trainCorpus.size()];
        Arrays.fill(alpha, 0);
        double C = 1.0 / (2.0 * lambda);
        DenseVector oldModel = new DenseVector(model.values.length);
        long totalBegin = System.currentTimeMillis();

        for (int i = 0; ; i ++) {
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            Arrays.fill(model.values, 0);
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
            for (int j = 0; j < trainCorpus.size(); j++) {
                if(deltaAlpha[j] != 0) {
                    LabeledData labeledData = trainCorpus.get(j);
                    int r = 0;
                    for (Integer idx : labeledData.data.indices) {
                        if (labeledData.data.values == null) {
                            model.values[idx] += deltaAlpha[j] * labeledData.label;
                        } else {
                            model.values[idx] += deltaAlpha[j] * labeledData.label * labeledData.data.values[r];
                            r++;
                        }
                    }
                }
            }

            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime " + trainTime + " ");
            testAndSummary(trainCorpus, testCorpus, model, lambda);

            if(converge(oldModel, model)){
                if(earlyStop)

                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );

        }
    }

    public static void train(List<LabeledData> corpus) {
        SVMMultiCore svmCD = new SVMMultiCore();
        model = new DenseVector(featureDimension);
        long start = System.currentTimeMillis();
        svmCD.trainCore(corpus);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.SVMMultiCore threadNum dim train_path lambda [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        lambda = Double.parseDouble(argv[3]);
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
        System.out.println("File Path " + path);
        System.out.println("Lambda " + lambda);
        System.out.println("TrainRatio " + trainRatio);
        System.out.println("TimeLimit " + maxTimeLimit);
        System.out.println("EarlyStop " + earlyStop);
        train(corpus);
    }
}
