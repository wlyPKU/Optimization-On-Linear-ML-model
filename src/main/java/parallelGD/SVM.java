package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by WLY on 2016/9/4.
 */
public class SVM extends model.SVM{
    public DenseVector globalModel;
    public static double trainRatio = 0.5;
    public static int threadNum;
    public static double lambda = 0.1;

    public double learningRate = 0.001;
    public int iteration = 0;

    public void setNewLearningRate(){
    }

    public class executeRunnable implements Runnable
    {
        List<LabeledData> localList;
        DenseVector localModel;
        double lambda;
        int globalCorpusSize;
        public executeRunnable(List<LabeledData> list, DenseVector model, double lambda, int globalCorpusSize){
            localList = list;
            localModel = new DenseVector(model.dim);
            System.arraycopy(model.values, 0, localModel.values, 0, model.dim);
            this.lambda = lambda;
            this.globalCorpusSize = globalCorpusSize;
        }
        public void run() {
            sgdOneEpoch(localList, localModel, learningRate, lambda, globalCorpusSize);
            globalModel.plusDense(localModel);
        }
        public void sgdOneEpoch(List<LabeledData> list, DenseVector model,
                                double lr, double lambda, double globalCorpusSize) {
            double modelPenalty = -2 * lr * lambda / globalCorpusSize;
            for (LabeledData labeledData : list) {
                //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf Pg 3.
                /* model pennalty */
                //model.value[i] -= model.value[i] * 2 * lr * lambda / N;
                model.multiplySparse(labeledData.data, modelPenalty);
                double dotProd = model.dot(labeledData.data);
                if (1 - dotProd * labeledData.label > 0) {
                    /* residual pennalty */
                    model.plusGradient(labeledData.data, lr * labeledData.label);
                }
            }
        }
    }

    public void train(List<LabeledData> corpus, DenseVector model) {
        Collections.shuffle(corpus);

        int size = corpus.size();
        int end = (int) (size * trainRatio);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        List<List<LabeledData>> ThreadTrainCorpus = new ArrayList<List<LabeledData>>();
        for(int threadID = 0; threadID < threadNum; threadID++){
            int from = end * threadID / threadNum;
            int to = end * (threadID + 1) / threadNum;
            List<LabeledData> threadCorpus = corpus.subList(from, to);
            ThreadTrainCorpus.add(threadCorpus);
        }

        DenseVector oldModel = new DenseVector(model.values.length);
        globalModel = new DenseVector(model.dim);

        long totalBegin = System.currentTimeMillis();

        for (int i = 0; i < 200; i ++) {
            long startTrain = System.currentTimeMillis();
            System.out.println("learning rate " + learningRate);

            //StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(ThreadTrainCorpus.get(threadID),
                        model, lambda, trainCorpus.size()));
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
            globalModel.allDividedBy(threadNum);
            System.arraycopy(globalModel.values, 0, model.values, 0, model.dim);
            Arrays.fill(globalModel.values, 0);

            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime " + trainTime + " ");
            testAndSummary(trainCorpus, testCorpus, model, lambda);

            if(converge(oldModel, model)){
                //break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );

            iteration++;
            setNewLearningRate();
        }
    }


    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.SVM threadNum dim train_path lambda [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        lambda = Double.parseDouble(argv[3]);
        if(argv.length >= 5){
            trainRatio = Double.parseDouble(argv[4]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }

        SVM svm = new SVM();
        DenseVector model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        svm.train(corpus, model);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
