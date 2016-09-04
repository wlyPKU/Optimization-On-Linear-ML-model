package parallelCD;

/**
 * Created by WLY on 2016/9/4.
 */
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.*;
import Utils.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

//http://www.tuicool.com/m/articles/RRZvYb
//https://github.com/acharuva/svm_cd/blob/master/svm_cd.py
public class SVM extends model.SVM{

    static double trainRatio = 0.5;
    static double lambda;
    static DenseVector model;
    static double []alpha;
    static double []Q;
    static List<LabeledData> trainCorpus;

    public class executeRunnable implements Runnable
    {
        int from, to;
        double C;
        public executeRunnable(int from, int to, double C){
            this.from = from;
            this.to = to;
            this.C = C;
        }
        public void run() {
            localTrain();
        }
        public void localTrain() {
            for (int j = from; j < to; j++) {
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

    public void trainCore(List<LabeledData> corpus, int threadNum) {
        Collections.shuffle(corpus);
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);

        //TODO https://github.com/acharuva/svm_cd/blob/master/svm_cd.py
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
        Arrays.fill(alpha, 0);

        double C = 1.0 / (2.0 * lambda);
        DenseVector oldModel = new DenseVector(model.values.length);
        for (int i = 0; i < 300; i ++) {
            long startTrain = System.currentTimeMillis();
            //Coordinate Descent
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = trainCorpus.size() * threadID / threadNum;
                int to = trainCorpus.size() * (threadID + 1) / threadNum;
                threadPool.execute(new executeRunnable(from, to, C));
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
            long startTest = System.currentTimeMillis();
            double loss = SVMLoss(trainCorpus, model, lambda);
            double trainAuc = auc(trainCorpus, model);
            double testAuc = auc(testCorpus, model);
            double trainAccuracy = accuracy(trainCorpus, model);
            double testAccuracy = accuracy(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc
                    + " trainAccuracy=" + trainAccuracy + " testAccuracy=" + testAccuracy
                    + " trainTime=" + trainTime + " testTime=" + testTime);
            if(converage(oldModel, model)){
                //break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
        }
    }

    public static void train(List<LabeledData> corpus, int threadNum) {
        int dim = corpus.get(0).data.dim;
        SVM svmCD = new SVM();
        model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        svmCD.trainCore(corpus, threadNum);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.SVM threadNum dim train_path lambda [trainRatio]");
        int threadNum = Integer.parseInt(argv[0]);
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
        train(corpus, threadNum);
    }
}
