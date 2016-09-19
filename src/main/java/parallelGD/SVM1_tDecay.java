package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.List;

/**
 * Created by WLY on 2016/9/4.
 */
public class SVM1_tDecay extends SVM{

    double initalLearningRate = 0.01;
    double decayRate = 1;

    public void setNewLearningRate(){
        learningRate = initalLearningRate / (1.0 + iteration * decayRate);
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.SVM1_tDecay threadNum dim train_path lambda [trainRatio]");
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

        SVM1_tDecay svm = new SVM1_tDecay();
        DenseVector model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        svm.train(corpus, model);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
