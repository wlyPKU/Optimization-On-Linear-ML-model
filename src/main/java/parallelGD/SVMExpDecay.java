package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.List;

/**
 * Created by WLY on 2016/9/4.
 */
public class SVMExpDecay extends SVM{

    static double initalLearningRate = 0.01;
    double exponentialDecayRate = 0.025;

    public void setNewLearningRate(){
        learningRate = initalLearningRate / Math.exp(iteration *exponentialDecayRate);
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.SVMExpDecay threadNum dim train_path lambda initalLearningRate [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        lambda = Double.parseDouble(argv[3]);
        initalLearningRate = Double.parseDouble(argv[4]);
        learningRate = initalLearningRate;
        if(argv.length >= 6){
            trainRatio = Double.parseDouble(argv[5]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }

        SVMExpDecay svm = new SVMExpDecay();
        DenseVector model = new DenseVector(dim);
        start = System.currentTimeMillis();
        svm.train(corpus, model);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
