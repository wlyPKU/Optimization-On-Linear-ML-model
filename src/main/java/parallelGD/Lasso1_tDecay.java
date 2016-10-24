
package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;
import java.util.List;



/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class Lasso1_tDecay extends Lasso{

    static double initalLearningRate = 0.01;
    double decayRate = 1;

    public void setNewLearningRate(){
        learningRate = initalLearningRate / (1.0 + iteration * decayRate);
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.Lasso1_tDecay threadNum dim train_path lambda learningRate [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        initalLearningRate = Double.parseDouble(argv[4]);
        learningRate = initalLearningRate;
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        if(argv.length >= 6){
            trainRatio = Double.parseDouble(argv[5]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }

        Lasso1_tDecay lasso = new Lasso1_tDecay();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector modelOfU = new DenseVector(dim);
        DenseVector modelOfV = new DenseVector(dim);
        start = System.currentTimeMillis();
        lasso.train(corpus, modelOfU, modelOfV);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
